"""This package includes tweaked Nilearn functions (orig. author = B.Thirion)
and utilitary functions to use SansSouci on fMRI data (author = A.Blain)

"""
import warnings
import numpy as np
from scipy.stats import norm
from nilearn.input_data import NiftiMasker
from nilearn.image import threshold_img
from nilearn.image.resampling import coord_transform
from nilearn._utils import check_niimg_3d
from nilearn._utils.niimg import safe_get_data
from nilearn.reporting.get_clusters_table import _local_max
from nilearn.datasets import get_data_dirs
from scipy import stats
import sanssouci as sa
import os
import json
import pandas as pd
from tqdm import tqdm
from string import ascii_lowercase
from scipy import ndimage
import sys
from sanssouci.lambda_calibration import get_pivotal_stats_shifted, calibrate_jer
from sanssouci.reference_families import shifted_linear_template
from sanssouci.post_hoc_bounds import min_tdp
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def get_data_driven_template_two_tasks(
        task1, task2, smoothing_fwhm=4,
        collection=1952, B=100, cap_subjects=False, n_jobs=1, seed=None):
    """
    Get (task1 - task2) data-driven template for two Neurovault contrasts

    Parameters
    ----------

    task1 : str
        Neurovault contrast
    task2 : str
        Neurovault contrast
    smoothing_fwhm : float
        smoothing parameter for fMRI data (in mm)
    collection : int
        Neurovault collection ID
    B : int
        number of permutations at training step
    cap_subjects : boolean
        If True, use only the first 15 subjects
    seed : int

    Returns
    -------

    pval0_quantiles : matrix of shape (B, p)
        Learned template (= sorted quantile curves)
    """
    fmri_input, nifti_masker = get_processed_input(task1, task2, smoothing_fwhm=smoothing_fwhm, collection=collection)
    if cap_subjects:
        # Let's compute the permuted p-values
        pval0 = sa.get_permuted_p_values_one_sample(fmri_input[:10, :],
                                                    B=B, seed=seed, n_jobs=n_jobs)
        # Sort to obtain valid template
        pval0_quantiles = np.sort(pval0, axis=0)
    else:
        # Let's compute the permuted p-values
        pval0 = sa.get_permuted_p_values_one_sample(fmri_input, B=B, seed=seed, n_jobs=n_jobs)
        # Sort to obtain valid template
        pval0_quantiles = np.sort(pval0, axis=0)

    return pval0_quantiles


def get_processed_input(task1, task2, smoothing_fwhm=4, collection=1952):
    """
    Get (task1 - task2) processed input for a pair of Neurovault contrasts
    """

    # Localisation des données téléchargées
    data_path = get_data_dirs()[0]
    data_location = os.path.join(data_path, f'neurovault/collection_{collection}')

    # Liste des fichiers JSON de métadonnées
    json_files = [
        os.path.join(data_location, f) for f in os.listdir(data_location)
        if f.endswith(".json") and 'collection_metadata' not in f
    ]

    files_id = []

    for json_path in json_files:
        with open(json_path) as f:
            data = json.load(f)
            if 'relative_path' in data:
                relative_path = data['relative_path']
                full_img_path = os.path.join(data_location, relative_path)
                files_id.append((full_img_path, data['file']))

    subjects1, subjects2 = [], []
    images_task1, images_task2 = [], []

    for full_path, filename in files_id:
        if task1 in filename:
            images_task1.append(full_path)
            subjects1.append(filename.split("base")[-1])
        elif task2 in filename:
            images_task2.append(full_path)
            subjects2.append(filename.split("base")[-1])

    images_task1 = np.array(images_task1)
    images_task2 = np.array(images_task2)

    # Identifier les sujets communs
    common_subjects = sorted(set(subjects1) & set(subjects2))
    indices1 = [subjects1.index(s) for s in common_subjects]
    indices2 = [subjects2.index(s) for s in common_subjects]

    # Appliquer le masque et le lissage
    nifti_masker = NiftiMasker(smoothing_fwhm=smoothing_fwhm)
    all_imgs = np.concatenate([images_task1[indices1], images_task2[indices2]])
    nifti_masker.fit(all_imgs)

    fmri_input1 = nifti_masker.transform(images_task1[indices1])
    fmri_input2 = nifti_masker.transform(images_task2[indices2])
    fmri_input = fmri_input1 - fmri_input2

    return fmri_input, nifti_masker


def get_stat_img(task1, task2, smoothing_fwhm=4, collection=1952):
    """
    Get (task1 - task2) z-values map for two Neurovault contrasts

    Parameters
    ----------

    task1 : str
        Neurovault contrast
    task2 : str
        Neurovault contrast
    smoothing_fwhm : float
        smoothing parameter for fMRI data (in mm)
    collection : int
        Neurovault collection ID

    Returns
    -------

    z_vals_ :
        Unmasked z-values
    """
    fmri_input, nifti_masker = get_processed_input(
        task1, task2, smoothing_fwhm=smoothing_fwhm, collection=collection)
    stats_, p_values = stats.ttest_1samp(fmri_input, 0)
    z_vals = norm.isf(p_values)
    z_vals_ = nifti_masker.inverse_transform(z_vals)

    return z_vals_


def calibrate_simes(fmri_input, alpha, k_max, B=100, n_jobs=1, seed=None):
    """
    Perform calibration using the Simes template

    Parameters
    ----------

    fmri_input : array of shape (n_subjects, p)
        Masked fMRI data
    alpha : float
        Risk level
    k_max : int
        threshold families length
    B : int
        number of permutations at inference step
    n_jobs : int
        number of CPUs used for computation. Default = 1
    seed : int

    Returns
    -------

    pval0 : matrix of shape (B, p)
        Permuted p-values
    simes_thr : list of length k_max
        Calibrated Simes template
    """
    p = fmri_input.shape[1]  # number of voxels

    # Compute the permuted p-values
    pval0 = sa.get_permuted_p_values_one_sample(fmri_input,
                                                B=B,
                                                seed=seed,
                                                n_jobs=n_jobs)

    # Compute pivotal stats and alpha-level quantile
    piv_stat = sa.get_pivotal_stats(pval0, K=k_max)
    lambda_quant = np.quantile(piv_stat, alpha)

    # Compute chosen template
    simes_thr = sa.linear_template(lambda_quant, k_max, p)

    return pval0, simes_thr


def calibrate_shifted_simes(fmri_input, alpha, B=100, n_jobs=1, seed=None, k_min=0):
    """
    Perform calibration using the Simes template

    Parameters
    ----------

    fmri_input : array of shape (n_subjects, p)
        Masked fMRI data
    alpha : float
        Risk level
    B : int
        number of permutations at inference step
    n_jobs : int
        number of CPUs used for computation. Default = 1
    seed : int

    Returns
    -------

    pval0 : matrix of shape (B, p)
        Permuted p-values
    simes_thr : list of length k_max
        Calibrated Simes template
    """
    p = fmri_input.shape[1]  # number of voxels

    # Compute the permuted p-values
    pval0 = sa.get_permuted_p_values_one_sample(fmri_input,
                                                B=B,
                                                seed=seed,
                                                n_jobs=n_jobs)

    # Compute pivotal stats and alpha-level quantile
    piv_stat = get_pivotal_stats_shifted(pval0, k_min=k_min)
    lambda_quant = np.quantile(piv_stat, alpha)
    # Compute chosen template
    shifted_simes_thr = shifted_linear_template(alpha=lambda_quant, k=p, m=p, k_min=k_min)

    return pval0, shifted_simes_thr


def ari_inference(p_values, tdp, alpha, nifti_masker):
    """
    Find largest FDP controlling region using ARI.

    Parameters
    ----------

    p_values : 1D numpy.array
        A 1D numpy array containing all p-values,sorted non-decreasingly
    tdp : float
        True Discovery Proportion (= 1 - FDP)
    alpha : float
        Risk level
    nifti_masker: NiftiMasker
        masker used on current data

    Returns
    -------

    z_unmasked : nifti image of z_values of the FDP controlling region
    region_size_ARI : size of FDP controlling region

    """

    z_vals = norm.isf(p_values)
    hommel = _compute_hommel_value(z_vals, alpha)
    ari_thr = sa.linear_template(alpha, hommel, hommel)
    z_unmasked, region_size_ARI = sa.find_largest_region(p_values, ari_thr,
                                                         tdp,
                                                         nifti_masker)
    return z_unmasked, region_size_ARI


def get_clusters_table_with_TDP(stat_img, fmri_input, stat_threshold=3,
                                alpha=0.05,
                                k_max=1000, n_permutations=1000, cluster_threshold=None,
                                methods=['Notip'],
                                two_sided=False, min_distance=8., n_jobs=2, seed=None, delta=27):
    """Creates pandas dataframe with img cluster statistics.
    Parameters
    ----------
    stat_img : Niimg-like object,
       Statistical image (presumably in z- or p-scale).
    stat_threshold : `float`
        Cluster forming threshold in same scale as `stat_img` (either a
        p-value or z-scale value).
    fmri_input : array of shape (n_subjects, p)
        Masked fMRI data
    learned_templates : array of shape (B_train, p)
        sorted quantile curves computed on training data
    alpha : float
        risk level
    k_max : int
        threshold families length
    B : int
        number of permutations at inference step
    cluster_threshold : `int` or `None`, optional
        Cluster size threshold, in voxels.
    two_sided : `bool`, optional
        Whether to employ two-sided thresholding or to evaluate positive values
        only. Default=False.
    min_distance : `float`, optional
        Minimum distance between subpeaks in mm. Default=8mm.
    Returns
    -------
    df : `pandas.DataFrame`
        Table with peaks, subpeaks and estimated TDP using three methods
        from thresholded `stat_img`. For binary clusters
        (clusters with >1 voxel containing only one value), the table
        reports the center of mass of the cluster,
        rather than any peaks/subpeaks.
    """
    # Replace None with 0
    cluster_threshold = 0 if cluster_threshold is None else cluster_threshold
    # print(cluster_threshold)
    # check that stat_img is niimg-like object and 3D
    stat_img = check_niimg_3d(stat_img)

    stat_map_ = safe_get_data(stat_img)
    # Perform calibration before thresholding
    stat_map_nonzero = stat_map_[stat_map_ != 0]
    hommel = _compute_hommel_value(stat_map_nonzero, alpha)
    ari_thr = sa.linear_template(alpha, hommel, hommel)
    pval0, simes_thr = calibrate_simes(fmri_input, alpha,
                                       k_max=k_max, B=n_permutations, seed=seed)
    learned_templates_ = sa.get_permuted_p_values_one_sample(fmri_input,
                                                             B=n_permutations,
                                                             n_jobs=n_jobs,
                                                             seed=None)
    learned_templates = np.sort(learned_templates_, axis=0)
    notip_thr = calibrate_jer(alpha, learned_templates, pval0, k_max)
    _, pari_thr = calibrate_shifted_simes(fmri_input, alpha, B=n_permutations, seed=seed, k_min=delta)

    # Apply threshold(s) to image
    stat_img = threshold_img(
        img=stat_img,
        threshold=stat_threshold,
        cluster_threshold=cluster_threshold,
        two_sided=two_sided,
        mask_img=None,
        copy=True,
    )

    # If cluster threshold is used, there is chance that stat_map will be
    # modified, therefore copy is needed
    stat_map = safe_get_data(stat_img, ensure_finite=True,
                              copy_data=(cluster_threshold is not None))
    # Define array for 6-connectivity, aka NN1 or "faces"
    conn_mat = np.zeros((3, 3, 3), int)
    conn_mat[1, 1, :] = 1
    conn_mat[1, :, 1] = 1
    conn_mat[:, 1, 1] = 1
    voxel_size = np.prod(stat_img.header.get_zooms())
    signs = [1, -1] if two_sided else [1]
    no_clusters_found = True
    rows = []
    for sign in signs:
        # Flip map if necessary
        temp_stat_map = stat_map * sign

        # Binarize using CDT
        binarized = temp_stat_map > stat_threshold
        binarized = binarized.astype(int)

        # If the stat threshold is too high simply return an empty dataframe
        if np.sum(binarized) == 0:
            warnings.warn(
                'Attention: No clusters with stat {0} than {1}'.format(
                    'higher' if sign == 1 else 'lower',
                    stat_threshold * sign,
                )
            )
            continue

        # Now re-label and create table
        label_map = ndimage.measurements.label(binarized, conn_mat)[0]
        clust_ids = sorted(list(np.unique(label_map)[1:]))
        peak_vals = np.array(
            [np.max(temp_stat_map * (label_map == c)) for c in clust_ids])
        # Sort by descending max value
        clust_ids = [clust_ids[c] for c in (-peak_vals).argsort()]

        for c_id, c_val in enumerate(clust_ids):
            cluster_mask = label_map == c_val
            masked_data = temp_stat_map * cluster_mask
            masked_data_ = masked_data[masked_data != 0]
            # Compute TDP bounds on cluster using our 3 methods
            cluster_p_values = norm.sf(masked_data_)
            ari_tdp = min_tdp(cluster_p_values, ari_thr)
            simes_tdp = min_tdp(cluster_p_values, simes_thr)
            notip_tdp = min_tdp(cluster_p_values, notip_thr)
            cluster_size_mm = int(np.sum(cluster_mask) * voxel_size)
            pari_tdp = min_tdp(cluster_p_values, pari_thr)

            # Get peaks, subpeaks and associated statistics
            subpeak_ijk, subpeak_vals = _local_max(
                masked_data,
                stat_img.affine,
                min_distance=min_distance,
            )
            subpeak_vals *= sign  # flip signs if necessary
            subpeak_xyz = np.asarray(
                coord_transform(
                    subpeak_ijk[:, 0],
                    subpeak_ijk[:, 1],
                    subpeak_ijk[:, 2],
                    stat_img.affine,
                )
            ).tolist()
            subpeak_xyz = np.array(subpeak_xyz).T

            # Only report peak and, at most, top 3 subpeaks.
            n_subpeaks = np.min((len(subpeak_vals), 4))
            for subpeak in range(n_subpeaks):
                if subpeak == 0:
                    if methods == ['ARI', 'Notip', 'pARI']:
                        cols = ['Cluster ID', 'X', 'Y', 'Z', 'Peak Stat', 'Cluster Size (mm3)',
                                    'TDP (ARI)', 'TDP (Notip)', 'TDP (pARI)']
                        row = [
                            c_id + 1,
                            subpeak_xyz[subpeak, 0],
                            subpeak_xyz[subpeak, 1],
                            subpeak_xyz[subpeak, 2],
                            "{0:.2f}".format(subpeak_vals[subpeak]),
                            cluster_size_mm,
                            "{0:.2f}".format(ari_tdp),
                            "{0:.2f}".format(notip_tdp),
                            "{0:.2f}".format(pari_tdp)]
                    else:
                        cols = ['Cluster ID', 'X', 'Y', 'Z', 'Peak Stat', 'Cluster Size (mm3)',
                                'TDP (Notip)']
                        row = [
                            c_id + 1,
                            subpeak_xyz[subpeak, 0],
                            subpeak_xyz[subpeak, 1],
                            subpeak_xyz[subpeak, 2],
                            "{0:.2f}".format(subpeak_vals[subpeak]),
                            cluster_size_mm,
                            "{0:.2f}".format(notip_tdp)]                           
                                    
                else:
                    # Subpeak naming convention is cluster num+letter:
                    # 1a, 1b, etc
                    sp_id = '{0}{1}'.format(
                        c_id + 1,
                        ascii_lowercase[subpeak - 1],
                    )
                    row = [
                        sp_id,
                        subpeak_xyz[subpeak, 0],
                        subpeak_xyz[subpeak, 1],
                        subpeak_xyz[subpeak, 2],
                        "{0:.2f}".format(subpeak_vals[subpeak]),
                        '']
                    
                    row += [''] * len(methods)

                rows += [row]

        # If we reach this point, there are clusters in this sign
        no_clusters_found = False

    if no_clusters_found:
        cols = ['Cluster ID', 'X', 'Y', 'Z', 'Peak Stat', 'Cluster Size (mm3)',
                                    'TDP (ARI)', 'TDP (Notip)', 'TDP (pARI)']
        df = pd.DataFrame(columns=cols)
    else:
        df = pd.DataFrame(columns=cols, data=rows)

    return df


def _compute_hommel_value(z_vals, alpha, verbose=False):
    """Compute the All-Resolution Inference hommel-value"""
    if alpha < 0 or alpha > 1:
        raise ValueError('alpha should be between 0 and 1')
    z_vals_ = - np.sort(- z_vals)
    p_vals = norm.sf(z_vals_)
    n_samples = len(p_vals)

    if len(p_vals) == 1:
        return p_vals[0] > alpha
    if p_vals[0] > alpha:
        return n_samples
    slopes = (alpha - p_vals[: - 1]) / np.arange(n_samples, 1, -1)
    slope = np.max(slopes)
    hommel_value = np.trunc(n_samples + (alpha - slope * n_samples) / slope)
    if verbose:
        try:
            from matplotlib import pyplot as plt
        except ImportError:
            warnings.warn('"verbose" option requires the package Matplotlib.'
                          'Please install it using `pip install matplotlib`.')
        else:
            plt.figure()
            plt.plot(p_vals, 'o')
            plt.plot([n_samples - hommel_value, n_samples], [0, alpha])
            plt.plot([0, n_samples], [0, 0], 'k')
            plt.show(block=False)
    return np.minimum(hommel_value, n_samples)
