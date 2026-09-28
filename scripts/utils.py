"""This package includes tweaked Nilearn functions (orig. author = B.Thirion)
and utilitary functions to use SansSouci on fMRI data (author = A.Blain)

"""
import json
import os
import sys
import time
import warnings
from contextlib import contextmanager
from string import ascii_lowercase

import numpy as np
import pandas as pd
import psutil
import sanssouci as sa
from nilearn._utils import check_niimg_3d
from nilearn._utils.niimg import safe_get_data
from nilearn.datasets import get_data_dirs
from nilearn.image import threshold_img
from nilearn.image.resampling import coord_transform
from nilearn.maskers import NiftiMasker
from nilearn.reporting.get_clusters_table import _local_max
from sanssouci.post_hoc_bounds import min_tdp
from scipy import ndimage, stats
from scipy.stats import norm

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
    fmri_input, _ = get_processed_input(task1, task2, smoothing_fwhm=smoothing_fwhm, collection=collection)
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

    # Location of the downloaded data
    data_path = get_data_dirs()[0]
    data_location = os.path.join(data_path, f'neurovault/collection_{collection}')

    # List of metadata JSON files
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

    # Identify common subjects
    common_subjects = sorted(set(subjects1) & set(subjects2))
    indices1 = [subjects1.index(s) for s in common_subjects]
    indices2 = [subjects2.index(s) for s in common_subjects]

    # Apply masking and smoothing
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
    _, p_values = stats.ttest_1samp(fmri_input, 0)
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
    piv_stat = sa.get_pivotal_stats_shifted(pval0, k_min=k_min)
    lambda_quant = np.quantile(piv_stat, alpha)
    # Compute chosen template
    shifted_simes_thr = sa.shifted_linear_template(alpha=lambda_quant, k=p, m=p, k_min=k_min)

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


# Available methods for get_clusters_table_with_TDP_task:
#   method name -> key in the thresholds .npz
# (the column title in the table is "TDP (<method name>)")
TDP_METHODS = {
    'ARI': 'ari_thr',
    'calibrated Simes': 'pari0_thr',     # pARI with delta=0
    'Notip': 'notip_thr',
    'pARI': 'pari_thr',                  # delta=27
    'pARI1': 'pari1_thr',                # delta=1
}


def get_clusters_table_with_TDP_task(stat_img, thr, stat_threshold=3,
                                cluster_threshold=None,
                                methods=None,
                                two_sided=False, min_distance=8.):
    """Creates pandas dataframe with img cluster statistics.
    Parameters
    ----------
    stat_img : Niimg-like object,
       Statistical image (presumably in z- or p-scale).
    thr : mapping (e.g. the object returned by np.load on a thresholds .npz)
        Threshold curves, as produced by compute_thresholds.py for a given
        task/alpha. Must contain the keys of the requested `methods`
        (see TDP_METHODS).
    stat_threshold : `float`
        Cluster forming threshold in same scale as `stat_img` (either a
        p-value or z-scale value).
    cluster_threshold : `int` or `None`, optional
        Cluster size threshold, in voxels.
    methods : list of str or `None`, optional
        Names of the methods to report, among the keys of TDP_METHODS
        ('ARI', 'calibrated Simes', 'Notip', 'pARI', 'pARI1'). One column
        "TDP (<method>)" is reported per method, in the order of the list.
        Default: ['Notip'].
    two_sided : `bool`, optional
        Whether to employ two-sided thresholding or to evaluate positive values
        only. Default=False.
    min_distance : `float`, optional
        Minimum distance between subpeaks in mm. Default=8mm.
    Returns
    -------
    df : `pandas.DataFrame`
        Table with peaks, subpeaks and estimated TDP (one column per entry
        of `methods`) from thresholded `stat_img`. For binary clusters
        (clusters with >1 voxel containing only one value), the table
        reports the center of mass of the cluster,
        rather than any peaks/subpeaks.
    """
    if methods is None:
        methods = ['Notip']
    unknown = [m for m in methods if m not in TDP_METHODS]
    if unknown:
        raise ValueError(
            f"Unknown method(s) {unknown}, available: {list(TDP_METHODS)}")
    # Replace None with 0
    cluster_threshold = 0 if cluster_threshold is None else cluster_threshold
    # check that stat_img is niimg-like object and 3D
    stat_img = check_niimg_3d(stat_img)

    # Only load the requested threshold curves (KeyError if one is missing)
    thresholds = {m: thr[TDP_METHODS[m]] for m in methods}
    cols = ['Cluster ID', 'X', 'Y', 'Z', 'Peak Stat', 'Cluster Size (mm3)',
            'Number of Voxels'] + [f'TDP ({m})' for m in methods]

    # Apply threshold(s) to image
    stat_img = threshold_img(
        img=stat_img,
        threshold=stat_threshold,
        cluster_threshold=cluster_threshold,
        two_sided=two_sided,
        mask_img=None,
        copy=True,
        copy_header=True
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
                'Attention: No clusters with stat {} than {}'.format(
                    'higher' if sign == 1 else 'lower',
                    stat_threshold * sign,
                )
            )
            continue

        # Now re-label and create table
        label_map = ndimage.measurements.label(binarized, conn_mat)[0]
        clust_ids = sorted(np.unique(label_map)[1:])
        peak_vals = np.array(
            [np.max(temp_stat_map * (label_map == c)) for c in clust_ids])
        # Sort by descending max value
        clust_ids = [clust_ids[c] for c in (-peak_vals).argsort()]

        for c_id, c_val in enumerate(clust_ids):
            cluster_mask = label_map == c_val
            masked_data = temp_stat_map * cluster_mask
            masked_data_ = masked_data[masked_data != 0]
            # Compute TDP bounds on cluster for each requested method
            cluster_p_values = norm.sf(masked_data_)
            tdps = {m: min_tdp(cluster_p_values, thresholds[m])
                    for m in methods}
            cluster_size_mm = int(np.sum(cluster_mask) * voxel_size)
            voxel_number = cluster_size_mm / 27

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
                    row = [
                        c_id + 1,
                        subpeak_xyz[subpeak, 0],
                        subpeak_xyz[subpeak, 1],
                        subpeak_xyz[subpeak, 2],
                        f"{subpeak_vals[subpeak]:.2f}",
                        cluster_size_mm,
                        round(voxel_number)]
                    row += [f"{tdps[m]:.2f}" for m in methods]
                else:
                    # Subpeak naming convention is cluster num+letter:
                    # 1a, 1b, etc
                    sp_id = f'{c_id + 1}{ascii_lowercase[subpeak - 1]}'
                    row = [
                        sp_id,
                        subpeak_xyz[subpeak, 0],
                        subpeak_xyz[subpeak, 1],
                        subpeak_xyz[subpeak, 2],
                        f"{subpeak_vals[subpeak]:.2f}",
                        '',
                        '']
                    
                    row += [''] * len(methods)

                rows += [row]

        # If we reach this point, there are clusters in this sign
        no_clusters_found = False

    if no_clusters_found:
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

# monitoring time and memory spent per step
class Timer:
    def __init__(self):
        self._process = psutil.Process(os.getpid())

    def _mem_mb(self):
        return self._process.memory_info().rss / (1024 ** 2)

    @contextmanager
    def step(self, msg):
        """Wrap a block of code to log its start, its end and its duration.

        Usage:
            with t.step("doing something"):
                ... code ...
        """
        print(f"[start |{self._mem_mb():7.1f} Mo] {msg}", flush=True)
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - t0
            print(f"[{elapsed:6.2f}s |{self._mem_mb():7.1f} Mo] {msg} -- done", flush=True)