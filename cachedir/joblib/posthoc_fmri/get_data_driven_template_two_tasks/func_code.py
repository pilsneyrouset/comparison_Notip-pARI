# first line: 35
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
