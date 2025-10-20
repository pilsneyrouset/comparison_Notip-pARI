import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from joblib import Memory, Parallel, delayed
from scipy import stats
import os
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from nilearn.datasets import fetch_neurovault
from nilearn._utils import check_niimg_3d
from nilearn._utils.niimg import safe_get_data
from scipy.stats import norm
import sanssouci as sa
from sanssouci.post_hoc_bounds import curve_min_tdp
from utils import (
    get_data_driven_template_two_tasks,
    get_processed_input,
    _compute_hommel_value,
    calibrate_shifted_simes,
)

# Paths setup
script_path = os.path.dirname(__file__)
fig_path_ = os.path.abspath(os.path.join(script_path, os.pardir))
sys.path.append(os.path.abspath(os.path.join(script_path, '..')))

# Parameters
seed = 42
alpha = 0.1
B = 10000
n_train = 10000
smoothing_fwhm = 4
k_max = 1000
delta = 27
n_jobs = 20

# Fetch NeuroVault dataset
fetch_neurovault(max_images=np.inf, mode='download_new', collection_id=1952)
sys.path.append(script_path)
location = './cachedir'
memory = Memory(location, mmap_mode='r', verbose=0)

# Load task contrasts from dataset
df_tasks = pd.read_csv(os.path.join(script_path, 'contrast_list2.csv'))
test_task1s, test_task2s = df_tasks['task1'], df_tasks['task2']


def process_task(i):
    """Pipeline to compute ARI, pARI and Notip TDPs for a given task index."""
    task1 = test_task1s[i]
    task2 = test_task2s[i]

    # Preprocess fMRI data
    fmri_input, nifti_masker = get_processed_input(task1, task2, smoothing_fwhm=smoothing_fwhm)

    # One-sample t-test
    stats_, p_values = stats.ttest_1samp(fmri_input, 0)
    z_vals = norm.isf(p_values)
    z_map = nifti_masker.inverse_transform(z_vals)
    stat_img = check_niimg_3d(z_map)
    stat_map_ = safe_get_data(stat_img)
    stat_map_nonzero = stat_map_[stat_map_ != 0]

    # Compute Hommel value
    hommel = _compute_hommel_value(stat_map_nonzero, alpha)

    # === ARI TDP ===
    ari_thr = sa.linear_template(alpha, hommel, hommel)
    TDP_ARI = curve_min_tdp(p_values, ari_thr)
    np.save(f"task{i}/TDP_ARI_{alpha}.npy", TDP_ARI)
    print(f"TDP ARI saved for task: {i}")

    # === pARI TDP ===
    pval0, pari_thr = calibrate_shifted_simes(fmri_input, alpha, B=B, seed=seed, k_min=delta)
    TDP_pARI = curve_min_tdp(p_values, pari_thr)
    np.save(f"task{i}/TDP_pARI_{alpha}.npy", TDP_pARI)
    print(f"TDP pARI saved for task: {i}")

    # === Notip training ===
    training_seed = 23
    print(f"Start Notip training for task: {i}")
    learned_templates = get_data_driven_template_two_tasks(
        task1, task2, B=n_train, seed=training_seed
    )
    print(f"End of Notip training for task: {i}")

    # === Notip TDP ===
    notip_thr = sa.calibrate_jer(alpha, learned_templates, pval0, k_max)
    TDP_notip = curve_min_tdp(p_values, notip_thr)
    np.save(f"task{i}/TDP_Notip_{alpha}.npy", TDP_notip)
    print(f"TDP Notip saved for task: {i}")


print('n_jobs:', n_jobs)

# Wrap the Parallel execution with tqdm_joblib to show a progress bar
with tqdm_joblib(tqdm(desc="Processing tasks", total=len(test_task1s))):
    Parallel(n_jobs=n_jobs)(
        delayed(process_task)(i) for i in range(len(test_task1s))
    )