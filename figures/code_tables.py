import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from joblib import Memory
from scipy import stats
from scipy.stats import norm
import os
from notip.posthoc_fmri import get_processed_input, get_clusters_table_with_TDP
from nilearn.datasets import fetch_neurovault

# Paths setup
script_path = os.path.dirname(__file__)
fig_path_ = os.path.abspath(os.path.join(script_path, os.pardir))
fig_path = os.path.join(fig_path_, 'figures')
os.makedirs(fig_path, exist_ok=True)
sys.path.append(os.path.abspath(os.path.join(script_path, '..')))

# Fetch NeuroVault dataset
fetch_neurovault(max_images=np.inf, mode='download_new', collection_id=1952)

# Cache location
location = './cachedir'
memory = Memory(location, mmap_mode='r', verbose=0)

# Parameters
smoothing_fwhm = 4
seed = 42
z_thresholds = [3, 3.5, 4, 4.5, 5, 5.5]
n_perm = 1000

# Load dataset contrasts
df_tasks = pd.read_csv(os.path.join(script_path, 'contrast_list2.csv'))
test_task1s, test_task2s = df_tasks['task1'], df_tasks['task2']

# Loop over tasks
for i in range(36, len(test_task1s)):
    print('Task number:', i)
    task1 = test_task1s[i]
    task2 = test_task2s[i]
    print(task1, task2)

    path = f'task{i}'
    os.makedirs(path, exist_ok=True)

    # Preprocess fMRI input
    fmri_input, nifti_masker = get_processed_input(task1, task2, smoothing_fwhm=smoothing_fwhm)

    # One-sample t-test and compute z-map
    stats_, p_values = stats.ttest_1samp(fmri_input, 0)
    z_vals = norm.isf(p_values)
    z_map = nifti_masker.inverse_transform(z_vals)

    print('Start of computations')
    for z in z_thresholds:
        print('z:', z)
        df = get_clusters_table_with_TDP(
            z_map,
            fmri_input,
            seed=seed,
            n_permutations=n_perm,
            stat_threshold=z,
            methods=['ARI', 'Notip', 'pARI']
        )
        output_file = os.path.join(path, f'{n_perm}_perm_z_threshold_{z}.csv')
        print(output_file)
        df.to_csv(output_file, index=False)
