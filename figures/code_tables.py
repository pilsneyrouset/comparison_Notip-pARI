import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from joblib import Memory
from scipy import stats
from scipy.stats import norm

import os

from nilearn.datasets import fetch_neurovault
from notip.posthoc_fmri import get_clusters_table_with_TDP

import sys
script_path = os.path.dirname(__file__)
fig_path_ = os.path.abspath(os.path.join(script_path, os.pardir))
fig_path = os.path.join(fig_path_, 'figures')
os.makedirs(fig_path, exist_ok=True)
sys.path.append(os.path.abspath(os.path.join(script_path, '..')))
from scripts.posthoc_fmri import get_processed_input


# Fetch data
fetch_neurovault(max_images=np.inf, mode='download_new', collection_id=1952)

location = './cachedir'
memory = Memory(location, mmap_mode='r', verbose=0)

smoothing_fwhm = 4

df_tasks = pd.read_csv(os.path.join(script_path, 'contrast_list2.csv'))
test_task1s, test_task2s = df_tasks['task1'], df_tasks['task2']

z_thresholds = [3, 3.5, 4, 4.5, 5, 5.5]

for i in range(len(test_task1s)):
    print('number of task :', i)
    task1 = test_task1s[i]
    task2 = test_task2s[i]
    path = 'task' + str(i) 
    fmri_input, nifti_masker = get_processed_input(
    task1, task2, smoothing_fwhm=smoothing_fwhm)
    stats_, p_values = stats.ttest_1samp(fmri_input, 0)
    z_vals = norm.isf(p_values)
    z_map = nifti_masker.inverse_transform(z_vals)
    os.makedirs(path, exist_ok=True)

    print('computations start')
    for z in z_thresholds:
        print('z :', z)
        df = get_clusters_table_with_TDP(
        z_map, 
        fmri_input, 
        n_permutations=200,
        stat_threshold=z,
        methods=['ARI', 'Notip', 'pARI'])
        output_file = os.path.join(path, f'z_threshold_{z}.csv')
        print(output_file)
        df.to_csv(output_file, index=False)



