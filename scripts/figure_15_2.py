import numpy as np
from scipy import stats
from tqdm import tqdm
import pandas as pd

import sanssouci as sa
from joblib import Parallel, delayed, Memory
import multiprocessing
from functools import partial

import os
import sys

from nilearn.datasets import fetch_neurovault

script_path = os.path.dirname(__file__)
fig_path_ = os.path.abspath(os.path.join(script_path, os.pardir))
fig_path = os.path.join(fig_path_, 'figures')

sys.path.append(script_path)
from posthoc_fmri import get_processed_input
from posthoc_fmri import ari_inference, get_data_driven_template_two_tasks
from sanssouci.lambda_calibration import calibrate_jer
from sanssouci.reference_families import shifted_template

fetch_neurovault(max_images=np.infty, mode='download_new', collection_id=1952)

num_cores = multiprocessing.cpu_count()

seed = 42
alpha = 0.05
TDP = 0.95
B = 5
n_train = 100
k_max = 1000

location = './cachedir'
memory = Memory(location, mmap_mode='r', verbose=0)

train_task1 = 'task001_vertical_checkerboard_vs_baseline'
train_task2 = 'task001_horizontal_checkerboard_vs_baseline'

get_data_driven_template_two_tasks = memory.cache(
                                    get_data_driven_template_two_tasks)

learned_templates = get_data_driven_template_two_tasks(
                    train_task1, train_task2, B=n_train, seed=seed)

df_tasks = pd.read_csv(os.path.join(script_path, 'contrast_list2.csv'))

test_task1s, test_task2s = df_tasks['task1'], df_tasks['task2']

pvals_perm_tot = np.load(os.path.join(script_path, "pvals_perm_tot.npy"),
                         mmap_mode="r")

p = pvals_perm_tot.shape[2]

k_mins = [1, 3, 9, 27, 45, 81, 100, 150, 243, 300, 729]
TDPs = [0.95, 0.9, 0.8]
alphas = [0.05, 0.1, 0.2]


def compute_regions(
        k_max, pvals_perm, p_values, alpha, TDP, nifti_masker, task_idx, k_min):

    shifted_templates = np.array([lambd*shifted_template(p, p, k_min=k_min) for lambd in np.linspace(0, 1, n_train)])
    calibrated_shifted_simes_tpl = calibrate_jer(alpha, shifted_templates,
                                                 pvals_perm_tot[task_idx], k_max=p,
                                                 k_min=k_min)

    learned_templates_kmin = learned_templates.copy()
    learned_templates_kmin[:, :k_min] = np.zeros((n_train, k_min))
    calibrated_tpl = calibrate_jer(alpha, learned_templates_kmin,
                                   pvals_perm_tot[task_idx], k_max, k_min)

    _, region_size_shifted_simes = sa.find_largest_region(p_values, calibrated_shifted_simes_tpl,
                                                          TDP,
                                                          nifti_masker)

    _, region_size_notip = sa.find_largest_region(p_values, calibrated_tpl,
                                                  TDP,
                                                  nifti_masker)
    return np.array([region_size_shifted_simes, region_size_notip])


for j in range(len(TDPs)):
    TDP = TDPs[j]
    for k in range(len(alphas)):
        alpha = alphas[k]
        for i in tqdm(range(len(test_task1s))):
            fmri_input, nifti_masker = get_processed_input(test_task1s[i],
                                                           test_task2s[i])
            stats_, p_values = stats.ttest_1samp(fmri_input, 0)

            compute_regions_ = partial(compute_regions, k_max=k_max, pvals_perm=pvals_perm_tot,
                                       p_values=p_values, alpha=alpha, TDP=TDP,
                                       nifti_masker=nifti_masker, task_idx=i)
            k_min_curve = Parallel(n_jobs=num_cores)(
                            delayed(compute_regions_)(k_min=k_min) for k_min in k_mins)
            np.save(os.path.join(fig_path,
                    "fig10/kmin_curve_task%d_tdp%.2f_alpha%.2f" % (i, TDP, alpha)),
                    k_min_curve)
