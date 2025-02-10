import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from joblib import Memory
from scipy import stats
import os
from tqdm import tqdm
import sanssouci as sa
import warnings
from nilearn.datasets import fetch_neurovault

script_path = os.path.dirname(__file__)
fig_path_ = os.path.abspath(os.path.join(script_path, os.pardir))
fig_path = os.path.join(fig_path_, 'figures')

# Fetch data
fetch_neurovault(max_images=np.inf, mode='download_new', collection_id=1952)

sys.path.append(script_path)

from posthoc_fmri import compute_bounds, get_data_driven_template_two_tasks
from sanssouci.lambda_calibration import calibrate_jer, calibrate_jer_param
from posthoc_fmri import get_processed_input, ari_inference, calibrate_simes, calibrate_shifted_simes, calibrate_truncated_simes
from sanssouci.reference_families import shifted_template, shifted_template_lambda, linear_template_kmin
from sanssouci.post_hoc_bounds import curve_min_tdp

seed = 42
alpha = 0.05
B = 10000
n_train = 10000
smoothing_fwhm = 4
k_max = 1000
delta = 27
TDP = 0.8
n_jobs = 1

location = './cachedir'
memory = Memory(location, mmap_mode='r', verbose=0)

train_task1 = 'task001_vertical_checkerboard_vs_baseline'
train_task2 = 'task001_horizontal_checkerboard_vs_baseline'

get_data_driven_template_two_tasks = memory.cache(
                                    get_data_driven_template_two_tasks)

learned_templates = get_data_driven_template_two_tasks(
                    train_task1, train_task2, B=n_train, seed=seed)
                    
print("La phase d'entrainement de Notip est terminée")

if len(sys.argv) > 1:
    n_jobs = int(sys.argv[1])
else:
    n_jobs = 1

df_tasks = pd.read_csv(os.path.join(script_path, 'contrast_list2.csv'))
test_task1s, test_task2s = df_tasks['task1'], df_tasks['task2']


Sk = [100, 200, 500, 1000, 2000, 5000, 10000]

list_TDP_notip = np.zeros((len(df_tasks), len(Sk)))
list_TDP_pARI = np.zeros((len(df_tasks), len(Sk)))
list_TDP_calibrated_simes = np.zeros((len(df_tasks), len(Sk)))

for i in tqdm(range(len(test_task1s))):
    fmri_input, nifti_masker = get_processed_input(
                                                test_task1s[i], test_task2s[i],
                                                smoothing_fwhm=smoothing_fwhm)
    stats_, p_values = stats.ttest_1samp(fmri_input, 0)
    pval0, simes_thr = calibrate_simes(fmri_input, alpha,
                                       k_max=k_max, B=B,
                                       n_jobs=n_jobs, seed=seed)
    TDP_calibrated_simes = curve_min_tdp(p_values, simes_thr)
    list_TDP_calibrated_simes[i, :] = np.array([TDP_calibrated_simes[k] for k in Sk])

    calibrated_tpl = calibrate_jer(alpha, learned_templates,
                                   pval0, k_max)
    TDP_notip = curve_min_tdp(p_values, calibrated_tpl)
    list_TDP_notip[i, :] = np.array([TDP_notip[k] for k in Sk])

    pval0, pari_thr = calibrate_shifted_simes(fmri_input, alpha, B=B, seed=seed, k_min=delta)
    TDP_pARI = curve_min_tdp(p_values, pari_thr)
    list_TDP_pARI[i, :] = np.array([TDP_pARI[k] for k in Sk])

np.save("TDP_notip.npy", list_TDP_notip)
np.save("TDP_pARI.npy", list_TDP_pARI)
np.save("TDP_calibrated_simes.npy", list_TDP_calibrated_simes)
