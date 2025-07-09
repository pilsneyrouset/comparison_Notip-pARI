import numpy as np
import matplotlib.pyplot as plt
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
from nilearn.maskers import NiftiMasker
from nilearn._utils import check_niimg_3d
from nilearn._utils.niimg import safe_get_data
from scipy.stats import norm
from nilearn.datasets import fetch_localizer_contrasts

import sys
script_path = os.path.dirname(__file__)
fig_path_ = os.path.abspath(os.path.join(script_path, os.pardir))
fig_path = os.path.join(fig_path_, 'figures')
os.makedirs(fig_path, exist_ok=True)
sys.path.append(os.path.abspath(os.path.join(script_path, '..')))
script_path = os.path.dirname(__file__)
fig_path_ = os.path.abspath(os.path.join(script_path, os.pardir))
fig_path = os.path.join(fig_path_, 'figures')

from scripts.posthoc_fmri import compute_bounds, get_data_driven_template_two_tasks
from sanssouci.lambda_calibration import calibrate_jer, calibrate_jer_param
from scripts.posthoc_fmri import get_processed_input, ari_inference, calibrate_simes, calibrate_shifted_simes, calibrate_truncated_simes, _compute_hommel_value
from sanssouci.reference_families import shifted_template, shifted_template_lambda, linear_template_kmin
from sanssouci.post_hoc_bounds import curve_min_tdp


# Paramètres
seed = 42
alpha = 0.05
B = 10000
n_train = 10000
smoothing_fwhm = 4
k_max = 1000
delta = 27
n_jobs = 1

def FDP(TDP):
    return 1 - TDP

# Fetch data
fetch_neurovault(max_images=np.inf, mode='download_new', collection_id=1952)
sys.path.append(script_path)
location = './cachedir'
memory = Memory(location, mmap_mode='r', verbose=0)

# Données du dataset
df_tasks = pd.read_csv(os.path.join(script_path, 'contrast_list2.csv'))
test_task1s, test_task2s = df_tasks['task1'], df_tasks['task2']

for i in range(len(test_task1s)):
    print("task number :", i)
    task1 = test_task1s[i]
    task2 = test_task2s[i]

    fmri_input, nifti_masker = get_processed_input(task1, 
    task2, smoothing_fwhm=smoothing_fwhm)
    stats_, p_values = stats.ttest_1samp(fmri_input, 0)
    z_vals = norm.isf(p_values)
    z_map = nifti_masker.inverse_transform(z_vals)


    stat_img = check_niimg_3d(z_map)
    stat_map_ = safe_get_data(stat_img)

    z_thresholds = [3, 3.5, 4, 4.5]
    voxel_counts = {}

    for z in z_thresholds:
        count = np.sum(stat_map_ > z)
        voxel_counts[z] = count
        print(f"Nombre de voxels avec z > {z} : {count}")

    TDP_ARI = np.load(f'task{i}/TDP_ARI_{alpha}.npy')
    TDP_Notip = np.load(f'task{i}/TDP_Notip_{alpha}.npy')
    TDP_pARI = np.load(f'task{i}/TDP_pARI_{alpha}.npy')


    base_color = 'blue'
    alphas = np.linspace(0.3, 0.9, len(voxel_counts))

    plt.figure()
    plt.title(rf'$\alpha = {alpha}$')
    plt.plot(TDP_ARI[:10000], label='ARI')
    plt.plot(TDP_Notip[:10000], label='Notip')
    plt.plot(TDP_pARI[:10000], label='pARI')
    for (z, count), seuil in zip(sorted(voxel_counts.items()), alphas):
        plt.axvline(x=count, color=base_color, linestyle='--', alpha=seuil, label=f'Seuil z={z}')
    plt.xscale("log")
    plt.ylabel("Borne sur le TDP")
    plt.xlabel("k")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.savefig(f'task{i}/confidence_curve_TDP_{alpha}.pdf')


    plt.figure()
    plt.title(rf'$\alpha = {alpha}$')
    plt.plot(FDP(TDP_ARI)[:10000], label='ARI')
    plt.plot(FDP(TDP_Notip)[:10000], label='Notip')
    plt.plot(FDP(TDP_pARI)[:10000], label='pARI')
    for (z, count), seuil in zip(sorted(voxel_counts.items()), alphas):
        plt.axvline(x=count, color=base_color, linestyle='--', alpha=seuil, label=f'Seuil z={z}')
    plt.xscale("log")
    plt.ylabel("Borne sur le FDP")
    plt.xlabel("k")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.savefig(f'task{i}/confidence_curve_FDP_{alpha}.pdf')




# ## Données du dataset Localizer
# n_subjects = 30
# data = fetch_localizer_contrasts(
#     ["left vs right button press"],
#     n_subjects,
#     get_tmaps=True,
#     # legacy_format=False
# )

# TDP_ARI = np.load(f'task_notebook/TDP_ARI_{alpha}.npy')
# TDP_Notip = np.load(f'task_notebook/TDP_Notip_{alpha}.npy')
# TDP_pARI = np.load(f'task_notebook/TDP_pARI_{alpha}.npy')

# smoothing_fwhm = 8.0
# nifti_masker = NiftiMasker(smoothing_fwhm=smoothing_fwhm)
# fmri_input = nifti_masker.fit_transform(data["cmaps"])
# stats_, p_values = stats.ttest_1samp(fmri_input, 0)
# z_vals_ = norm.isf(p_values)
# z_map = nifti_masker.inverse_transform(z_vals_)
# stats_, p_values = stats.ttest_1samp(fmri_input, 0)
# stat_img = check_niimg_3d(z_map)
# stat_map_ = safe_get_data(stat_img)

# z_thresholds = [3, 3.5, 4, 4.5]
# voxel_counts = {}

# for z in z_thresholds:
#     count = np.sum(stat_map_ > z)
#     voxel_counts[z] = count
#     print(f"Nombre de voxels avec z > {z} : {count}")

# base_color = 'blue'
# alphas = np.linspace(0.3, 0.9, len(voxel_counts))

# plt.figure()
# plt.title(rf'$\alpha = {alpha}$')
# plt.plot(TDP_ARI[:10000], label='ARI')
# plt.plot(TDP_Notip[:10000], label='Notip')
# plt.plot(TDP_pARI[:10000], label='pARI')
# for (z, count), seuil in zip(sorted(voxel_counts.items()), alphas):
#     plt.axvline(x=count, color=base_color, linestyle='--', alpha=seuil, label=f'Seuil z={z}')
# plt.xscale("log")
# plt.ylabel("Borne sur le TDP")
# plt.xlabel("k")
# plt.grid(True, which="both", linestyle="--", linewidth=0.5)
# plt.legend()
# plt.savefig(f'task_notebook/confidence_curve_TDP_{alpha}.pdf')


# plt.figure()
# plt.title(rf'$\alpha = {alpha}$')
# plt.plot(FDP(TDP_ARI)[:10000], label='ARI')
# plt.plot(FDP(TDP_Notip)[:10000], label='Notip')
# plt.plot(FDP(TDP_pARI)[:10000], label='pARI')
# for (z, count), seuil in zip(sorted(voxel_counts.items()), alphas):
#     plt.axvline(x=count, color=base_color, linestyle='--', alpha=seuil, label=f'Seuil z={z}')
# plt.xscale("log")
# plt.ylabel("Borne sur le FDP")
# plt.xlabel("k")
# plt.grid(True, which="both", linestyle="--", linewidth=0.5)
# plt.legend()
# plt.savefig(f'task_notebook/confidence_curve_FDP_{alpha}.pdf')


