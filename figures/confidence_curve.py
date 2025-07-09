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
alpha = 0.1
B = 10000
n_train = 10000
smoothing_fwhm = 4
k_max = 1000
delta = 27
n_jobs = 20

# Fetch data
fetch_neurovault(max_images=np.inf, mode='download_new', collection_id=1952)
sys.path.append(script_path)
location = './cachedir'
memory = Memory(location, mmap_mode='r', verbose=0)

# Données du dataset
df_tasks = pd.read_csv(os.path.join(script_path, 'contrast_list2.csv'))
test_task1s, test_task2s = df_tasks['task1'], df_tasks['task2']

def process_task(i):
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


# ## Données du dataset Localizer
# n_subjects = 30
# data = fetch_localizer_contrasts(
#     ["left vs right button press"],
#     n_subjects,
#     get_tmaps=True,
#     # legacy_format=False
# )

    # smoothing_fwhm = 8.0
    # nifti_masker = NiftiMasker(smoothing_fwhm=smoothing_fwhm)
    # fmri_input = nifti_masker.fit_transform(data["cmaps"])
    # stats_, p_values = stats.ttest_1samp(fmri_input, 0)
    # z_vals_ = norm.isf(p_values)
    # z_map = nifti_masker.inverse_transform(z_vals_)
    # stats_, p_values = stats.ttest_1samp(fmri_input, 0)

    stat_img = check_niimg_3d(z_map)
    stat_map_ = safe_get_data(stat_img)
    p = fmri_input.shape[1]
    stat_map_nonzero = stat_map_[stat_map_ != 0]
    hommel = _compute_hommel_value(stat_map_nonzero, alpha)

    # Calcul du TDP pour ARI
    ari_thr = sa.linear_template(alpha, hommel, hommel)
    TDP_ARI = curve_min_tdp(p_values, ari_thr)
    np.save(f"task{i}/TDP_ARI_{alpha}.npy", TDP_ARI)
    print(f"TDP ARI saved{i}")

    print(f"Calcul de pval0{i}")
    pval0, _ = calibrate_simes(fmri_input, alpha, k_max=k_max, B=B, seed=seed)

    # Entraînement de Notip sur le dataset d'inférence
    training_seed = 23
    train_task1 = test_task1s[i]
    train_task2 = test_task2s[i]
    print(f"Start Notip training{i}")
    learned_templates = get_data_driven_template_two_tasks(
                        train_task1, train_task2, B=n_train, seed=training_seed)
    print(f"End of Notip training{i}")
    # Calcul du TDP pour Notip
    calibrated_tpl = calibrate_jer(alpha, learned_templates, pval0, k_max)
    TDP_notip = curve_min_tdp(p_values, calibrated_tpl)
    np.save(f"task{i}/TDP_Notip_{alpha}.npy", TDP_notip)
    print(f"TDP Notip saved{i}")

    # Calcul du TDP pour pARI
    pval0, pari_thr = calibrate_shifted_simes(fmri_input, alpha, B=B, seed=seed, k_min=delta)
    TDP_pARI = curve_min_tdp(p_values, pari_thr)
    np.save(f"task{i}/TDP_pARI_{alpha}.npy", TDP_pARI)
    print(f"TDP pARI saved{i}")


from joblib import Parallel, delayed

print('n_jobs: ', n_jobs)

Parallel(n_jobs=n_jobs)(
    delayed(process_task)(i) for i in range(len(test_task1s))
)
