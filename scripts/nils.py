#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from joblib import Memory
from scipy import stats
import os
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
from sanssouci.reference_families import shifted_linear_template, linear_template_kmin

location = './cachedir'
memory = Memory(location, mmap_mode='r', verbose=0)

seed = 42
alpha = 0.05
B = 10000
n_train = 10000
smoothing_fwhm = 4
k_max = 1000
k_min = 27
TDP = 0.8
n_jobs = 1

train_task1 = 'task001_vertical_checkerboard_vs_baseline'
train_task2 = 'task001_horizontal_checkerboard_vs_baseline'

get_data_driven_template_two_tasks = memory.cache(

                                    get_data_driven_template_two_tasks)
learned_templates = get_data_driven_template_two_tasks(

                    train_task1, train_task2, B=n_train, seed=seed)
print("La phase d'entrainement de Notip est terminée")

df_tasks = pd.read_csv(os.path.join(script_path, 'contrast_list2.csv'))
test_task1s, test_task2s = df_tasks['task1'], df_tasks['task2']
task1 = test_task1s[25]
task2 = test_task2s[25]

fmri_input, nifti_masker = get_processed_input(

                                                task1, task2,

                                                smoothing_fwhm=smoothing_fwhm)
stats_, p_values = stats.ttest_1samp(fmri_input, 0)
p = fmri_input.shape[1]
#%%
k_min = 0
learned_templates_kmin = learned_templates.copy()
learned_templates_kmin[:, :k_min] = np.zeros((n_train, k_min))
pval0, simes_tpl = calibrate_simes(fmri_input, alpha, k_max=p, B=B, n_jobs=n_jobs, seed=seed)
notip_tpl_kmin0 = calibrate_jer(alpha, learned_templates_kmin, pval0=pval0,
                                   k_max=k_max, k_min=k_min)
_, region_notip_0 = sa.find_largest_region(p_values, notip_tpl_kmin0, TDP, nifti_masker)
    
plt.plot(notip_tpl_kmin0, label=f'Notip + kmin = {k_min} /{region_notip_0} voxels') 
plt.legend()
plt.show()
#%%
k_min = 10
learned_templates_kmin = learned_templates.copy()
learned_templates_kmin[:, :k_min] = np.zeros((n_train, k_min))
pval0, simes_tpl = calibrate_simes(fmri_input, alpha, k_max=p, B=B, n_jobs=n_jobs, seed=seed)
notip_tpl_kmin10 = calibrate_jer(alpha, learned_templates_kmin, pval0=pval0,
                                   k_max=k_max, k_min=k_min)
_, region_notip_10 = sa.find_largest_region(p_values, notip_tpl_kmin10, TDP, nifti_masker)
    
plt.plot(notip_tpl_kmin10, label=f'Notip + kmin = {k_min} /{region_notip_10} voxels') 
plt.legend()
plt.show()
#%%
k_min = 20
learned_templates_kmin = learned_templates.copy()
learned_templates_kmin[:, :k_min] = np.zeros((n_train, k_min))
pval0, simes_tpl = calibrate_simes(fmri_input, alpha, k_max=p, B=B, n_jobs=n_jobs, seed=seed)
notip_tpl_kmin20 = calibrate_jer(alpha, learned_templates_kmin, pval0=pval0,
                                   k_max=k_max, k_min=k_min)
_, region_notip_20 = sa.find_largest_region(p_values, notip_tpl_kmin20, TDP, nifti_masker)
    
plt.plot(notip_tpl_kmin20, label=f'Notip + kmin = {k_min} /{region_notip_20} voxels') 
plt.legend()
plt.show()
#%%
k_min = 27
learned_templates_kmin = learned_templates.copy()
learned_templates_kmin[:, :k_min] = np.zeros((n_train, k_min))
pval0, simes_tpl = calibrate_simes(fmri_input, alpha, k_max=p, B=B, n_jobs=n_jobs, seed=seed)
notip_tpl_kmin27 = calibrate_jer(alpha, learned_templates_kmin, pval0=pval0,
                                   k_max=k_max, k_min=k_min)
_, region_notip_27 = sa.find_largest_region(p_values, notip_tpl_kmin27, TDP, nifti_masker)
    
plt.plot(notip_tpl_kmin27, label=f'Notip + kmin = {k_min} /{region_notip_27} voxels') 
plt.legend()
plt.show()
#%%
k_min = 50
learned_templates_kmin = learned_templates.copy()
learned_templates_kmin[:, :k_min] = np.zeros((n_train, k_min))
pval0, simes_tpl = calibrate_simes(fmri_input, alpha, k_max=p, B=B, n_jobs=n_jobs, seed=seed)
notip_tpl_kmin50 = calibrate_jer(alpha, learned_templates_kmin, pval0=pval0,
                                   k_max=k_max, k_min=k_min)
_, region_notip_50 = sa.find_largest_region(p_values, notip_tpl_kmin50, TDP, nifti_masker)
    
plt.plot(notip_tpl_kmin50, label=f'Notip + kmin = {k_min} /{region_notip_50} voxels') 
plt.legend()
plt.show()
#%%
k_min = 100
learned_templates_kmin = learned_templates.copy()
learned_templates_kmin[:, :k_min] = np.zeros((n_train, k_min))
pval0, simes_tpl = calibrate_simes(fmri_input, alpha, k_max=p, B=B, n_jobs=n_jobs, seed=seed)
notip_tpl_kmin100 = calibrate_jer(alpha, learned_templates_kmin, pval0=pval0,
                                   k_max=k_max, k_min=k_min)
_, region_notip_100 = sa.find_largest_region(p_values, notip_tpl_kmin100, TDP, nifti_masker)
    
plt.plot(notip_tpl_kmin100, label=f'Notip + kmin = {k_min} /{region_notip_100} voxels') 
plt.legend()
plt.show()
#%%
k_min = 200
learned_templates_kmin = learned_templates.copy()
learned_templates_kmin[:, :k_min] = np.zeros((n_train, k_min))
pval0, simes_tpl = calibrate_simes(fmri_input, alpha, k_max=p, B=B, n_jobs=n_jobs, seed=seed)
notip_tpl_kmin200 = calibrate_jer(alpha, learned_templates_kmin, pval0=pval0,
                                   k_max=k_max, k_min=k_min)
_, region_notip_200 = sa.find_largest_region(p_values, notip_tpl_kmin200, TDP, nifti_masker)
    
plt.plot(notip_tpl_kmin200, label=f'Notip + kmin = {k_min} /{region_notip_200} voxels') 
plt.legend()
plt.show()
#%%
k_min = 500
learned_templates_kmin = learned_templates.copy()
learned_templates_kmin[:, :k_min] = np.zeros((n_train, k_min))
pval0, simes_tpl = calibrate_simes(fmri_input, alpha, k_max=p, B=B, n_jobs=n_jobs, seed=seed)
notip_tpl_kmin500 = calibrate_jer(alpha, learned_templates_kmin, pval0=pval0,
                                   k_max=k_max, k_min=k_min)
_, region_notip_500 = sa.find_largest_region(p_values, notip_tpl_kmin500, TDP, nifti_masker)
    
plt.plot(notip_tpl_kmin500, label=f'Notip + kmin = {k_min} /{region_notip_500} voxels') 
plt.legend()
plt.show()
#%%
k_min = 1000
learned_templates_kmin = learned_templates.copy()
learned_templates_kmin[:, :k_min] = np.zeros((n_train, k_min))
pval0, simes_tpl = calibrate_simes(fmri_input, alpha, k_max=p, B=B, n_jobs=n_jobs, seed=seed)
notip_tpl_kmin1000 = calibrate_jer(alpha, learned_templates_kmin, pval0=pval0,
                                   k_max=k_max, k_min=k_min)
_, region_notip_1000 = sa.find_largest_region(p_values, notip_tpl_kmin1000, TDP, nifti_masker)
    
plt.plot(notip_tpl_kmin1000, label=f'Notip + kmin = {k_min} /{region_notip_1000} voxels') 
plt.legend()
plt.show()
#%%
k_min = 2000
learned_templates_kmin = learned_templates.copy()
learned_templates_kmin[:, :k_min] = np.zeros((n_train, k_min))
pval0, simes_tpl = calibrate_simes(fmri_input, alpha, k_max=p, B=B, n_jobs=n_jobs, seed=seed)
notip_tpl_kmin2000 = calibrate_jer(alpha, learned_templates_kmin, pval0=pval0,
                                   k_max=k_max, k_min=k_min)
_, region_notip_2000 = sa.find_largest_region(p_values, notip_tpl_kmin2000, TDP, nifti_masker)
    
plt.plot(notip_tpl_kmin2000, label=f'Notip + kmin = {k_min} /{region_notip_2000} voxels') 
plt.legend()
plt.show()
#%%
#Comparaison des différents templates de Notip

plt.title('Templates Notip pour différents kmin')

plt.plot(notip_tpl_kmin0, label=f'Notip + kmin = {0} /{region_notip_0} voxels') 
plt.plot(notip_tpl_kmin10, label=f'Notip + kmin = {10} /{region_notip_10} voxels') 
plt.plot(notip_tpl_kmin20, label=f'Notip + kmin = {20} /{region_notip_20} voxels')
plt.plot(notip_tpl_kmin27, label=f'Notip + kmin = {27} /{region_notip_27} voxels')
plt.plot(notip_tpl_kmin50, label=f'Notip + kmin = {50} /{region_notip_50} voxels') 
plt.plot(notip_tpl_kmin100, label=f'Notip + kmin = {100} /{region_notip_100} voxels')
plt.plot(notip_tpl_kmin200, label=f'Notip + kmin = {200} /{region_notip_200} voxels') 
plt.plot(notip_tpl_kmin500, label=f'Notip + kmin = {500} /{region_notip_500} voxels') 
plt.legend()

plt.savefig('/home/onyxia/work/Notip/figures/Comparaison_Notip.png')
plt.show()
#%%
#Comparaison des différentes méthodes statistiques pivotale vs dichotomie
k_min = 27
df_tasks = pd.read_csv(os.path.join(script_path, 'contrast_list2.csv'))

test_task1s, test_task2s = df_tasks['task1'], df_tasks['task2']
task1 = test_task1s[25]
task2 = test_task2s[25]

fmri_input, nifti_masker = get_processed_input(

                                                task1, task2,

                                                smoothing_fwhm=smoothing_fwhm)
stats_, p_values = stats.ttest_1samp(fmri_input, 0)
p = fmri_input.shape[1]
pval0, calibrated_shifted_simes_tpl = calibrate_shifted_simes(fmri_input, alpha, B=B, n_jobs=n_jobs, seed=seed, k_min=k_min)
_, region_calibrated_shifted_simes_tpl = sa.find_largest_region(p_values, calibrated_shifted_simes_tpl, TDP, nifti_masker)

dicho_shifted_simes_tpl = calibrate_jer_param(alpha, generate_template=shifted_linear_template, pval0=pval0, k_max=p, m=p, k_min=k_min, epsilon=0.0001)
_, region_dicho_shifted_simes_tpl = sa.find_largest_region(p_values, dicho_shifted_simes_tpl, TDP, nifti_masker)

plt.figure(figsize=(15, 8))
plt.suptitile('Templates pARI')

plt.subplot(1, 2, 1)
plt.plot(calibrated_shifted_simes_tpl, label=f'pARI statistique pivotale /{region_calibrated_shifted_simes_tpl} voxels')
plt.plot(dicho_shifted_simes_tpl, label=f'pARI dichotomie paramétrique /{region_dicho_shifted_simes_tpl} voxels')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(calibrated_shifted_simes_tpl[:100], label=f'pARI statistique pivotale /{region_calibrated_shifted_simes_tpl} voxels')
plt.plot(dicho_shifted_simes_tpl[:100], label=f'pARI dichotomie paramétrique /{region_dicho_shifted_simes_tpl} voxels')
plt.legend()

plt.savefig('/home/onyxia/work/Notip/figures/pARI.png')
plt.show()
#%%
#Comparaison avec l'idée de Simes tronqué

k_min = 27
df_tasks = pd.read_csv(os.path.join(script_path, 'contrast_list2.csv'))
test_task1s, test_task2s = df_tasks['task1'], df_tasks['task2']
task1 = test_task1s[25]
task2 = test_task2s[25]

fmri_input, nifti_masker = get_processed_input(

                                                task1, task2,

                                                smoothing_fwhm=smoothing_fwhm)
stats_, p_values = stats.ttest_1samp(fmri_input, 0)
p = fmri_input.shape[1]
pval0, calibrated_simes_tpl = calibrate_simes(fmri_input, alpha, k_max=p, B=B, n_jobs=n_jobs, seed=seed)
_, region_calibrated_simes_tpl = sa.find_largest_region(p_values, calibrated_simes_tpl, TDP, nifti_masker)


truncated_simes_tpl1 = calibrate_jer_param(alpha, generate_template=linear_template_kmin, pval0=pval0, k_max=p, m=p, k_min=k_min, epsilon=0.001)
_, region_truncated_simes_tpl1 = sa.find_largest_region(p_values, truncated_simes_tpl1, TDP, nifti_masker)

pval0, truncated_simes_tpl2 = calibrate_truncated_simes(fmri_input, alpha, B=B, n_jobs=n_jobs, seed=seed, k_min=k_min)
_, region_truncated_simes_tpl2 = sa.find_largest_region(p_values, truncated_simes_tpl2, TDP, nifti_masker)

plt.figure(figsize=(15, 8))
plt.suptitle('Templates Calibrated Simes / Calibrated Simes tronqué')

plt.subplot(1, 2, 1)
plt.plot(truncated_simes_tpl1, label=f'Calibrated Simes tronqué dochotomie paramétrique /{region_truncated_simes_tpl1} voxels')
plt.plot(truncated_simes_tpl2, label=f'Calibrated Simes tronqué statistique pivotale /{region_truncated_simes_tpl2} voxels')
plt.plot(calibrated_simes_tpl, label=f'Calibrated Simes /{region_calibrated_simes_tpl} voxels')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(truncated_simes_tpl1[:100], label=f'Calibrated Simes tronqué dochotomie paramétrique /{region_truncated_simes_tpl1} voxels')
plt.plot(truncated_simes_tpl2[:100], label=f'Calibrated Simes tronqué statistique pivotale /{region_truncated_simes_tpl2} voxels')
plt.plot(calibrated_simes_tpl[:100], label=f'Calibrated Simes /{region_calibrated_simes_tpl} voxels')
plt.legend()

plt.savefig('/home/onyxia/work/Notip/figures/Calibrates_simes.png')
plt.show()