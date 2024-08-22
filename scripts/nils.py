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
from sanssouci.reference_families import shifted_template, shifted_template_lambda, linear_template_kmin

location = './cachedir'
memory = Memory(location, mmap_mode='r', verbose=0)

seed = 42
alpha = 0.05
B = 1000
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
#%%
#Comparaison des kmin
templates = {}

df_tasks = pd.read_csv(os.path.join(script_path, 'contrast_list2.csv'))
test_task1s, test_task2s = df_tasks['task1'], df_tasks['task2']
task1 = test_task1s[25]
task2 = test_task2s[25]

fmri_input, nifti_masker = get_processed_input(

                                                task1, task2,

                                                smoothing_fwhm=smoothing_fwhm)
stats_, p_values = stats.ttest_1samp(fmri_input, 0)
p = fmri_input.shape[1]

for k_min in [0, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]:
    learned_templates_kmin = learned_templates.copy()
    learned_templates_kmin[:, :k_min] = np.zeros((n_train, k_min))
    pval0, simes_tpl = calibrate_simes(fmri_input, alpha, k_max=p, B=B, n_jobs=n_jobs, seed=seed)
    notip_tpl_kmin = calibrate_jer(alpha, learned_templates_kmin, pval0=pval0,
                                   k_max=k_max, k_min=k_min)
    templates[k_min] = notip_tpl_kmin

for k in list(templates.keys()):
    plt.plot(templates[k], label=f'Notip + kmin = {k}') 
plt.legend()
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

dicho_shifted_simes_tpl = calibrate_jer_param(alpha, generate_template=shifted_template_lambda, pval0=pval0, k_max=p, m=p, k_min=k_min, epsilon=0.0001)

plt.plot(calibrated_shifted_simes_tpl, label='pARI statistique pivotale')
plt.plot(dicho_shifted_simes_tpl, label='pARI dichotomie paramétrique')
plt.legend()
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

truncated_simes_tpl1 = calibrate_jer_param(alpha, generate_template=linear_template_kmin, pval0=pval0, k_max=p, m=p, k_min=k_min, epsilon=0.001)
pval0, truncated_simes_tpl2 = calibrate_truncated_simes(fmri_input, alpha, B=B, n_jobs=n_jobs, seed=seed, k_min=k_min)

plt.plot(truncated_simes_tpl1, label='Calibrated Simes tronqué dochotomie paramétrique')
plt.plot(truncated_simes_tpl2, label='Calibrated Simes tronqué statistique pivotale')
plt.legend()
plt.show()