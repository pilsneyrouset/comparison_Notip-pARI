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
fetch_neurovault(max_images=np.infty, mode='download_new', collection_id=1952)

sys.path.append(script_path)
from posthoc_fmri import compute_bounds, get_data_driven_template_two_tasks
from sanssouci.lambda_calibration import calibrate_jer
from posthoc_fmri import get_processed_input, ari_inference, calibrate_simes, calibrate_shifted_simes
from sanssouci.reference_families import shifted_template, shifted_template_lambda

location = './cachedir'
memory = Memory(location, mmap_mode='r', verbose=0)

seed = 42
alpha = 0.05
B = 1000
n_train = 10000
smoothing_fwhm = 4
k_min = 0
TDP = 0.8
n_jobs = 1

location = './cachedir'
memory = Memory(location, mmap_mode='r', verbose=0)

df_tasks = pd.read_csv(os.path.join(script_path, 'contrast_list2.csv'))

test_task1s, test_task2s = df_tasks['task1'], df_tasks['task2']
task1 = test_task1s[25]
task2 = test_task2s[25]

fmri_input, nifti_masker = get_processed_input(
                                                task1, task2,
                                                smoothing_fwhm=smoothing_fwhm)
stats_, p_values = stats.ttest_1samp(fmri_input, 0)
p = fmri_input.shape[1]
pval0, calibrated_simes_tpl = calibrate_simes(fmri_input, alpha, k_max=p, B=B, n_jobs=n_jobs,
                                              seed=seed)
        
shifted_templates = np.array([shifted_template_lambda(p, p, k_min, lbd) for lbd in np.linspace(0, 1, n_train)])
pari_dicho = calibrate_jer(alpha, shifted_templates,
                           pval0, k_max=p, k_min=k_min)
pval0, pari_calibrated = calibrate_shifted_simes(fmri_input, alpha, B=B, n_jobs=n_jobs, 
                                                 seed=seed, k_min=k_min)

_, region_size_simes = sa.find_largest_region(p_values, calibrated_simes_tpl,
                                              TDP,
                                              nifti_masker)
print("Région trouvée par Calibrated Simes :", region_size_simes)

_, region_size_pari_calibrated = sa.find_largest_region(p_values, pari_calibrated,
                                                        TDP,
                                                        nifti_masker)
print("Région trouvée par pARI calibré :", region_size_pari_calibrated)

_, region_size_pari_dicho = sa.find_largest_region(p_values,
                                                   pari_dicho,
                                                   TDP,
                                                   nifti_masker)
print("Région trouvée par pARI dichotomie:", region_size_pari_dicho)

plt.plot(pari_calibrated, label='pARI calibré')
plt.plot(calibrated_simes_tpl, label='Calibrated Simes')
plt.plot(pari_dicho, label='pARI dichotomie')
plt.legend()
plt.show()
#%%
## Vérification que pARI avec kmin=0 coïncide avec calibrated_simes
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

fetch_neurovault(max_images=np.infty, mode='download_new', collection_id=1952)

sys.path.append(script_path)
from posthoc_fmri import compute_bounds, get_data_driven_template_two_tasks
from sanssouci.lambda_calibration import calibrate_jer
from posthoc_fmri import get_processed_input, ari_inference, calibrate_simes, calibrate_shifted_simes
from sanssouci.reference_families import shifted_template
from numpy.testing import assert_almost_equal

location = './cachedir'
memory = Memory(location, mmap_mode='r', verbose=0)

seed = 42
alpha = 0.05
B = 5
n_train = 10000
k_max = 1000
smoothing_fwhm = 4
k_min = 0
TDP = 0.8
n_jobs = 1

location = './cachedir'
memory = Memory(location, mmap_mode='r', verbose=0)
df_tasks = pd.read_csv(os.path.join(script_path, 'contrast_list2.csv'))
test_task1s, test_task2s = df_tasks['task1'], df_tasks['task2']

for n_train in [10000]:
    simes_regions = []
    shifted_simes_regions = []
    for i in range(len(test_task1s)):
        print(i)
        fmri_input, nifti_masker = get_processed_input(test_task1s[i], test_task2s[i])
        p = fmri_input.shape[1]
        stats_, p_values = stats.ttest_1samp(fmri_input, 0)
        pval0, simes_thr = calibrate_simes(fmri_input, alpha,
                                        k_max=p, B=B, n_jobs=n_jobs, seed=seed)
        
        shifted_templates = np.array([lambd*shifted_template(p, p, k_min=k_min) for lambd in np.linspace(0, 1, n_train)])
        calibrated_shifted_simes_tpl = calibrate_jer(alpha, shifted_templates,
                                                    pval0, k_max=p, # est-ce que c'est le bon p ?
                                                    k_min=k_min)
        # pval0, shifted_template_thr = calibrate_shifted_simes(fmri_input, alpha,
        #                                     B=B, n_jobs=n_jobs, seed=seed, k_min=0)
        _, region_size_simes = sa.find_largest_region(p_values, simes_thr,
                                                            TDP,
                                                            nifti_masker)
        _, region_size_shifted_simes = sa.find_largest_region(p_values, calibrated_shifted_simes_tpl,
                                                            TDP,
                                                            nifti_masker)
        shifted_simes_regions.append(region_size_shifted_simes)
        simes_regions.append(region_size_simes)
    np.save(f'/home/onyxia/work/Notip/figures/shifted_simes_regions_{n_train}.npy', shifted_simes_regions)
    np.save(f'/home/onyxia/work/Notip/figures/simes_regions_{n_train}.npy', simes_regions)
#%%
import matplotlib.pyplot as plt
import numpy as np

regions_shifted_simes_100 = np.load("/home/onyxia/work/Notip/figures/shifted_simes_regions_100.npy")
regions_shifted_simes_1000 = np.load("/home/onyxia/work/Notip/figures/shifted_simes_regions_1000.npy")
regions_shifted_simes_5000 = np.load("/home/onyxia/work/Notip/figures/shifted_simes_regions_5000.npy")
regions_shifted_simes_10000 = np.load("/home/onyxia/work/Notip/figures/shifted_simes_regions_10000.npy")
regions_simes = np.load("/home/onyxia/work/Notip/figures/simes_regions_100.npy")

plt.figure()
plt.plot(regions_shifted_simes_100, label='pARI with n_train=100')
plt.plot(regions_shifted_simes_1000, label='pARI with n_train=1000')
plt.plot(regions_shifted_simes_5000, label='pARI with n_train=5000')
plt.plot(regions_shifted_simes_10000, label='pARI with n_train=10000')
plt.plot(regions_simes, label='Simes')
plt.legend()
plt.show()