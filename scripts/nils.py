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
from posthoc_fmri import get_processed_input, ari_inference
from sanssouci.reference_families import shifted_template

location = './cachedir'
memory = Memory(location, mmap_mode='r', verbose=0)

seed = 42
alpha = 0.05
B = 5
n_train = 1000
k_max = 1000
smoothing_fwhm = 4
k_min = 27
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
learned_templates_kmin = learned_templates.copy()
learned_templates_kmin[:, :k_min] = np.zeros((n_train, k_min))

df_tasks = pd.read_csv(os.path.join(script_path, 'contrast_list2.csv'))

test_task1s, test_task2s = df_tasks['task1'], df_tasks['task2']
task1 = test_task1s[7]
task2 = test_task2s[7]

fmri_input, nifti_masker = get_processed_input(
                                                task1, task2,
                                                smoothing_fwhm=smoothing_fwhm)
stats_, p_values = stats.ttest_1samp(fmri_input, 0)
p = fmri_input.shape[1]
_, region_size_ARI = ari_inference(p_values, TDP, alpha, nifti_masker)
pval0 = sa.get_permuted_p_values_one_sample(fmri_input,
                                            B=B,
                                            seed=seed,
                                            n_jobs=n_jobs)
        
shifted_templates = np.array([lambd*shifted_template(p, p, k_min=k_min) for lambd in np.linspace(0, 1, n_train)])
calibrated_shifted_template = calibrate_jer(alpha, shifted_templates,
                                            pval0, k_max=p, k_min=k_min)
calibrated_tpl = calibrate_jer(alpha, learned_templates_kmin,
                               pval0, k_max, k_min=k_min)
calibrated_tpl_bis = calibrate_jer(alpha, learned_templates,
                                   pval0, k_max, k_min=k_min)

plt.plot(calibrated_shifted_template, color='red',
         label="pARI")
plt.plot(calibrated_tpl, color='blue',
         label="Notip")
plt.legend()
plt.show()

plt.figure()
for i in range(B):
    plt.plot(pval0[i][:50], color='black')
plt.plot(calibrated_shifted_template[:50], color='red',
         label="pARI")
plt.plot(calibrated_tpl_bis[:50], color='green')
plt.plot(calibrated_tpl[:50], color='blue',
         label="Notip")
plt.legend()
plt.show()

_, region_size_notip = sa.find_largest_region(p_values, calibrated_tpl,
                                              TDP,
                                              nifti_masker)
print("Région trouvée par Notip est :", region_size_notip)

_, region_size_pari = sa.find_largest_region(p_values,
                                             calibrated_shifted_template,
                                             TDP,
                                             nifti_masker)
print("Région trouvée par pARI est :", region_size_pari)

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
n_train = 1000
k_max = 1000
smoothing_fwhm = 4
k_min = 0
TDP = 0.8
n_jobs = 1

location = './cachedir'
memory = Memory(location, mmap_mode='r', verbose=0)
df_tasks = pd.read_csv(os.path.join(script_path, 'contrast_list2.csv'))
test_task1s, test_task2s = df_tasks['task1'], df_tasks['task2']

fmri_input, nifti_masker = get_processed_input(test_task1s[0], test_task2s[0])
p = fmri_input.shape[1]
stats_, p_values = stats.ttest_1samp(fmri_input, 0)
pval0, simes_thr = calibrate_simes(fmri_input, alpha,
                                       k_max=p, B=B, n_jobs=n_jobs, seed=seed)
pval0, shifted_template_thr = calibrate_shifted_simes(fmri_input, alpha,
                                        B=B, n_jobs=n_jobs, seed=seed)
plt.plot(simes_thr, label="calibrated_simes")
plt.plot(shifted_template_thr, label="pARI with kmin=0")
plt.legend()
plt.show()
a = simes_thr
b = shifted_template_thr
print(assert_almost_equal(a, b))

# for i in range(len(test_task1s)):
#     print(i)
#     fmri_input, nifti_masker = get_processed_input(test_task1s[i], test_task2s[i])
#     p = fmri_input.shape[1]
#     stats_, p_values = stats.ttest_1samp(fmri_input, 0)
#     pval0, simes_thr = calibrate_simes(fmri_input, alpha,
#                                        k_max=p, B=B, n_jobs=n_jobs, seed=seed)
#     pval0, shifted_template_thr = calibrate_shifted_simes(fmri_input, alpha,
#                                         B=B, n_jobs=n_jobs, seed=seed)
#     print(assert_almost_equal(simes_thr, shifted_template_thr))
#%%
import numpy as np
import matplotlib.pyplot as plt

res_01 = np.load('/home/onyxia/work/Notip/figures/res_01.npy')


def gen_boxplot_data(res):
    idx_ok = np.where(res[0] > 25)[0]  # exclude 3 tasks with trivial sig
    power_change_notip = ((res[0] - res[1]) / res[1]) * 100
    return [power_change_notip[idx_ok]]

    
pos_center = np.array([0])
pos0 = pos_center - 0.6
fig, ax = plt.subplots(figsize=(10, 6))

data_a = gen_boxplot_data(res_01)

for nb in range(len(data_a)):
    for i in range(len(data_a[nb])):
        y = data_a[nb][i]
        x = np.random.normal(pos0[nb], 0.1)
        ax.scatter(x, y, c='#66c2a4', alpha=0.75, marker='v')

ax.set_ylim(-20, +20)
ax.set_xlim(-1, 1)

ax.set_xticks(pos_center)
ax.set_xticklabels(ticks)
ax.set_ylabel('Detection rate variation (%)')
ax.hlines(0, xmin=-1, xmax=1, color='black')
ax.set_title(r'Detection rate variation for $\alpha = 0.05$ and various FDPs')
plt.tight_layout()
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
plt.savefig('/home/onyxia/work/Notip/figures/comparison_Notip_pARI.png')
plt.show()