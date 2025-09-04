from nilearn.datasets import fetch_localizer_contrasts
import numpy as np
from nilearn.maskers import NiftiMasker
from scipy import stats
from scipy.stats import norm
from nilearn import plotting
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
import sys
import nibabel as nib
from joblib import Memory
from scipy import stats
import os
from nilearn import image
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
i = 36

task1 = test_task1s[i]
task2 = test_task2s[i]
print(task1, task2)
fmri_input, nifti_masker = get_processed_input(
task1, task2, smoothing_fwhm=smoothing_fwhm)
stats_, p_values = stats.ttest_1samp(fmri_input, 0)
z_vals = norm.isf(p_values)
z_map = nifti_masker.inverse_transform(z_vals)



# === Clusters ===
clusters = {
    1:  {'coord': (-33.0, -94.0, -17.0), 'tdp': (0.38, 0.55, 0.48), 'size': 3213, 'view': 'y'},
    2:  {'coord': (66.0, 2.0, 16.0),     'tdp': (0.38, 0.77, 0.77), 'size': 7425, 'view': 'z'},
    3:  {'coord': (-12.0, -82.0, -8.0),  'tdp': (0.46, 0.79, 0.80), 'size': 8397, 'view': 'z'},
    4:  {'coord': (-6.0, 11.0, 52.0),    'tdp': (0.23, 0.50, 0.49), 'size': 3321, 'view': 'z'},
    5:  {'coord': (45.0, 14.0, 25.0),    'tdp': (0.38, 0.52, 0.46), 'size': 2835, 'view': 'z'},
    6:  {'coord': (12.0, -43.0, -26.0),  'tdp': (0.15, 0.20, 0.00), 'size': 1107, 'view': 'z'},
    7:  {'coord': (39.0, -73.0, 4.0),    'tdp': (0.08, 0.43, 0.42), 'size': 2862, 'view': 'y'},
    8:  {'coord': (-63.0, -34.0, 16.0),  'tdp': (0.46, 0.82, 0.82), 'size': 9585, 'view': 'z'},
    9:  {'coord': (-27.0, -19.0, 4.0),   'tdp': (0.06, 0.06, 0.00), 'size': 837, 'view': 'z'},
    10: {'coord': (36.0, -94.0, -8.0),   'tdp': (0.25, 0.42, 0.30), 'size': 2160, 'view': 'y'},
    11: {'coord': (12.0, -46.0, 1.0),    'tdp': (0.00, 0.00, 0.00), 'size': 783, 'view': 'x'},
    12: {'coord': (15.0, 56.0, 19.0),    'tdp': (0.00, 0.00, 0.00), 'size': 297, 'view': 'x'},
    13: {'coord': (3.0, -25.0, 52.0),    'tdp': (0.00, 0.00, 0.00), 'size': 324, 'view': 'x'},
    14: {'coord': (0.0, -64.0, -14.0),   'tdp': (0.00, 0.25, 0.14), 'size': 1755, 'view': 'y'},
    15: {'coord': (54.0, -22.0, 58.0), 'tdp': (0.00, 0.00, 0.00), 'size': 648, 'view': 'y'},
    16: {'coord': (-45.0, -67.0, 34.0),  'tdp': (0.00, 0.21, 0.13), 'size': 1890, 'view': 'z'},
}

threshold = 3.5
target_y = {1: 120, 2: 85, 3: -130, 4: 85, 5: 110, 6: -110, 7: 95, 8: 85, 9: 110, 10: -90, 11: -125, 12:110, 13:90, 14: -110, 15: 120, 16:-110}

# Charger les données de ton z_map
z_data = z_map.get_fdata()

# 1. Binariser la carte avec ton seuil
binary_data = (z_data > threshold).astype(int)

# 2. Labelliser tous les clusters
labeled_data, n_labels = ndimage.label(binary_data)

# 3. On crée une liste des clusters à garder (correspondant à ton dictionnaire)
# On localise le label de chaque cluster en cherchant le voxel proche de sa coordonnée
mask_data = np.zeros_like(binary_data)
affine_inv = np.linalg.inv(z_map.affine)

for idx, cl in clusters.items():
    x, y, z = cl['coord']
    # Transformer coord MNI -> index voxel
    i, j, k = nib.affines.apply_affine(affine_inv, [x, y, z])
    i, j, k = int(round(i)), int(round(j)), int(round(k))
    
    if (0 <= i < labeled_data.shape[0] and 
        0 <= j < labeled_data.shape[1] and 
        0 <= k < labeled_data.shape[2]):
        label_val = labeled_data[i, j, k]
        if label_val > 0:
            mask_data[labeled_data == label_val] = 1

# 4. Créer une image masque et appliquer à la carte Z
mask_img = nib.Nifti1Image(mask_data.astype(np.int16), z_map.affine)
masked_z_map = image.math_img("img * mask", img=z_map, mask=mask_img)


# Fonction pour ajouter les labels des clusters
def annotate_clusters(display, clusters, target_y):
    transparent_labels = [11, 12, 13, 15]
    for label, cl in clusters.items():
        x, y, z = cl['coord']
        view = cl['view']
        ax = display.axes[view].ax

        if view == 'x':
            px, py, final_py = y, z, target_y[label]
        elif view == 'y':
            px, py, final_py = x, z, target_y[label]
        else:
            px, py, final_py = x, y, target_y[label]

        ymin, ymax = ax.get_ylim()
        if ymin < ymax:
            if final_py > ymax: ymax = final_py + 5
            if final_py < ymin: ymin = final_py - 5
            ax.set_ylim(ymin, ymax)
            upward = final_py > py
        else:
            if final_py < ymax: ymax = final_py - 5
            if final_py > ymin: ymin = final_py + 5
            ax.set_ylim(ymin, ymax)
            upward = final_py < py

        ax.plot([px, px], [py, final_py], color='black', lw=3)
        va = 'bottom' if upward else 'top'
        text_y = final_py + (5 if upward else -5)

        # Définir alpha en fonction du label
        alpha_val = 0.3 if label in transparent_labels else 1.0

        ax.text(px, text_y, str(label),
                fontsize=45, fontweight='bold',
                color='black', ha='center', va=va,
                alpha=alpha_val)


# Fonction pour agrandir la colorbar
def enlarge_colorbar(display, fig):
    if hasattr(display, '_cbar') and display._cbar is not None:
        cbar = display._cbar
    else:
        cbar = fig.axes[-1]
    cbar.ax.tick_params(labelsize=35, width=3, length=12)
    cbar.set_label('Z-values', fontsize=32, weight='bold', labelpad=25)
    bbox = cbar.ax.get_position()
    cbar.ax.set_position([bbox.x0 + 0.02, bbox.y0, bbox.width * 2.0, bbox.height * 1.2])


# === FIGURE 1 : Carte principale ===
fig = plt.figure(figsize=(18, 10))
display = plotting.plot_glass_brain(
    masked_z_map,
    display_mode='ortho',
    threshold=threshold,
    figure=fig,
    colorbar=True,
    annotate=False,
    cbar_tick_format='%.2f',
)
annotate_clusters(display, clusters, target_y)
enlarge_colorbar(display, fig)
plt.savefig("task36/main_plot.pdf", bbox_inches='tight')
