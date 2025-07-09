
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from joblib import Memory
from scipy import stats
from scipy.stats import norm
from matplotlib import colors

import os

from nilearn.datasets import fetch_neurovault
from notip.posthoc_fmri import get_clusters_table_with_TDP
from nilearn import plotting


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

# Threshold
threshold = 3.5
contrast_number = 25

# Contraste 
df_tasks = pd.read_csv(os.path.join(script_path, 'contrast_list2.csv'))
test_task1s, test_task2s = df_tasks['task1'], df_tasks['task2']
task1 = test_task1s[contrast_number]
task2 = test_task2s[contrast_number]

fmri_input, nifti_masker = get_processed_input(
task1, task2, smoothing_fwhm=smoothing_fwhm)
stats_, p_values = stats.ttest_1samp(fmri_input, 0)
z_vals = norm.isf(p_values)
z_map = nifti_masker.inverse_transform(z_vals)


# Clusters pour le contraste et le seuil
clusters = {
    1:  {'coord': (-27.0, -94.0,  16.0), 'tdp': (0.96, 0.98, 0.98), 'size': 120771, 'view': 'x'},

    2:  {'coord': ( 57.0, -22.0,  19.0), 'tdp': (0.90, 0.95, 0.95), 'size':  46440, 'view': 'x'},


    3:  {'coord': (-42.0, -10.0,  10.0), 'tdp': (0.82, 0.91, 0.91), 'size':  24867, 'view': 'x'},


    4:  {'coord': (  3.0, -28.0,   7.0), 'tdp': (0.62, 0.73, 0.62), 'size':   5157, 'view': 'y'},


    5:  {'coord': (-18.0, -25.0,  28.0), 'tdp': (0.53, 0.63, 0.00), 'size':   1026, 'view': 'x'},


    6:  {'coord': (  0.0, -25.0,  25.0), 'tdp': (0.82, 0.91, 0.91), 'size':  23598, 'view': 'x'},

    7:  {'coord': (  9.0,  26.0, -14.0), 'tdp': (0.51, 0.72, 0.65), 'size':   5643, 'view': 'y'},


    8:  {'coord': (  3.0, -34.0,   4.0), 'tdp': (0.32, 0.44, 0.00), 'size':    675, 'view': 'x'},


    9:  {'coord': (-18.0, -34.0,  61.0), 'tdp': (0.47, 0.67, 0.68), 'size':   6804, 'view': 'x'},


    10: {'coord': ( -3.0,  20.0,  49.0), 'tdp': (0.48, 0.66, 0.64), 'size':   5967, 'view': 'y'},


    11: {'coord': (-42.0,  23.0,   1.0), 'tdp': (0.62, 0.78, 0.78), 'size':   9963, 'view': 'y'},


    12: {'coord': ( 21.0, -40.0,  64.0), 'tdp': (0.34, 0.52, 0.39), 'size':   3105, 'view': 'y'},


    13: {'coord': ( 57.0,  35.0,  10.0), 'tdp': (0.71, 0.84, 0.84), 'size':  13284, 'view': 'y'},


    14: {'coord': (-63.0, -34.0, -20.0), 'tdp': (0.26, 0.46, 0.25), 'size':   2403, 'view': 'y'},


    15: {'coord': ( 18.0, -52.0, -26.0), 'tdp': (0.27, 0.49, 0.39), 'size':   3294, 'view': 'y'},


    16: {'coord': (  3.0, -28.0,  16.0), 'tdp': (1.00, 1.00, 0.00), 'size':     54, 'view': 'y'},

    17: {'coord': (-45.0, -64.0,  37.0), 'tdp': (0.08, 0.18, 0.00), 'size':   1323, 'view': 'y'},


    18: {'coord': ( -3.0, -31.0,  13.0), 'tdp': (1.00, 1.00, 0.00), 'size':     27, 'view': 'y'},


    19: {'coord': (-45.0,   8.0, -35.0), 'tdp': (0.11, 0.28, 0.14), 'size':   2484, 'view': 'y'},


    20: {'coord': (-18.0, -31.0,  -5.0), 'tdp': (0.05, 0.14, 0.00), 'size':    594, 'view': 'y'},


    21: {'coord': (  6.0, -91.0, -29.0), 'tdp': (0.05, 0.12, 0.00), 'size':   1134, 'view': 'y'},


    22: {'coord': (  3.0, -16.0, -32.0), 'tdp': (0.00, 0.18, 0.00), 'size':    459, 'view': 'y'},


    23: {'coord': ( 12.0,  26.0,   1.0), 'tdp': (0.00, 0.12, 0.00), 'size':    216, 'view': 'y'},


    24: {'coord': (-30.0, -10.0, -35.0), 'tdp': (0.09, 0.16, 0.00), 'size':    864, 'view': 'y'},


    25: {'coord': ( 33.0, -37.0,   1.0), 'tdp': (0.08, 0.15, 0.00), 'size':    351, 'view': 'y'},


    26: {'coord': (-21.0, -34.0,  22.0), 'tdp': (0.00, 0.00, 0.00), 'size':     81, 'view': 'y'},

    27: {'coord': ( -3.0, -16.0,  22.0), 'tdp': (0.00, 0.00, 0.00), 'size':     27, 'view': 'y'},


    28: {'coord': ( 27.0,  -4.0, -23.0), 'tdp': (0.09, 0.40, 0.35), 'size':   3348, 'view': 'y'},


}

# Niveau des traits pour les clusters
target_y = {
    1: 110,
    2: 90,
    3: -80,
    4: -100,
    5: 90,
    6: 110,
    7: -100,
    8: -110
}

# Tracer de la figure
fig = plt.figure(figsize=(16, 9))
display = plotting.plot_glass_brain(
    z_map,
    display_mode='ortho',
    threshold=threshold,
    figure=fig,
    title="Main Plot",
    colorbar=True
)

for label, cl in clusters.items():
    x, y, z = cl['coord']

    for view in ['x', 'y', 'z']:
        if view not in display.axes:
            continue
        ax = display.axes[view].ax

        # Convertir les coordonnées MNI en coordonnées d'affichage
        if view == 'x':  # Sagittal (affiche y vs z)
            px, py = y, z
        elif view == 'y':  # Coronal (affiche x vs z)
            px, py = x, z
        else:  # Axial (affiche x vs y)
            px, py = x, y

        ax.text(px, py, str(label), color='black', fontsize=10,
                ha='center', va='center')

# for label, cl in clusters.items():
#     x, y, z = cl['coord']
#     view = cl['view']
#     ax = display.axes[view].ax

#     if view == 'x':
#         px, py, final_py = y, z, target_y[label]
#     elif view == 'y':
#         px, py, final_py = x, z, target_y[label]
#     else:
#         px, py, final_py = x, y, target_y[label]

#     ymin, ymax = ax.get_ylim()
#     if ymin < ymax:
#         if final_py > ymax: ymax = final_py + 5
#         if final_py < ymin: ymin = final_py - 5
#         ax.set_ylim(ymin, ymax)
#         upward = final_py > py
#     else:
#         if final_py < ymax: ymax = final_py - 5
#         if final_py > ymin: ymin = final_py + 5
#         ax.set_ylim(ymin, ymax)
#         upward = final_py < py

#     ax.plot([px, px], [py, final_py], color='black', lw=2)
#     va = 'bottom' if upward else 'top'
#     text_y = final_py + (3 if upward else -3)
#     ax.text(px, text_y, str(label),
#             fontsize=20, fontweight='bold',
#             color='black', ha='center', va=va)

plt.savefig("main_plot/main_plot_numbers.pdf", bbox_inches='tight')