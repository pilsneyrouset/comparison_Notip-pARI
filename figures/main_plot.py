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

# Threshold et contraste
threshold = 4
contrast_number = 2


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

# Clusters du contraste
clusters = {
    1: {'coord': (6.0, -28.0, 73.0),    'tdp': (0.93, 0.96, 0.90), 'size': 8991, 'view': 'y'},
    2: {'coord': (-6.0, -34.0, 73.0),   'tdp': (0.90, 0.95, 0.89), 'size': 7803, 'view': 'y'},
    3: {'coord': (-12.0, -37.0, -23.0), 'tdp': (0.83, 0.90, 0.78), 'size': 3888, 'view': 'y'},
    4: {'coord': (15.0, -37.0, -23.0),  'tdp': (0.79, 0.90, 0.77), 'size': 3753, 'view': 'y'},
    5: {'coord': (33.0, -22.0, 16.0),   'tdp': (0.57, 0.76, 0.41), 'size': 1458, 'view': 'y'},
    6: {'coord': (-36.0, -28.0, 61.0),  'tdp': (0.68, 0.81, 0.56), 'size': 1971, 'view': 'y'},
    7: {'coord': (-33.0, -22.0, 16.0),  'tdp': (0.14, 0.14, 0.00), 'size': 189,  'view': 'y'},
    8: {'coord': (-30.0, -46.0, -26.0), 'tdp': (0.00, 0.00, 0.00), 'size': 81,   'view': 'z'},
}

# Niveau des traits pour les différents clusters
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


## Tracer de la figure
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

    ax.plot([px, px], [py, final_py], color='black', lw=2)
    va = 'bottom' if upward else 'top'
    text_y = final_py + (3 if upward else -3)
    ax.text(px, text_y, str(label),
            fontsize=20, fontweight='bold',
            color='black', ha='center', va=va)

plt.savefig("main_plot_numbers.pdf", bbox_inches='tight')