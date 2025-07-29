from nilearn.datasets import fetch_localizer_contrasts
import numpy as np
from nilearn.maskers import NiftiMasker
from scipy import stats
from scipy.stats import norm
from notip.posthoc_fmri import get_clusters_table_with_TDP
from nilearn import plotting
import matplotlib.pyplot as plt


n_subjects = 30
data = fetch_localizer_contrasts(
    ["left vs right button press"],
    n_subjects,
    get_tmaps=True,
    legacy_format=False,
)

smoothing_fwhm = 8.0
nifti_masker = NiftiMasker(smoothing_fwhm=smoothing_fwhm)
fmri_input = nifti_masker.fit_transform(data["cmaps"])

# Let's run a one-sample t test on these data
stats_, p_values = stats.ttest_1samp(fmri_input, 0)
# Let's z-transform these p-values into z values
z_vals_ = norm.isf(p_values)
# Let's make this an image by using the inverse_transform method of the masker
z_map = nifti_masker.inverse_transform(z_vals_)
# todo :immediately plot the z_map


clusters = {
    1:  {'coord': (-39.0, -24.0, 57.0), 'tdp': (0.77, 0.85, 0.89), 'size': 15903, 'view': 'y'},
    2:  {'coord': (39.0, -21.0, 54.0), 'tdp': (0.83, 0.89, 0.92), 'size': 21492, 'view': 'y'},
    3:  {'coord': (-18.0, -51.0, -24.0), 'tdp': (0.56, 0.68, 0.72), 'size': 6075, 'view': 'y'},
    4:  {'coord': (21.0, -48.0, -24.0), 'tdp': (0.43, 0.54, 0.47), 'size': 3159, 'view': 'y'},
    5:  {'coord': (45.0, -18.0, 18.0), 'tdp': (0.41, 0.59, 0.65), 'size': 5022, 'view': 'y'},
    6:  {'coord': (9.0, -18.0, 51.0), 'tdp': (0.20, 0.37, 0.07), 'size': 1593, 'view': 'y'},
    7:  {'coord': (-42.0, -18.0, 15.0), 'tdp': (0.02, 0.12, 0.00), 'size': 1107, 'view': 'y'},
    8:  {'coord': (-6.0, -21.0, 48.0), 'tdp': (0.03, 0.13, 0.00), 'size': 1026, 'view': 'y'},
}
# Threshold
threshold = 3.5

# Niveau des traits pour les clusters
target_y = {
    1: 110,
    2: 90,
    3: -80,
    4: -100,
    5: -80,
    6: 110,
    7: -100,
    8: 90
}

# Tracer de la figure
fig = plt.figure(figsize=(16, 9))
display = plotting.plot_glass_brain(
    z_map,
    display_mode='ortho',
    threshold=threshold,
    figure=fig,
    title="Main Plot",
    colorbar=True,
    annotate=False,
    cbar_tick_format='%.2f'
)

# for label, cl in clusters.items():
#     x, y, z = cl['coord']

#     for view in ['x', 'y', 'z']:
#         if view not in display.axes:
#             continue
#         ax = display.axes[view].ax

#         # Convertir les coordonnées MNI en coordonnées d'affichage
#         if view == 'x':  # Sagittal (affiche y vs z)
#             px, py = y, z
#         elif view == 'y':  # Coronal (affiche x vs z)
#             px, py = x, z
#         else:  # Axial (affiche x vs y)
#             px, py = x, y

#         ax.text(px, py, str(label), color='blue', fontsize=10,
#                 ha='center', va='center')



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
            fontsize=28, fontweight='bold',
            color='black', ha='center', va=va)

plt.savefig("main_plot/main_plot.pdf", bbox_inches='tight')