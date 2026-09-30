import os

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from matplotlib import patches
from nilearn import image, plotting
from scipy import ndimage

# Paths setup
script_path = os.path.dirname(__file__)
repo_path = os.path.abspath(os.path.join(script_path, '..'))
tables_path_ = os.path.join(repo_path, 'supplementary')
results_path_ = os.path.join(repo_path, 'results')

# Parameters
# This figure is specific to contrast 36
id_task = 36
plot_path = os.path.join(
    tables_path_,
    f"contrast{id_task}"
)

# Load task contrasts from dataset
df_tasks = pd.read_csv(os.path.join(script_path, 'contrast_list2.csv'))
test_task1s, test_task2s = df_tasks['task1'], df_tasks['task2']

task1 = test_task1s[id_task]
task2 = test_task2s[id_task]
print("task1:", task1)
print("task2:", task2)

# Load the z-map, computed once during the fit step (compute_thresholds.py)
# instead of redoing the fMRI preprocessing
zmap_path = os.path.join(results_path_, f"zmap_contrast{id_task}.nii.gz")
if not os.path.exists(zmap_path):
    raise FileNotFoundError(f"[ERROR] z-map file not found:\n{zmap_path}")
z_map = nib.load(zmap_path)

# === Clusters ===
file = os.path.join(tables_path_, f"contrast{id_task}", "z_threshold_3.5.csv")
df = pd.read_csv(file)

# keep only main clusters (numeric IDs)
df_main = df[df["Cluster ID"].astype(str).str.fullmatch(r"\d+")].copy()

clusters_to_plot = list(range(1, 17))

df_main = df[
    pd.to_numeric(df["Cluster ID"], errors="coerce").notna()
].copy()

df_main["Cluster ID"] = df_main["Cluster ID"].astype(int)

df_main = df_main[
    df_main["Cluster ID"].isin(clusters_to_plot)
].copy()

# predefined view mapping
# The parameters are specific to Figure 36.
views = {
    1: 'y', 2: 'z', 3: 'z', 4: 'z', 5: 'z', 6: 'z', 7: 'y', 8: 'z',
    9: 'z', 10: 'y', 11: 'x', 12: 'x', 13: 'x', 14: 'y', 15: 'y', 16: 'z'
}

# build clusters dict with view
clusters = {}
for _, row in df_main.iterrows():
    cluster_id = int(row["Cluster ID"])
    coord = (row["X"], row["Y"], row["Z"])
    tdp = (row["TDP (ARI)"], row["TDP (Notip)"], row["TDP (pARI)"])
    size = int(row["Cluster Size (mm3)"])
    view = views.get(cluster_id, 'z')  # default 'z' if missing

    clusters[cluster_id] = {
        'coord': coord,
        'tdp': tdp,
        'size': size,
        'view': view
    }

threshold = 3.5
# The parameters are specific to Figure 36.
target_y = {1: 120, 2: 85, 3: -130, 4: 85, 5: 110, 6: -110, 7: 95, 8: 85, 
            9: 110, 10: -90, 11: -125, 12:110, 13:90, 14: -110, 15: 120, 16:-110}

# Load z-map data
z_data = z_map.get_fdata()

# 1. Binarize the map with threshold
binary_data = (z_data > threshold).astype(int)

# 2. Label all clusters
labeled_data, n_labels = ndimage.label(binary_data)

# 3. Keep only clusters defined in the dictionary
mask_data = np.zeros_like(binary_data)
affine_inv = np.linalg.inv(z_map.affine)

for cl in clusters.values():
    x, y, z = cl['coord']
    # Transform MNI coordinates -> voxel indices
    i, j, k = nib.affines.apply_affine(affine_inv, [x, y, z])
    i, j, k = round(i), round(j), round(k)
    
    if (0 <= i < labeled_data.shape[0] and 
        0 <= j < labeled_data.shape[1] and 
        0 <= k < labeled_data.shape[2]):
        label_val = labeled_data[i, j, k]
        if label_val > 0:
            mask_data[labeled_data == label_val] = 1

# 4. Apply mask to z-map
mask_img = nib.Nifti1Image(mask_data.astype(np.int16), z_map.affine)
masked_z_map = image.math_img("img * mask", img=z_map, mask=mask_img)


# Add cluster labels to plots
def annotate_clusters(display, clusters, target_y):
    transparent_labels = [11, 12, 13, 15]

    # target_y was hand-picked for a specific set of clusters, so a 
    # mismatch indicates that the result table has changed (e.g. 
    # a different B producing more clusters with non trivial signal)
    missing = sorted(set(clusters) - set(target_y))
    if missing:
        raise KeyError(
            f"No manual target_y position for cluster label(s) {missing}. "
            f"target_y was tuned by hand for a specific clustering result; "
            f"getting a different set of cluster labels usually means the "
            f"upstream thresholds/clusters changed (different B?)"
            f"Check results/contrast{{id}}/z_threshold_{{threshold}}.csv before "
            f"adding an entry to target_y for this label."
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

        ax.plot([px, px], [py, final_py], color='black', lw=3)
        va = 'bottom' if upward else 'top'
        text_y = final_py + (5 if upward else -5)

        # Use transparency for some cluster labels
        alpha_val = 0.3 if label in transparent_labels else 1.0

        ax.text(px, text_y, str(label),
                fontsize=45, fontweight='bold',
                color='black', ha='center', va=va,
                alpha=alpha_val)


# Enlarge the colorbar
def enlarge_colorbar(display, fig):
    if hasattr(display, '_cbar') and display._cbar is not None:
        cbar = display._cbar
    else:
        cbar = fig.axes[-1]
    cbar.ax.tick_params(labelsize=35, width=3, length=12)
    cbar.set_label('Z-values', fontsize=32, labelpad=25)
    bbox = cbar.ax.get_position()
    cbar.ax.set_position([bbox.x0 + 0.02, bbox.y0, bbox.width * 2.0, bbox.height * 1.2])

# === Main glass brain plot ===
fig = plt.figure(figsize=(18, 10))
display = plotting.plot_glass_brain(
    masked_z_map,
    display_mode='ortho',
    threshold=threshold,
    figure=fig,
    colorbar=True,
    annotate=False,
    cbar_tick_format='%.2f'
)

cbar_ax = display._cbar.ax  # get colorbar axis

# Cover values below threshold with a white rectangle
rect = patches.Rectangle(
    (0, 0),
    1,
    threshold,
    transform=cbar_ax.transData,
    color='white',
    alpha=0.95,
    zorder=10
)
cbar_ax.add_patch(rect)

annotate_clusters(display, clusters, target_y)
enlarge_colorbar(display, fig)

plt.savefig(os.path.join(plot_path, "brain_plot.pdf"), bbox_inches='tight')