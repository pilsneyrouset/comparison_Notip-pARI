from joblib import Memory
from nilearn.datasets import fetch_neurovault
from nilearn._utils import check_niimg_3d
from nilearn._utils.niimg import safe_get_data
from scipy.stats import norm, stats
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

# Set up paths and ensure figure directory exists
script_path = os.path.dirname(__file__)
fig_path_ = os.path.abspath(os.path.join(script_path, os.pardir))
fig_path = os.path.join(fig_path_, 'figures')
os.makedirs(fig_path, exist_ok=True)
sys.path.append(os.path.abspath(os.path.join(script_path, '..')))

from tools.posthoc_fmri import get_processed_input

# Parameters
seed = 42
alpha = 0.05
B = 10000
n_train = 10000
smoothing_fwhm = 4
k_max = 1000
delta = 27
n_jobs = 1

def FDP(TDP):
    return 1 - TDP

# Download NeuroVault dataset
fetch_neurovault(max_images=np.inf, mode='download_new', collection_id=1952)
location = './cachedir'
memory = Memory(location, mmap_mode='r', verbose=0)

# Load dataset task list
df_tasks = pd.read_csv(os.path.join(script_path, 'contrast_list2.csv'))
test_task1s, test_task2s = df_tasks['task1'], df_tasks['task2']

for i in range(len(test_task1s)):
    task1 = test_task1s[i]
    task2 = test_task2s[i]

    # Preprocess fMRI data
    fmri_input, nifti_masker = get_processed_input(task1, task2, smoothing_fwhm=smoothing_fwhm)
    stats_, p_values = stats.ttest_1samp(fmri_input, 0)
    z_vals = norm.isf(p_values)
    z_map = nifti_masker.inverse_transform(z_vals)

    # Ensure 3D image and extract data
    stat_img = check_niimg_3d(z_map)
    stat_map_ = safe_get_data(stat_img)

    # Count voxels above thresholds
    z_thresholds = [3, 3.5, 4, 4.5]
    voxel_counts = {}
    for z in z_thresholds:
        count = np.sum(stat_map_ > z)
        voxel_counts[z] = count

    # Set up ticks for secondary axis
    z_max = int(np.floor(np.max(stat_map_)))
    z_ticks = np.arange(1, z_max + 1)
    k_ticks = [np.sum(stat_map_ > z) for z in z_ticks]
    z_labels = [str(z) if (z % 2 == 1 or z in [2, 4]) else "" for z in z_ticks]

    # Load precomputed TDP data
    TDP_ARI = np.load(f'task{i}/TDP_ARI_{alpha}.npy')
    TDP_Notip = np.load(f'task{i}/TDP_Notip_{alpha}.npy')
    TDP_pARI = np.load(f'task{i}/TDP_pARI_{alpha}.npy')

    # --- Plot TDP Curve ---
    fig, ax = plt.subplots()
    ax.plot(TDP_ARI, label='ARI', color='red', alpha=0.5)
    ax.plot(TDP_Notip, label='Notip', color='green', alpha=0.5)
    ax.plot(TDP_pARI, label='pARI', color='blue', alpha=0.5)
    for (z, count), thresh in zip(sorted(voxel_counts.items()), np.linspace(0.3, 0.9, len(voxel_counts))):
        ax.axvline(x=count, color='purple', linestyle='--', alpha=thresh, label=f'z={z}')
    ax.set_xscale("log")
    ax.set_ylabel("TDP lower bound")
    ax.set_xlabel("k")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend()
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # Secondary x-axis showing z-values
    secax = ax.secondary_xaxis("top")
    secax.set_xticks(k_ticks)
    secax.set_xticks([], minor=True)
    secax.set_xticklabels(z_labels)
    secax.set_xlabel("z-value")
    secax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False, length=5, which='both')
    secax.set_xlim(ax.get_xlim())

    plt.tight_layout()
    plt.savefig(f'task{i}/confidence_curve_TDP_{alpha}_full.pdf')

    # --- Plot FDP Curve ---
    fig, ax = plt.subplots()
    ax.plot(FDP(TDP_ARI), label='ARI', color='red', alpha=0.5)
    ax.plot(FDP(TDP_Notip), label='Notip', color='green', alpha=0.5)
    ax.plot(FDP(TDP_pARI), label='pARI', color='blue', alpha=0.5)
    for (z, count), thresh in zip(sorted(voxel_counts.items()), np.linspace(0.3, 0.9, len(voxel_counts))):
        ax.axvline(x=count, color='purple', linestyle='--', alpha=thresh, label=f'z={z}')
    ax.set_xscale("log")
    ax.set_ylabel("FDP upper bound")
    ax.set_xlabel("k")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend()
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # Secondary x-axis showing z-values
    secax = ax.secondary_xaxis("top")
    secax.set_xticks(k_ticks)
    secax.set_xticks([], minor=True)
    secax.set_xticklabels(z_labels)
    secax.set_xlabel("z-value")
    secax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False, length=5, which='both')
    secax.set_xlim(ax.get_xlim())

    plt.tight_layout()
    plt.savefig(f'task{i}/confidence_curve_FDP_{alpha}_full.pdf')
