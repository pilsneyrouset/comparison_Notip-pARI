from joblib import Memory
from nilearn.datasets import fetch_neurovault
from nilearn._utils import check_niimg_3d
from nilearn._utils.niimg import safe_get_data
from scipy.stats import norm
from scipy.stats import ttest_1samp
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from utils import get_processed_input
import sanssouci as sa

# Set up paths and ensure figure directory exists
script_path = os.path.dirname(__file__)
fig_path_ = os.path.abspath(os.path.join(script_path, os.pardir))
sys.path.append(os.path.abspath(os.path.join(script_path, '..')))

# Parameters
seed = 42
ALPHAS = [0.05, 0.1]
B = 10000
n_train = 10000
smoothing_fwhm = 4
k_max = 1000
delta = 27
n_jobs = 1


# Download NeuroVault dataset
fetch_neurovault(max_images=np.inf, mode='download_new', collection_id=1952)
location = './cachedir'
memory = Memory(location, mmap_mode='r', verbose=0)

# Load dataset task list
df_tasks = pd.read_csv(os.path.join(script_path, 'contrast_list2.csv'))
test_task1s, test_task2s = df_tasks['task1'], df_tasks['task2']

for alpha in ALPHAS:
    for i in range(len(test_task1s)):
        task1 = test_task1s[i]
        task2 = test_task2s[i]

        # Preprocess fMRI data
        fmri_input, nifti_masker = get_processed_input(task1, task2, smoothing_fwhm=smoothing_fwhm)
        stats_, p_values = ttest_1samp(fmri_input, 0)
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

        threshold_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'results', 'thresholds')
        )

        threshold_path = os.path.join(
            threshold_dir,
            f"thresholds_contrast{i}_alpha{alpha}.npz"
        )

        if not os.path.exists(threshold_path):
            raise FileNotFoundError(f"[ERROR] Threshold file not found:\n{threshold_path}")

        # Load thresholds
        thr = np.load(threshold_path)

        ari_thr = thr["ari_thr"]
        TDP_ARI = sa.curve_min_tdp(p_values, ari_thr)

        pari_thr = thr["pari_thr"]
        TDP_pARI = sa.curve_min_tdp(p_values, pari_thr)

        notip_thr = thr["notip_thr"]
        TDP_Notip = sa.curve_min_tdp(p_values, notip_thr)

        threshold_path1 = os.path.join(
            threshold_dir,
            f"thresholds_contrast{i}_alpha{alpha}_delta1.npz"
        )

        if not os.path.exists(threshold_path1):
            raise FileNotFoundError(f"[ERROR] Threshold file not found:\n{threshold_path1}")

        thr1 = np.load(threshold_path1)

        pari1_thr = thr1["pari_thr"]
        TDP_pARI1 = sa.curve_min_tdp(p_values, pari1_thr)

        threshold_path2 = os.path.join(
            threshold_dir,
            f"thresholds_contrast{i}_alpha{alpha}_calibrated_simes.npz"
        )

        if not os.path.exists(threshold_path2):
            raise FileNotFoundError(f"[ERROR] Threshold file not found:\n{threshold_path1}")

        thr2 = np.load(threshold_path2)

        calibrated_simes_thr = thr2["calibrated_simes_thr"]
        TDP_calibrated_simes = sa.curve_min_tdp(p_values, calibrated_simes_thr)

        # Set up ticks for secondary axis
        z_max = int(np.floor(np.max(stat_map_)))
        z_ticks = list(np.arange(1, z_max + 1))  # + [3.5, 4.5]
        z_ticks = sorted(set(z_ticks))  # éviter les doublons
        k_ticks = [np.sum(stat_map_ > z) for z in z_ticks]
        z_labels = [str(z) if (z % 2 == 1 or z in [2, 4]) else "" for z in z_ticks] # [2, 4, 3.5, 4.5]

        # --- Plot TDP Curve ---
        fig, ax = plt.subplots()
        ax.plot(np.arange(1, len(TDP_pARI1)+1), TDP_ARI, label='ARI', color='red', alpha=0.5)
        ax.plot(np.arange(1, len(TDP_pARI1)+1), TDP_Notip, label='Notip', color='green', alpha=0.5)
        ax.plot(np.arange(1, len(TDP_pARI1)+1), TDP_pARI, label=r'pARI ($\delta=27$)', color='blue', alpha=0.5)
        ax.plot(np.arange(1, len(TDP_pARI1)+1), TDP_pARI1, label=r'pARI ($\delta=1$)', color='pink')
        ax.plot(np.arange(1, len(TDP_pARI1)+1), TDP_calibrated_simes, label=r'pARI ($\delta=0$)', color='orange')

        for (z, count), thresh in zip(sorted(voxel_counts.items()), np.linspace(0.3, 0.9, len(voxel_counts))):
            ax.axvline(x=count, color='purple', linestyle='--', alpha=thresh)
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
        plt.savefig(f'results/contrast{i}/confidence_curve_TDP_{alpha}_full.pdf')
        print(f"Plot completed for task {i}")
