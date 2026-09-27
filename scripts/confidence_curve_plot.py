import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sanssouci as sa
from matplotlib.ticker import FormatStrFormatter

# Set up paths and ensure figure directory exists
script_path = os.path.dirname(__file__)
repo_path = os.path.abspath(os.path.join(script_path, '..'))
sys.path.append(repo_path)

results_path_ = os.path.abspath(os.path.join(repo_path, 'results'))
fig_path_ = os.path.abspath(os.path.join(repo_path, 'figures'))

# Parameters
ALPHAS = [0.05, 0.1]
B_calib = 100
B_train = 100
PLOT_ALL_PARI = False  # if False, only plot ARI, Notip and pARI (delta=27)

# Load dataset task list
df_tasks = pd.read_csv(os.path.join(script_path, 'contrast_list2.csv'))
test_task1s, test_task2s = df_tasks['task1'], df_tasks['task2']

for alpha in ALPHAS:
    for i in range(len(test_task1s)):
        task1 = test_task1s[i]
        task2 = test_task2s[i]

        # Load z_vals / p_values (instead of redoing the fMRI preprocessing)
        stats_path = os.path.join(results_path_, f"stats_contrast{i}.npz")
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"[ERROR] z-values file not found:\n{stats_path}")
        zvals_data = np.load(stats_path)
        z_vals = zvals_data["z_vals"]
        p_values = zvals_data["p_values"]

        # Count voxels above thresholds. This is equivalent to counting on
        # the reconstructed 3D map since all thresholds here are positive
        # and background (out-of-mask) voxels are 0.
        z_thresholds = [3, 3.5, 4, 4.5]
        voxel_counts = {z: np.sum(z_vals > z) for z in z_thresholds}

        threshold_path = os.path.join(
            results_path_,
            f"thresholds_contrast{i}_alpha{alpha}_Bcalib{B_calib}_Btrain{B_train}.npz"
        )
        if not os.path.exists(threshold_path):
            raise FileNotFoundError(f"[ERROR] Threshold file not found:\n{threshold_path}")
        
        fig_path = os.path.join(
            fig_path_,
            f'contrast{i}'
        )
        os.makedirs(fig_path, exist_ok=True)

        # Load thresholds
        thr = np.load(threshold_path)

        TDP_ARI = sa.curve_min_tdp(p_values, thr["ari_thr"])
        TDP_pARI = sa.curve_min_tdp(p_values, thr["pari_thr"])
        TDP_Notip = sa.curve_min_tdp(p_values, thr["notip_thr"])

        if PLOT_ALL_PARI:
            TDP_pARI1 = sa.curve_min_tdp(p_values, thr["pari1_thr"])
            TDP_calibrated_simes = sa.curve_min_tdp(p_values, thr["pari0_thr"])

        # Set up ticks for secondary axis
        z_max = int(np.floor(np.max(z_vals)))
        z_ticks = list(np.arange(1, z_max + 1))  # + [3.5, 4.5]
        z_ticks = sorted(set(z_ticks))  # avoid duplicates
        k_ticks = [np.sum(z_vals > z) for z in z_ticks]
        z_labels = [str(z) if (z % 2 == 1 or z in [2, 4]) else "" for z in z_ticks] # [2, 4, 3.5, 4.5]

        # --- Plot TDP Curve ---
        fig, ax = plt.subplots()
        ax.plot(np.arange(1, len(TDP_ARI)+1), TDP_ARI, label='ARI', color='red', alpha=0.5)
        ax.plot(np.arange(1, len(TDP_Notip)+1), TDP_Notip, label='Notip', color='green', alpha=0.5)
        ax.plot(np.arange(1, len(TDP_pARI)+1), TDP_pARI, label=r'pARI ($\delta=27$)', color='blue', alpha=0.5)
        if PLOT_ALL_PARI:
            ax.plot(np.arange(1, len(TDP_pARI1)+1), TDP_pARI1, label=r'pARI ($\delta=1$)', color='pink')
            ax.plot(np.arange(1, len(TDP_calibrated_simes)+1), TDP_calibrated_simes, label=r'pARI ($\delta=0$)', color='orange')

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
        # add contrast names? too crowded
        # plt.suptitle(f"{task1} vs {task2}", y=1, fontsize=12)
        
        plt.tight_layout()
        suffix = "_all-curves" if PLOT_ALL_PARI else ""
        fig_pathname = os.path.join(
            fig_path,
            f'confidence_curve_TDP_{alpha}{suffix}.pdf'
        )
        plt.savefig(fig_pathname, bbox_inches='tight')
        plt.close()
        print(f"Plot completed for task {i}")
