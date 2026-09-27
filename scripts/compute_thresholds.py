import os
import sys

import numpy as np
import pandas as pd
import sanssouci as sa
from joblib import Parallel, delayed
from scipy import stats
from scipy.stats import norm
from utils import (
    Timer,
    _compute_hommel_value,
    get_data_driven_template_two_tasks,
    get_processed_input,
)

# Paths setup
script_path = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(script_path, '..')))


# -------------------- PARAMETERS --------------------
OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

ALPHAS = [0.1, 0.05]
B_train = 100        
B_calib = 100        
n_jobs = 5
seed = 42
training_seed = 23
delta = 27
k_max = 1000

# Load dataset contrasts
df_tasks = pd.read_csv(os.path.join(script_path, 'contrast_list2.csv'))
test_task1s, test_task2s = df_tasks['task1'], df_tasks['task2']

tasks = list(zip(test_task1s, test_task2s))


# ------------------- CORE FUNCTION -------------------
def compute_for_task(i, task1, task2):
    t = Timer()

    with t.step(f"[{i}] Loading fMRI input for {task1} vs {task2}"):
        fmri_input, nifti_masker = get_processed_input(task1, task2,
                                            smoothing_fwhm=4,
                                            collection=1952)
    p = fmri_input.shape[1]

    # ----- Compute Z-values (common to all alphas) -----
    with t.step(f"[{i}] Computing test statistics"):
        _, p_values = stats.ttest_1samp(fmri_input, 0)
        z_vals = norm.isf(p_values)
        z_nonzero = z_vals[z_vals != 0]

    # save z_vals/p_values and reconstructed 3D z-map, so that plotting 
    # scripts that require this info can use it instead of recalculating
    zvals_fname = os.path.join(OUT_DIR, f"stats_contrast{i}.npz")
    np.savez_compressed(zvals_fname, z_vals=z_vals, p_values=p_values)

    zmap_fname = os.path.join(OUT_DIR, f"zmap_contrast{i}.nii.gz")
    nifti_masker.inverse_transform(z_vals).to_filename(zmap_fname)

    # ----- Permutations for pARI / Notip  -----
    with t.step(f"[{i}] Permutations (B={B_calib})"):
        pval0 = sa.get_permuted_p_values_one_sample(
            fmri_input, B=B_calib, n_jobs=n_jobs, seed=seed
        )

    # ----- Data-driven template  -----
    with t.step(f"[{i}] Training data-driven templates (Notip, B={B_train})"):
        learned_templates = get_data_driven_template_two_tasks(
            task1, task2, B=B_train, seed=training_seed
        )

    # ----- Pivotal stats: independent of alpha -----
    with t.step(f"[{i}] Computing pivotal stats (Simes + shifted Simes x3)"):
        piv_stat_simes = sa.get_pivotal_stats(pval0, K=p)
        piv_stat_pari = sa.get_pivotal_stats_shifted(pval0, k_min=delta)
        piv_stat_pari0 = sa.get_pivotal_stats_shifted(pval0, k_min=0)
        piv_stat_pari1 = sa.get_pivotal_stats_shifted(pval0, k_min=1)

    # ----- Compute thresholds for each alpha in ALPHAS -----
    #       (only the quantile + template step left)
    # ------------------------------------------------------
    outputs = {}

    for alpha in ALPHAS:
        with t.step(f"[{i}] Computing thresholds for alpha={alpha}"):

            # --- Hommel & ARI ---
            hommel = _compute_hommel_value(z_nonzero, alpha)
            ari_thr = sa.linear_template(alpha, hommel, hommel)

            # --- Shifted Simes (pARI) ---
            lambda_quant = np.quantile(piv_stat_pari, alpha)
            pari_thr = sa.shifted_linear_template(alpha=lambda_quant, k=p, m=p, k_min=delta)

            # --- Calibrated Simes ---
            lambda_quant = np.quantile(piv_stat_simes, alpha)
            calibrated_simes_thr = sa.linear_template(lambda_quant, p, p)

            # --- pARI with delta=0
            lambda_quant = np.quantile(piv_stat_pari0, alpha)
            pari0_thr = sa.shifted_linear_template(alpha=lambda_quant, k=p, m=p, k_min=0)

            # --- pARI with delta=1
            lambda_quant = np.quantile(piv_stat_pari1, alpha)
            pari1_thr = sa.shifted_linear_template(alpha=lambda_quant, k=p, m=p, k_min=1)

            # --- Notip ---
            notip_thr = sa.calibrate_jer(
                alpha,
                learned_templates,
                pval0,
                k_max=k_max
            )

            # sanity check: pARI with k_min=0 and calibrated Simes should coincide
            if not np.allclose(pari0_thr, calibrated_simes_thr):
                print(f"[{i}] WARNING: pari0_thr != calibrated_simes_thr for alpha={alpha}")

            # store results
            outputs[alpha] = {
                'ari_thr': ari_thr,
                'pari_thr': pari_thr,
                'notip_thr': notip_thr,
                'pari1_thr': pari1_thr,
                'pari0_thr': pari0_thr
            }

            # save results
            fname = os.path.join(
                OUT_DIR,
                f"thresholds_contrast{i}_alpha{alpha}_Bcalib{B_calib}_Btrain{B_train}.npz"
            )
            np.savez_compressed(fname, **outputs[alpha])
    return outputs


# -------------------- PARALLEL EXECUTION --------------------
global_timer = Timer()
with global_timer.step(f"Running all {len(tasks)} tasks (n_jobs={n_jobs})"):
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_for_task)(i, t1, t2)
        for i, (t1, t2) in enumerate(tasks)
    )

print("Finished.")
