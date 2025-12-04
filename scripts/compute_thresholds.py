import os
import sys
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import sanssouci as sa
from utils import (
    get_processed_input,
    calibrate_simes,
    calibrate_shifted_simes,
    get_data_driven_template_two_tasks,
    _compute_hommel_value
)
from sanssouci.lambda_calibration import calibrate_jer

from scipy import stats
from scipy.stats import norm

# Paths setup
script_path = os.path.dirname(__file__)
fig_path_ = os.path.abspath(os.path.join(script_path, os.pardir))
sys.path.append(os.path.abspath(os.path.join(script_path, '..')))


# -------------------- PARAMÈTRES --------------------
OUT_DIR = "results/thresholds"
os.makedirs(OUT_DIR, exist_ok=True)

ALPHAS = [0.1, 0.05]   # <--- les deux valeurs
B_train = 10000        
B_calib = 10000        
n_jobs = 10
seed = 42
delta = 27

# Load dataset contrasts
df_tasks = pd.read_csv(os.path.join(script_path, 'contrast_list2.csv'))
test_task1s, test_task2s = df_tasks['task1'], df_tasks['task2']

tasks = list(zip(test_task1s, test_task2s))


# ------------------- CORE FUNCTION -------------------
def compute_for_task(i, task1, task2):
    print(f"[{i}] Loading fMRI input for {task1} vs {task2}")
    fmri_input, _ = get_processed_input(task1, task2,
                                        smoothing_fwhm=4,
                                        collection=1952)

    # ----- Compute Z-values (common to all alphas) -----
    stats_, p_values = stats.ttest_1samp(fmri_input, 0)
    z_vals = norm.isf(p_values)
    z_nonzero = z_vals[z_vals != 0]

    # ----- Permutations for Simes / Notip (only once!) -----
    print(f"[{i}] Permutations (B={B_calib})...")
    pval0 = sa.get_permuted_p_values_one_sample(
        fmri_input, B=B_calib, n_jobs=n_jobs, seed=seed
    )

    # ----- Data-driven template (only once!) -----
    print(f"[{i}] Training data-driven templates (Notip, B={B_train})...")
    learned_templates = get_data_driven_template_two_tasks(
        task1, task2, B=B_train, seed=seed
    )
    learned_templates_sorted = np.sort(learned_templates, axis=0)

    # ------------------------------------------------------
    #       COMPUTE THRESHOLDS FOR EACH α IN ALPHAS
    # ------------------------------------------------------
    outputs = {}

    for alpha in ALPHAS:
        print(f"[{i}] Computing thresholds for alpha={alpha}")

        # --- Hommel & ARI ---
        hommel = _compute_hommel_value(z_nonzero, alpha)
        ari_thr = sa.linear_template(alpha, hommel, hommel)

        # --- Simes ---
        _, simes_thr = calibrate_simes(
            fmri_input, alpha,
            k_max=1000, B=B_calib, n_jobs=n_jobs, seed=seed
        )

        # --- Shifted Simes (pARI) ---
        _, pari_thr = calibrate_shifted_simes(
            fmri_input, alpha,
            B=B_calib, n_jobs=n_jobs, seed=seed, k_min=delta
        )

        # --- Notip ---
        notip_thr = calibrate_jer(
            alpha,
            learned_templates_sorted,
            pval0,
            k_max=1000
        )

        # store results
        outputs[alpha] = dict(
            ari_thr=ari_thr,
            simes_thr=simes_thr,
            pari_thr=pari_thr,
            notip_thr=notip_thr
        )

        # save immediately
        fname = os.path.join(
            OUT_DIR, f"thresholds_task{i}_alpha{alpha}.npz"
        )
        np.savez_compressed(fname, **outputs[alpha])
        print(f"[{i}] Saved -> {fname}")

    return outputs


# -------------------- PARALLEL EXECUTION --------------------
results = Parallel(n_jobs=4)(
    delayed(compute_for_task)(i, t1, t2)
    for i, (t1, t2) in enumerate(tasks)
)

print("Finished.")
