import os

import nibabel as nib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from utils import get_clusters_table_with_TDP_task

# Paths setup
script_path = os.path.dirname(__file__)
repo_path = os.path.abspath(os.path.join(script_path, '..'))

results_path_ = os.path.join(repo_path, 'results')
tables_path_ = os.path.join(repo_path, 'supplementary')

# Parameters
z_thresholds = [3, 3.5, 4, 4.5, 5, 5.5]
alpha = 0.05     # must match one of the ALPHAS used in compute_thresholds.py
B_calib = 1000  # must match B_calib used in compute_thresholds.py
B_train = 1000  # must match B_train used in compute_thresholds.py
n_jobs = 5

# Load dataset contrasts
df_tasks = pd.read_csv(os.path.join(script_path, 'contrast_list2.csv'))
test_task1s, test_task2s = df_tasks['task1'], df_tasks['task2']

def process_task(i, task1, task2):
    out_dir = os.path.join(tables_path_, f'contrast{i}')
    os.makedirs(out_dir, exist_ok=True)

    # Load the z-map, computed once during the fit step (compute_thresholds.py)
    # instead of redoing the fMRI preprocessing
    zmap_path = os.path.join(results_path_, f"zmap_contrast{i}.nii.gz")
    if not os.path.exists(zmap_path):
        raise FileNotFoundError(f"[ERROR] z-map file not found:\n{zmap_path}")
    z_map = nib.load(zmap_path)

    # Load the thresholds once per task (reused for every z below, instead
    # of being reloaded from disk at each iteration)
    threshold_path = os.path.join(
        results_path_,
        f"thresholds_contrast{i}_alpha{alpha}_Bcalib{B_calib}_Btrain{B_train}.npz"
    )
    if not os.path.exists(threshold_path):
        raise FileNotFoundError(f"[ERROR] Threshold file not found:\n{threshold_path}")
    thr = np.load(threshold_path)

    for z in z_thresholds:
        df = get_clusters_table_with_TDP_task(
            z_map,
            thr=thr,
            stat_threshold=z,
            methods=['ARI', 'Notip', 'pARI', 'pARI1']
        )
        output_file = os.path.join(out_dir, f'z_threshold_{z}.csv')
        df.to_csv(output_file, index=False)


Parallel(n_jobs=n_jobs, verbose=10)(
    delayed(process_task)(i, t1, t2)
    for i, (t1, t2) in enumerate(zip(test_task1s, test_task2s))
)

print("Finished.")
