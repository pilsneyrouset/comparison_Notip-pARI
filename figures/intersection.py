import numpy as np
import pandas as pd
import os
import sys
script_path = os.path.dirname(__file__)
fig_path_ = os.path.abspath(os.path.join(script_path, os.pardir))
fig_path = os.path.join(fig_path_, 'figures')
os.makedirs(fig_path, exist_ok=True)
sys.path.append(os.path.abspath(os.path.join(script_path, '..')))
script_path = os.path.dirname(__file__)
fig_path_ = os.path.abspath(os.path.join(script_path, os.pardir))
fig_path = os.path.join(fig_path_, 'figures')

alpha = 0.1

def first_crossing_idx(tdp_a, tdp_b):
    """
    Renvoie l’indice i tel que tdp_a[i-1] > tdp_b[i-1] et tdp_a[i] <= tdp_b[i],
    c’est-à-dire le premier croisement de haut en bas de A vers B.
    """
    tdp_a = np.asarray(tdp_a)
    tdp_b = np.asarray(tdp_b)
    
    delta = tdp_a - tdp_b
    
    for i in range(1, len(delta)):
        if delta[i-1] > 0 and delta[i] <= 0:
            return i  # indice du croisement
        
    return None

# Données du dataset
df_tasks = pd.read_csv(os.path.join(script_path, 'contrast_list2.csv'))
test_task1s, test_task2s = df_tasks['task1'], df_tasks['task2']

data = []
for i in range(len(test_task1s)):
    TDP_ARI = np.load(f"task{i}/TDP_ARI_{alpha}.npy")
    TDP_Notip = np.load(f"task{i}/TDP_Notip_{alpha}.npy")
    TDP_pARI = np.load(f"task{i}/TDP_pARI_{alpha}.npy")
    
    intersection_Notip_pARI = first_crossing_idx(TDP_Notip, TDP_pARI)
    intersection_ARI_pARI = first_crossing_idx(TDP_ARI, TDP_pARI)
    
    data.append({
        'contraste' : i,
        'intersection Notip/pARI': intersection_Notip_pARI,
        'intersection ARI/pARI': intersection_ARI_pARI
    })

df = pd.DataFrame(data)
df.to_csv(f"intersections/intersection_table_{alpha}.csv", index=False)