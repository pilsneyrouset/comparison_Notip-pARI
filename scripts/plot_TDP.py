import matplotlib.pyplot as plt
import numpy as np

TDP_notip = np.load("TDP_notip.npy")
TDP_calibrated_simes = np.load("TDP_calibrated_simes.npy")
TDP_pARI = np.load("TDP_pARI.npy")

Sk = [100, 200, 500, 1000, 2000, 5000, 10000]

for k in range(len(Sk)):
    notip = TDP_notip[:, k]
    pari = TDP_pARI[:, k]
    calibrated_simes = TDP_calibrated_simes[:, k]

    plt.title(f'TDP for the subsets of the {Sk[k]}-th first p-values for 36 datasets')
    plt.boxplot([calibrated_simes, notip, pari], labels=["Calibrated Simes", "Notip", "pARI"])
    plt.savefig(f'comparison_TDP_{k}.png')
