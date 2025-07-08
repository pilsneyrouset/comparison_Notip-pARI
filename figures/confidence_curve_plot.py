import numpy as np
import matplotlib.pyplot as plt

TDP_ARI = np.load('TDP/TDP_ARI_out.npy')
TDP_Notip = np.load('TDP/TDP_Notip_out.npy')
TDP_pARI = np.load('TDP/TDP_pARI_out.npy')


def FDP(TDP):
    return 1 - TDP


plt.figure()
plt.title(r'$\alpha = 0.5$')
plt.plot(TDP_ARI[:10000], label='ARI')
plt.plot(TDP_Notip[:10000], label='Notip')
plt.plot(TDP_pARI[:10000], label='pARI')
plt.xscale("log")
plt.ylabel("Borne sur le TDP")
plt.xlabel("k")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend()
plt.savefig('confidence_curve_TDP.pdf')


plt.figure()
plt.title(r'$\alpha = 0.5$')
plt.plot(FDP(TDP_ARI)[:10000], label='ARI')
plt.plot(FDP(TDP_Notip)[:10000], label='Notip')
plt.plot(FDP(TDP_pARI)[:10000], label='pARI')
plt.xscale("log")
plt.ylabel("Borne sur le FDP")
plt.xlabel("k")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend()
plt.savefig('confidence_curve_FDP.pdf')
