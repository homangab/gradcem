import matplotlib.pyplot as plt
import numpy as np

results = np.load('results_new.npy')
x = np.arange(2,4)
g_ret = results[0]
g_std = results[1]
c_ret = results[2]
c_std = results[3]
cg_ret = results[3]
cg_std = results[4]

plt.errorbar(x, g_ret, yerr=g_std, label="Grad", fmt='o', capsize=2)
plt.errorbar(x, c_ret, yerr=c_std, label="CEM", fmt='o', capsize=2)
plt.errorbar(x, cg_ret, yerr=cg_std, label="Grad+CEM", fmt='o', capsize=2)
plt.legend()
plt.xticks(x)
plt.ylabel("Total Reward")
plt.xlabel("Action Dimensionality")
plt.savefig("./comp_plot.png")
