import matplotlib.pyplot as plt
import numpy as np

results = np.load('results_new.npy')
x = np.arange(2, 21)
g_ret = results[0]
g_std = results[1]
c_ret = results[2]
c_std = results[3]
gc_ret = results[4]
gc_std = results[5]

plt.errorbar(x, g_ret, yerr=g_std, label="Grad")
plt.errorbar(x, c_ret, yerr=c_std, label="CEM")
plt.errorbar(x, gc_ret, yerr=gc_std, label="Grad+CEM")
plt.legend()
plt.xticks(x)
plt.ylabel("Total Reward")
plt.xlabel("Action Dimensionality")
plt.savefig("./test20D_new.png")
