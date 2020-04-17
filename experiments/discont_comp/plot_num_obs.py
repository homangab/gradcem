import matplotlib.pyplot as plt
import numpy as np

results = np.load('results_new_num_obs1.npy')
x = np.arange(5,8)
g_ret = results[0][0:3]
g_std = results[1][0:3]
c_ret = results[2][0:3]
c_std = results[3][0:3]
cg_ret = results[3][0:3]
cg_std = results[4][0:3]
print(g_ret)

plt.errorbar(x, g_ret, yerr=g_std, label="Grad", fmt='-o', capsize=2)
plt.errorbar(x, c_ret, yerr=c_std, label="CEM", fmt='-o', capsize=2)
plt.errorbar(x, cg_ret, yerr=cg_std, label="Grad+CEM", fmt='-o', capsize=2)
plt.legend()
plt.xticks(x, x**2)
plt.ylabel("Total Reward")
plt.xlabel("Number of Obstacles")
plt.savefig("./comp_plot_num_obs.png")
