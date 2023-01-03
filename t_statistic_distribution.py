import matplotlib.pyplot as plt
import pickle
from cs_performance_evaluation import cs_performance_evaluation
import numpy as np
import random
import scipy
from matplotlib.ticker import MaxNLocator
import cluster_scanning

import set_matplotlib_default as smd
from load_old_bootstrap_experiments import (
    load_old_bootstrap_experiments05_1,
    load_old_bootstrap_experiments00,
)


random.seed(a=2, version=2)
plt.close("all")
output_path = "plots2/"
labeling = ">5sigma"  # "kmeans_der" #"kmeans_der"#">5sigma"#
filterr = "med"
mehtod = 1

counts_windows_boot = load_old_bootstrap_experiments00()

res_list = []
for i, counts_windows in enumerate(counts_windows_boot):
    counts_windows = np.array(counts_windows)
    res_list.append(
        cs_performance_evaluation(
            counts_windows,
            save=False,
            filterr=filterr,
            plotting=False,
            labeling=labeling,
            verbous=False,
        )
    )
    if i % 100 == 0:
        print(i)


plt.figure(figsize=(4, 3))
chisq_list = [el["chisq_ndof"] for el in res_list]
chisq_list = np.array(chisq_list)
ndof = res_list[0]["ndof"]
plt.hist(chisq_list, bins=40)
x = np.linspace(0.01, 5)
plt.xlabel(r"$\tilde{\chi}^2/n_{dof}$")
plt.ylabel("Tries")
# plt.axvline(np.mean(chisq_list), color="blue", label=r"Average for $H_0$")
print(np.mean(chisq_list))
print(np.std(chisq_list))
ndof = 7  # 2/(np.std(chisq_list))**2
# plt.plot(x, scipy.stats.chi2.pdf(x*ndof, df=ndof)*ndof)

#%%
# Contaminations
contamiantions = [0.01, 0.005, 0.0025]
cont_paths = [
    "char/old_char/BS26w60wk50ret0con0.1W100ste200rewsqrtsme1ID10/",
    "char/old_char/BS26w60wk50ret0con0.05W100ste200rewsqrtsme1ID10/",
    "char/old_char/BS26w60wk50ret0con0.025W100ste200rewsqrtsme1ID10/",
]
plt.plot(
    [1], [1], alpha=0, label=r"$n=0.5,\sigma=1$, method {:}".format(mehtod)
)
colors = ["red", "orange", "gold"]


def draw_contamination(c, path, col, old=False, postfix=""):
    arr = []
    ps = []
    for jj in range(10):
        if old:
            res = pickle.load(
                open(path + "res{0:04d}.pickle".format(jj), "rb")
            )
            counts_windows = np.array(res["counts_windows"][0])
        else:
            cs = cluster_scanning.ClusterScanning(path)
            cs.load_mjj()
            cs.ID = jj
            cs.load_results(jj)
            cs.sample_signal_events()
            counts_windows = cs.perform_binning()

        res = cs_performance_evaluation(
            counts_windows=counts_windows,
            save=False,
            filterr=filterr,
            plotting=False,
            labeling=labeling,
            verbous=False,
        )
        # print(res["chisq_ndof"])
        arr.append(res["chisq_ndof"])
        ps.append(
            (
                np.sum(chisq_list >= res["chisq_ndof"])
                + np.sum(chisq_list > res["chisq_ndof"])
            )
            / 2
            / len(chisq_list)
        )

    if np.mean(ps) == 0:
        label = (
            "$\epsilon$={:.4f}, $<p><${:.4f}".format(c, 1 / len(chisq_list))
            + postfix
        )
    else:
        label = (
            "$\epsilon$={:.4f}, $<p>=${:.4f}".format(c, np.mean(ps)) + postfix
        )
    plt.axvline(np.mean(arr), color=col, label=label)
    plt.axvspan(
        np.mean(arr) - np.std(arr),
        np.mean(arr) + np.std(arr),
        color=col,
        alpha=0.15,
    )


for c, path, col in zip(contamiantions, cont_paths, colors):
    draw_contamination(c, path, col, old=True)

cont_paths_MB = [
    "config/s0.1_0.5_1_MB.yml",
    "config/s0.05_0.5_1_MB.yml",
    "config/s0.025_0.5_1_MB.yml",
]
colors = ["darkgreen", "green", "lime"]
for c, path, col in zip(contamiantions, cont_paths_MB, colors):
    draw_contamination(c, path, col, old=False, postfix=" MB")

plt.legend(loc=1)
plt.yscale("log")
# plt.xlim((0, 30))

plt.savefig(
    output_path + "chi_s1sqrt_" + labeling[1:] + ".png",
    dpi=300,
    bbox_inches="tight",
)

#%%
# the same but now for no preprocessing
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("First part finished")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


counts_windows_boot = load_old_bootstrap_experiments05_1()

res_list = []
for i, counts_windows in enumerate(counts_windows_boot):
    counts_windows = np.array(counts_windows)
    res_list.append(
        cs_performance_evaluation(
            counts_windows,
            save=False,
            filterr=filterr,
            plotting=False,
            labeling=labeling,
            verbous=False,
        )
    )
    if i % 100 == 0:
        print(i)

plt.figure(figsize=(4, 3))
chisq_list = [el["chisq_ndof"] for el in res_list]
chisq_list = np.array(chisq_list)
ndof = res_list[0]["ndof"]
plt.hist(chisq_list, bins=40)
x = np.linspace(0.01, 5)
plt.xlabel(r"$\tilde{\chi}^2/n_{dof}$")
plt.ylabel("Tries")
# plt.axvline(np.mean(chisq_list), color="blue", label=r"Average for $H_0$")
print(np.mean(chisq_list))
print(np.std(chisq_list))
# plt.plot(x, scipy.stats.chi2.pdf(x*ndof, df=ndof)*ndof)

#%%
# Contaminations

contamiantions = [0.01, 0.005, 0.0025]
cont_paths = [
    "char/old_char/BS26w60wk50ret0con0.1W100ste200rewnonesme0ID10/",
    "char/old_char/BS26w60wk50ret0con0.05W100ste200rewnonesme0ID10/",
    "char/old_char/BS26w60wk50ret0con0.025W100ste200rewnonesme0ID10/",
]

plt.plot([1], [1], alpha=0, label=r"$n=0,\sigma=0$, method {:}".format(mehtod))
colors = ["red", "orange", "gold"]
for c, path, col in zip(contamiantions, cont_paths, colors):
    draw_contamination(c, path, col, old=True)

cont_paths_MB = [
    "config/s0.1_0.5_1_MB.yml",
    "config/s0.05_0.5_1_MB.yml",
    "config/s0.025_0.5_1_MB.yml",
]
colors = ["darkgreen", "green", "lime"]
for c, path, col in zip(contamiantions, cont_paths_MB, colors):
    draw_contamination(c, path, col, old=False, postfix=" MB")

plt.legend(loc=1)
plt.yscale("log")
# plt.xlim((0, 30))

ax = plt.gca()
ax.set_xticks([0, 1, 2])
plt.savefig(
    output_path + "chi_s0none_" + labeling[1:] + ".png",
    dpi=300,
    bbox_inches="tight",
)
