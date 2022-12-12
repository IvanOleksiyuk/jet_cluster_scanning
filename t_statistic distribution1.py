import matplotlib.pyplot as plt
import pickle
from cs_performance_evaluation import sliding_cluster_performance_evaluation
import numpy as np
import random
from matplotlib.ticker import MaxNLocator

random.seed(a=2, version=2)
plt.close("all")
save_path = (
    "char/0kmeans_scan/BS26w60wk50MBret0con0.0W100ste200rewsqrtsme1ID1/"
)
# /home/ivan/mnt/cluster/k_means_anomaly_jet/char/0kmeans_scan/BS26w60wk50MBret0con0.0W100ste200rewsqrtsme1ID1

labelings = [">5sigma", "kmeans_der"]
methods = [1, 2]
fig, axs = plt.subplots(2, 2, figsize=(8, 6.2))
# print(ax[0][1])
for labeling, method, iii in zip(labelings, methods, [0, 1]):
    counts_windows_boot = []
    res = pickle.load(open(save_path + "res.pickle", "rb"))
    counts_windows_boot += res["counts_windows_boot"]
    print("found boostraps: ", len(counts_windows_boot))
    for i in range(46):
        res = pickle.load(
            open(
                "char/0kmeans_scan/BS26w60wk50MBret0con0.0W100ste200rewsqrtsme1ID0/"
                + "res{0:04d}.pickle".format(i),
                "rb",
            )
        )
        counts_windows_boot += res["counts_windows_boot"]
    for i in range(40):
        res = pickle.load(
            open(
                "char/0kmeans_scan/BS26w60wk50MBret0con0.0W100ste200rewsqrtsme1ID1/"
                + "res{0:04d}.pickle".format(i),
                "rb",
            )
        )
        counts_windows_boot += res["counts_windows_boot"]
    for i in range(37):
        res = pickle.load(
            open(
                "char/0kmeans_scan/BS26w60wk50MBret0con0.0W100ste200rewsqrtsme1ID2/"
                + "res{0:04d}.pickle".format(i),
                "rb",
            )
        )
        counts_windows_boot += res["counts_windows_boot"]
    for i in range(44):
        res = pickle.load(
            open(
                "char/0kmeans_scan/BS26w60wk50MBret0con0.0W100ste200rewsqrtsme1ID3/"
                + "res{0:04d}.pickle".format(i),
                "rb",
            )
        )
        counts_windows_boot += res["counts_windows_boot"]
    for i in range(18):
        res = pickle.load(
            open(
                "char/0kmeans_scan/BS26w60wk50MBret0con0.0W100ste200rewsqrtsme1ID4/"
                + "res{0:04d}.pickle".format(i),
                "rb",
            )
        )
        counts_windows_boot += res["counts_windows_boot"]
    for i in range(4):
        res = pickle.load(
            open(
                "char/0kmeans_scan/BS26w60wk50MBret0con0.0W100ste200rewsqrtsme1ID5/"
                + "res{0:04d}.pickle".format(i),
                "rb",
            )
        )
        counts_windows_boot += res["counts_windows_boot"]
    # sliding_cluster_performance_evaluation(np.array(counts_windows_boot[36]), save_path=save_path, plotting=True, labeling=">5sigma")

    bootstraps = len(counts_windows_boot)
    print("found boostraps: ", bootstraps)

    res_list = []
    for i, counts_windows in enumerate(counts_windows_boot):
        counts_windows = np.array(counts_windows)
        res_list.append(
            sliding_cluster_performance_evaluation(
                counts_windows,
                save_path=save_path,
                filterr="med",
                plotting=False,
                labeling=labeling,
                verbous=False,
            )
        )
        if i % 100 == 0:
            print(i)

    #%%
    import scipy

    plt.sca(axs[1][iii])
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
    # cont_paths=["char/0kmeans_scan/26w60wk50ret0con0.1W100ste200rewnonesme0ID2/",
    #            "char/0kmeans_scan/26w60wk50ret0con0.05W100ste200rewsqrtsme1ID0/",
    #            "char/0kmeans_scan/26w60wk50MBret0con0.025W100ste201rewsqrtsme1ID0/"]
    cont_paths = [
        "char/0kmeans_scan/BS26w60wk50ret0con0.1W100ste200rewsqrtsme1ID10/",
        "char/0kmeans_scan/BS26w60wk50ret0con0.05W100ste200rewsqrtsme1ID10/",
        "char/0kmeans_scan/BS26w60wk50ret0con0.025W100ste200rewsqrtsme1ID10/",
    ]
    plt.plot(
        [1], [1], alpha=0, label=r"$n=0.5,\sigma=1$, method {:}".format(method)
    )
    colors = ["red", "orange", "gold"]
    for c, path, col in zip(contamiantions, cont_paths, colors):
        # res=sliding_cluster_performance_evaluation(save_path=path, filterr="med", plotting=False, labeling=labeling, verbous=False)
        # print(res["chisq_ndof"])
        # plt.axvline(res["chisq_ndof"], color=col, label="$\epsilon$={:.4f}, p={:.3f}".format(c, np.sum(chisq_list>res["chisq_ndof"])/len(chisq_list)))
        arr = []
        ps = []
        for jj in range(10):
            res = pickle.load(
                open(path + "res{0:04d}.pickle".format(jj), "rb")
            )
            counts_windows = np.array(res["counts_windows"][0])
            res = sliding_cluster_performance_evaluation(
                counts_windows=counts_windows,
                save_path=path,
                filterr="med",
                plotting=False,
                labeling=labeling,
                verbous=False,
            )
            print(res["chisq_ndof"])
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
            label = "$\epsilon$={:.4f}, $<p><${:.4f}".format(
                c, 1 / len(chisq_list)
            )
        else:
            label = "$\epsilon$={:.4f}, $<p>=${:.4f}".format(c, np.mean(ps))
        plt.axvline(np.mean(arr), color=col, label=label)
        plt.axvspan(
            np.mean(arr) - np.std(arr),
            np.mean(arr) + np.std(arr),
            color=col,
            alpha=0.15,
        )
    plt.legend(loc=1)
    plt.yscale("log")
    # plt.xlim((0, 30))

    #%%

    #%%
    # the same but now for no preprocessing

    # /home/ivan/mnt/cluster/k_means_anomaly_jet/char/0kmeans_scan/BS26w60wk50MBret0con0.0W100ste200rewnonesme0ID1

    counts_windows_boot = []
    for i in range(47):  # 26
        res = pickle.load(
            open(
                "char/0kmeans_scan/BS26w60wk50MBret0con0.0W100ste200rewnonesme0ID0/"
                + "res{0:04d}.pickle".format(i),
                "rb",
            )
        )
        counts_windows_boot += res["counts_windows_boot"]
    for i in range(49):  # 26
        res = pickle.load(
            open(
                "char/0kmeans_scan/BS26w60wk50MBret0con0.0W100ste200rewnonesme0ID1/"
                + "res{0:04d}.pickle".format(i),
                "rb",
            )
        )
        counts_windows_boot += res["counts_windows_boot"]
    for i in range(45):  # 26
        res = pickle.load(
            open(
                "char/0kmeans_scan/BS26w60wk50MBret0con0.0W100ste200rewnonesme0ID1/"
                + "res{0:04d}.pickle".format(i),
                "rb",
            )
        )
        counts_windows_boot += res["counts_windows_boot"]
    for i in range(44):  # 26
        res = pickle.load(
            open(
                "char/0kmeans_scan/BS26w60wk50MBret0con0.0W100ste200rewnonesme0ID1/"
                + "res{0:04d}.pickle".format(i),
                "rb",
            )
        )
        counts_windows_boot += res["counts_windows_boot"]
    for i in range(15):
        res = pickle.load(
            open(
                "char/0kmeans_scan/BS26w60wk50MBret0con0.0W100ste200rewsqrtsme1ID4/"
                + "res{0:04d}.pickle".format(i),
                "rb",
            )
        )
        counts_windows_boot += res["counts_windows_boot"]

    bootstraps = len(counts_windows_boot)
    print("found boostraps: ", bootstraps)
    # sliding_cluster_performance_evaluation(np.array(counts_windows_boot[36]), save_path=save_path, plotting=True, labeling=">5sigma")

    res_list = []
    for i, counts_windows in enumerate(counts_windows_boot):
        counts_windows = np.array(counts_windows)
        res_list.append(
            sliding_cluster_performance_evaluation(
                counts_windows,
                save_path=save_path,
                filterr="med",
                plotting=False,
                labeling=labeling,
                verbous=False,
            )
        )
        if i % 100 == 0:
            print(i)

    #%%
    import scipy

    plt.sca(axs[0][iii])
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
        "char/0kmeans_scan/BS26w60wk50ret0con0.1W100ste200rewnonesme0ID10/",
        "char/0kmeans_scan/BS26w60wk50ret0con0.05W100ste200rewnonesme0ID10/",
        "char/0kmeans_scan/BS26w60wk50ret0con0.025W100ste200rewnonesme0ID10/",
    ]
    plt.plot(
        [1], [1], alpha=0, label=r"$n=0,\sigma=0$, method {:}".format(method)
    )
    colors = ["red", "orange", "gold"]
    for c, path, col in zip(contamiantions, cont_paths, colors):
        arr = []
        ps = []
        for jj in range(10):
            res = pickle.load(
                open(path + "res{0:04d}.pickle".format(jj), "rb")
            )
            counts_windows = np.array(res["counts_windows"][0])
            res = sliding_cluster_performance_evaluation(
                counts_windows=counts_windows,
                save_path=path,
                filterr="med",
                plotting=False,
                labeling=labeling,
                verbous=False,
            )
            print(res["chisq_ndof"])
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
            label = "$\epsilon$={:.4f}, $<p><${:.4f}".format(
                c, 1 / len(chisq_list)
            )
        else:
            label = "$\epsilon$={:.4f}, $<p>=${:.4f}".format(c, np.mean(ps))
        plt.axvline(np.mean(arr), color=col, label=label)
        plt.axvspan(
            np.mean(arr) - np.std(arr),
            np.mean(arr) + np.std(arr),
            color=col,
            alpha=0.15,
        )
    plt.legend(loc=1)
    plt.yscale("log")
    # plt.xlim((0, 30))

    #%%
    # ax = plt.gca()
    # ax.set_xticks([0, 1, 2])

plt.savefig("plots2/chi_all.png", bbox_inches="tight")
