# imports
import os
import matplotlib.pyplot as plt
import pickle
from cs_performance_evaluation import cs_performance_evaluation, CS_evaluation_process
import numpy as np
import random
import time
from matplotlib.ticker import MaxNLocator
import cluster_scanning
import set_matplotlib_default as smd
from load_old_bootstrap_experiments import (
    load_old_bootstrap_experiments05_1,
    load_old_bootstrap_experiments00,
)
from config_utils import Config


def score_sample(cfg, counts_windows_boot_load):

    if counts_windows_boot_load == "old00":
        counts_windows_boot = load_old_bootstrap_experiments00()
    elif counts_windows_boot_load == "old05_1":
        counts_windows_boot = load_old_bootstrap_experiments05_1()
    else:
        print(len(os.listdir(counts_windows_boot_load)))
        counts_windows_boot = []
        for file in os.listdir(counts_windows_boot_load):
            if file.startswith("bres"):
                counts_windows_boot.append(
                    pickle.load(open(counts_windows_boot_load + file, "rb"))
                )

    tstat_list = []
    for i, counts_windows in enumerate(counts_windows_boot):
        counts_windows = np.array(counts_windows)
        tstat_list.append(
            cs_performance_evaluation(
                counts_windows=counts_windows, config_file_path=cfg.CSEconf
            )
        )
        if i % 100 == 0:
            print(i)

    tstat_list = np.array(tstat_list)
    return tstat_list


def draw_contamination(
    cfg, c, path, col, tstat_list, old=False, postfix="", style="Uall"
):
    arr = []
    ps = []
    for jj in range(10):
        if old == 1:
            res = pickle.load(open(path + "res{0:04d}.pickle".format(jj), "rb"))
            counts_windows = np.array(res["counts_windows"][0])
        elif old == 2:
            cs = cluster_scanning.ClusterScanning(path)
            cs.load_mjj()
            cs.ID = jj
            cs.load_results(jj)
            cs.sample_signal_events()
            cs.bootstrap_resample()
            counts_windows = cs.perform_binning()
        else:
            counts_windows = pickle.load(open(path + f"bres{jj}.pickle", "rb"))

        res = cs_performance_evaluation(
            counts_windows=counts_windows, config_file_path=cfg.CSEconf
        )
        # print(res["chisq_ndof"])
        arr.append(res)
        ps.append(
            (np.sum(tstat_list >= res) + np.sum(tstat_list > res)) / 2 / len(tstat_list)
        )

    if np.mean(ps) == 0:
        label = (
            "$\epsilon$={:.4f}, $<p><${:.4f}".format(c, 1 / len(tstat_list)) + postfix
        )
    else:
        label = "$\epsilon$={:.4f}, $<p>=${:.4f}".format(c, np.mean(ps)) + postfix

    if style[0] == "U":
        fig = plt.gcf()
        fig.add_subplot(2, 1, (1, 1))
        style = style[1:]

    if style == "mean_std":
        plt.axvline(np.mean(arr), color=col, label=label)
        plt.axvspan(
            np.mean(arr) - np.std(arr),
            np.mean(arr) + np.std(arr),
            color=col,
            alpha=0.15,
        )
    elif style == "all":
        for i, a in enumerate(arr):
            if i == 0:
                plt.axvline(a, color=col, label=label, alpha=0.2)
            else:
                plt.axvline(a, color=col, alpha=0.2)
    elif style == "mean":
        plt.axvline(np.mean(arr), color=col, label=label)
    elif style == "median_quartiles":
        plt.axvline(np.median(arr), color=col, label=label)
        plt.axvspan(
            np.quantile(arr, 0.25),
            np.quantile(arr, 0.75),
            color=col,
            alpha=0.15,
        )


def t_statistic_distribution(config_file_path):
    config = Config(config_file_path)
    cfg = config.get_dotmap()

    # set seed
    random.seed(a=cfg.seed, version=2)
    np.random.seed(cfg.seed)

    # Replace all these where needed
    output_path = cfg.output_path
    contamiantions = cfg.contaminations
    cont_paths = cfg.cont_paths
    colors = cfg.colors

    # initialise the main figure:
    plt.close("all")
    plt.figure(figsize=(4, 3))

    if isinstance(cfg.counts_windows_boot_load, str):
        chisq_list = score_sample(cfg, cfg.counts_windows_boot_load)
        if cfg.density:
            plt.hist(chisq_list, bins=40, density=True)
        else:
            plt.hist(chisq_list, bins=40)
    else:
        for counts_windows_boot_load in cfg.counts_windows_boot_load:
            chisq_list = score_sample(cfg, counts_windows_boot_load)
            # chisq_list_lists.append(chisq_list)
            if cfg.density:
                plt.hist(chisq_list, bins=40, density=True, alpha=0.5)
            else:
                plt.hist(chisq_list, bins=40, alpha=0.5)

    plt.xlabel(cfg.xlabel)
    if cfg.density:
        plt.ylabel("Density")
    else:
        plt.ylabel("Tries")
    # plt.axvline(np.mean(chisq_list), color="blue", label=r"Average for $H_0$")
    print(np.mean(chisq_list))
    print(np.std(chisq_list))
    ndof = 7  # 2/(np.std(chisq_list))**2
    # plt.plot(x, scipy.stats.chi2.pdf(x*ndof, df=ndof)*ndof)

    #%%
    # Contaminations
    plt.title(cfg.title)

    for c, path, col, old, postfix in zip(
        contamiantions, cont_paths, colors, cfg.old_CS, cfg.postfix
    ):
        draw_contamination(cfg, c, path, col, chisq_list, old=old, postfix=postfix)

    plt.legend(loc=1)
    plt.yscale("log")
    # plt.xlim((0, 30))
    plt.savefig(
        output_path + cfg.plot_name,
        dpi=300,
        bbox_inches="tight",
    )
    # plt.show()


if __name__ == "__main__":
    # t_statistic_distribution("config/distribution/prep0_0_LABkmeans_der.yaml")
    # t_statistic_distribution("config/distribution/compare/compare_old_to_new00.yaml")
    # t_statistic_distribution("config/distribution/compare/compare_old_to_new0.5_1.yaml")
    # t_statistic_distribution("config/distribution/prep0_0_LABmaxdev5CURTAINS.yaml")
    # t_statistic_distribution("config/distribution/prep05_1_LABmaxdev5CURTAINS.yaml")
    start_time = time.time()
    t_statistic_distribution("config/distribution/prep05_1_LABmaxdev5.yaml")
    print("done in time --- %s seconds ---" % (time.time() - start_time))
    # t_statistic_distribution("config/distribution/prep0_0_LABmaxdev5.yaml")

    # t_statistic_distribution("config/distribution/compare_old_distributions.yaml")
    # t_statistic_distribution("config/distribution/rest_vs_boot0.5_1.yaml")
    # t_statistic_distribution("config/distribution/compare_old_to_new0.5_1.yaml")
    # t_statistic_distribution("config/distribution/compare_old_to_new00_rest.yaml")
    # t_statistic_distribution("config/distribution/prep0_0_LABmaxdev5.yaml")
    # t_statistic_distribution("config/distribution/prep0_0_LABkmeans_der.yaml")
    # t_statistic_distribution("config/distribution/prep05_1_LABkmeans_der.yaml")
    # t_statistic_distribution(
    #     "config/distribution/prep05_1_LABmaxdev5_max-sumnorm-dev.yaml"
    # )
    # t_statistic_distribution(
    #     "config/distribution/prep0_0_LABmaxdev5_max-sumnorm-dev.yaml"
    # )
    # t_statistic_distribution(
    #     "config/distribution/prep05_1_LABmaxdev5_max-maxnorm-dev.yaml"
    # )
    # t_statistic_distribution(
    #     "config/distribution/prep0_0_LABmaxdev5_max-maxnorm-dev.yaml"
    # )
