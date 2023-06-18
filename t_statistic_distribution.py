# imports
import os
import copy
import logging
import scipy.stats as stats
import matplotlib.pyplot as plt
import pickle
from cs_performance_evaluation import cs_performance_evaluation, CS_evaluation_process
import numpy as np
import random
import time
from matplotlib.ticker import MaxNLocator
import cluster_scanning
import utils.set_matplotlib_default as smd
from load_old_bootstrap_experiments import (
    load_old_bootstrap_experiments05_1,
    load_old_bootstrap_experiments00,
)
from utils.config_utils import Config
from utils.binning_utils import default_binning
from utils.utils import (
    add_lists_in_dicts,
    p2Z,
    ensamble_means,
)

logger = logging.getLogger()
logger.setLevel(logging.WARNING)


def load_counts_windows(path):
    res = pickle.load(open(path, "rb"))
    if isinstance(res, list) or isinstance(res, np.ndarray):
        return res
    else:
        return res["counts_windows"]


def score_sample(cfg, counts_windows_boot_load):

    if counts_windows_boot_load == "old00":
        counts_windows_boot = load_old_bootstrap_experiments00()
        binning = default_binning()
    elif counts_windows_boot_load == "old05_1":
        counts_windows_boot = load_old_bootstrap_experiments05_1()
        binning = default_binning()
    else:
        print(len(os.listdir(counts_windows_boot_load)))
        files_list = os.listdir(counts_windows_boot_load)
        bres_files = [file for file in files_list if file.startswith("bres")]
        counts_windows_boot = []
        for file in bres_files:
            counts_windows_boot.append(
                load_counts_windows(counts_windows_boot_load + file)
            )
        binning = pickle.load(open(counts_windows_boot_load + "binning.pickle", "rb"))
    tstat_list = []
    for i, counts_windows in enumerate(counts_windows_boot):
        # counts_windows = np.array(counts_windows) #DELETE THIS
        tstat_list.append(
            cs_performance_evaluation(
                counts_windows=counts_windows,
                binning=binning,
                config_file_path=cfg.CSEconf,
            )
        )
        if i % 100 == 0:
            print(i)

    tstat_list = np.array(tstat_list)

    # plot the worst cases
    if cfg.evaluate_the_worst_cases:
        worst_cases = np.argsort(tstat_list)[-10:]
        bres_files = np.array(bres_files)
        print("worsdt_cases", bres_files[worst_cases], tstat_list[worst_cases])
        for i in worst_cases:
            counts_windows = pickle.load(
                open(counts_windows_boot_load + bres_files[i], "rb")
            )
            os.makedirs(counts_windows_boot_load + "worst_cases/", exist_ok=True)
            if isinstance(cfg.CSEconf, str):
                config_file_path = [cfg.CSEconf, "config/cs_eval/plotting.yaml"]
            else:
                config_file_path = cfg.CSEconf + ["config/cs_eval/plotting.yaml"]
            cs_performance_evaluation(
                counts_windows=counts_windows,
                binning=binning,
                config_file_path=config_file_path,
                ID=bres_files[i],
                path=counts_windows_boot_load + "worst_cases/",
            )

    return tstat_list


def p_value(stat, tstat_list):
    return (
        (np.sum(tstat_list >= stat) + np.sum(tstat_list > stat)) / 2 / len(tstat_list)
    )


def draw_contamination(
    cfg,
    c,
    path,
    col,
    tstat_list,
    old=False,
    postfix="",
    style="all",
    fig=None,
):

    # load restarts
    binning = None
    counts_windows_list = []
    if old == 1:
        for jj in range(10):
            res = pickle.load(open(path + "res{0:04d}.pickle".format(jj), "rb"))
            counts_windows = np.array(res["counts_windows"][0])
            binning = default_binning()
            counts_windows_list.append(counts_windows)
    elif old == 2:
        for jj in range(10):
            cs = cluster_scanning.ClusterScanning(path)
            cs.load_mjj()
            cs.ID = jj
            cs.load_results(jj)
            cs.sample_signal_events()
            cs.bootstrap_resample()
            counts_windows = cs.perform_binning()
            binning = default_binning()
            counts_windows_list.append(counts_windows)
    else:
        files_list = os.listdir(path)
        bres_files = [file for file in files_list if file.startswith("bres")]
        for file in bres_files:
            counts_windows = load_counts_windows(path + file)
            if binning is None:
                binning = pickle.load(open(path + "binning.pickle", "rb"))
            counts_windows_list.append(counts_windows)

    # evaluate TS for each
    arr = []
    ps = []
    os.makedirs(path + "evalutions/", exist_ok=True)
    for jj, counts_windows in enumerate(counts_windows_list):
        if cfg.do_plotting_in_contamination_evaluation:
            if isinstance(cfg.CSEconf, str):
                config_file_path = [cfg.CSEconf, "config/cs_eval/plotting.yaml"]
            else:
                config_file_path = cfg.CSEconf + ["config/cs_eval/plotting.yaml"]
            res = cs_performance_evaluation(
                counts_windows=counts_windows,
                binning=binning,
                config_file_path=config_file_path,
                path=path + "evalutions/",
                ID=bres_files[jj],
            )
        else:
            res = cs_performance_evaluation(
                counts_windows=counts_windows,
                binning=binning,
                config_file_path=cfg.CSEconf,
            )
        # print(res["chisq_ndof"])
        arr.append(res)

    # use ensamble if needed
    arr = np.array(arr)
    # if cfg.ensamble != 0 and cfg.ensamble != None:
    #    arr = ensamble_means(arr, cfg.ensamble)

    for res in arr:
        ps.append(p_value(res, tstat_list))
    meanres_ps = p_value(np.mean(arr), tstat_list)
    mean_ps = np.mean(ps)
    label = "$\epsilon$={:.4f}".format(c)
    if mean_ps == 0:
        label += ", $<p><${:.4f}, Z>{:2f}".format(
            1 / len(tstat_list), p2Z(np.mean(1 / len(tstat_list)))
        )
        mean_ps = 1 / len(tstat_list)
        upper_bound_mps = True
    else:
        label += ", $<p>=${:.4f}, Z={:2f}".format(np.mean(ps), p2Z(np.mean(ps)))
        upper_bound_mps = False

    if np.mean(meanres_ps) == 0:
        label += "\n p(<x>)<{:.4f} Z>{:.2f}".format(
            1 / len(tstat_list),
            p2Z(np.mean(1 / len(tstat_list))),
        )
        meanres_ps = 1 / len(tstat_list)
        upper_bound_mrps = True
    else:
        label += "\n p(<x>)={:.4f} Z={:.2f}".format(
            meanres_ps, p2Z(np.mean(meanres_ps))
        )
        upper_bound_mrps = False

    label += " " + postfix

    # the actual plotting
    plt.figure(fig)
    if style[0] == "U":
        style = style[1:]
    if style[:8] == "mean_std":
        plt.axvline(np.mean(arr), color=col, label=label)
        plt.axvspan(
            np.mean(arr) - np.std(arr),
            np.mean(arr) + np.std(arr),
            color=col,
            alpha=0.15,
        )
        style = style[8:]
    if style == "_meanstd":
        plt.errorbar(
            np.mean(arr),
            0,
            xerr=np.std(arr) / np.sqrt(len(arr)),
            capsize=2,
            color=col,
        )

    elif style == "all":
        for i, a in enumerate(arr):
            if i == 0:
                plt.axvline(a, color=col, label=label, alpha=0.3)
            else:
                plt.axvline(a, color=col, alpha=0.3)
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

    results = {}
    results["ps"] = ps
    results["p_upper_bound"] = 1 / len(tstat_list)
    results["mean_ps"] = mean_ps
    results["upper_bound_mps"] = upper_bound_mps
    results["meanres_ps"] = meanres_ps
    results["upper_bound_mrps"] = upper_bound_mrps
    results["Zs"] = p2Z(ps)
    results["Z_mean_ps"] = p2Z(mean_ps)
    results["Z_meanres_ps"] = p2Z(meanres_ps)

    for key in copy.deepcopy(list(results.keys())):
        results[key + "Z"] = [p2Z(results[key])]
        results[key] = [results[key]]
    return results


def t_statistic_distribution(config_file_path):
    config = Config(config_file_path)
    cfg = config.get_dotmap()

    # set defaults if not given in config
    if not hasattr(cfg, "ensamble"):
        cfg.ensamble = 0

    # set seed
    random.seed(a=cfg.seed, version=2)
    np.random.seed(cfg.seed)

    # Replace all these where needed
    output_path = cfg.output_path

    # initialise the main figure:
    plt.close("all")
    if cfg.contamination_style[0] == "U":
        fig = plt.figure(figsize=(6, 5))
        gs = fig.add_gridspec(2, hspace=0, height_ratios=[4, 1])
        (ax1, ax2) = gs.subplots(sharex=True)
        plt.sca(ax1)
    else:
        fig = plt.figure(figsize=(6, 3))

    # load the restarts
    if isinstance(cfg.counts_windows_boot_load, str):
        # if one location is given
        TS_list = score_sample(cfg, cfg.counts_windows_boot_load)
    else:
        # if multiple locations are given
        TS_list = []
        for counts_windows_boot_load in cfg.counts_windows_boot_load:
            TS_list += score_sample(cfg, counts_windows_boot_load)

    # ensamble if needed
    TS_list = np.array(TS_list)
    if cfg.ensamble != 0 and cfg.ensamble != None:
        TS_list = ensamble_means(TS_list, cfg.ensamble)

    # plot the TS distribution
    plt.figure(fig)
    if cfg.density:
        plt.hist(TS_list, bins=40, density=True, alpha=0.5)
    else:
        plt.hist(TS_list, bins=40, alpha=0.5)

    plt.figure(fig)
    if cfg.density:
        plt.ylabel("Density")
    else:
        plt.ylabel("Tries")
    # plt.axvline(np.mean(TS_list), color="blue", label=r"Average for $H_0$")
    TS_list = np.array(TS_list)
    print("mean", np.mean(TS_list))
    print("mean non-0", np.mean(TS_list[TS_list > 0]))
    print("std", np.std(TS_list))
    ndof = 7  # 2/(np.std(TS_list))**2
    # plt.plot(x, scipy.stats.chi2.pdf(x*ndof, df=ndof)*ndof)

    if "chi_df" in config.get_dict().keys():
        df = cfg.chi_df
        rv = stats.chi2(df)
        x = np.linspace(np.min(TS_list), np.max(TS_list), 100)
        y = rv.pdf(x * df) * df
        tr = 1e-3
        plt.plot(x[y > tr], y[y > tr], lw=2)

    #%%
    # Contaminations
    plt.title(cfg.title)
    plt.yscale("log")

    if cfg.contamination_style[0] == "U":
        plt.sca(ax2)

    results = {}
    if "contaminations" in config.get_dict().keys():
        for c, path, col, old, postfix in zip(
            cfg.contaminations, cfg.cont_paths, cfg.colors, cfg.old_CS, cfg.postfix
        ):
            results_1con = draw_contamination(
                cfg,
                c,
                path,
                col,
                TS_list,
                old=old,
                postfix=postfix,
                style=cfg.contamination_style,
                fig=fig,
            )
            results_1con["contaminations"] = [c]
            results = add_lists_in_dicts(results, results_1con)

    if cfg.contamination_style[0] == "U":
        handles, labels = ax2.get_legend_handles_labels()
        plt.sca(ax1)
        plt.legend(handles, labels, loc=1)
        plt.sca(ax2)
    else:
        plt.legend(loc=1)
    plt.xlabel(cfg.xlabel)

    # plt.xlim((0, 30))
    plt.savefig(
        output_path + cfg.plot_name,
        dpi=300,
        bbox_inches="tight",
    )
    # plt.show()

    # Dump some results
    results["TS_list"] = TS_list
    pickle.dump(results, open(output_path + cfg.plot_name + "_results.pickle", "wb"))


if __name__ == "__main__":
    # main plots v4 ===============================================================
    t_statistic_distribution("config/distribution/v4/prep05_1_maxdev5CURTAINS.yaml")

    # main plots ensambling ===============================================================
    # t_statistic_distribution(
    #     "config/distribution/ensambling/prep05_1_maxdev5CURTAINS_E20.yaml"
    # )
    # t_statistic_distribution(
    #     "config/distribution/ensambling/prep05_1_maxdev5CURTAINS_E5.yaml"
    # )
    # main plots ===============================================================

    # main plots ===============================================================
    # t_statistic_distribution("config/distribution/prep05_1_maxdev5_0005.yaml")
    # t_statistic_distribution("config/distribution/prep05_1_maxdev5.yaml")
    # t_statistic_distribution("config/distribution/prep05_1_2meansder.yaml")

    # t_statistic_distribution("config/distribution/prep05_1_maxdev5CURTAINS_0005.yaml")
    # t_statistic_distribution("config/distribution/prep05_1_maxdev5CURTAINS.yaml")
    # t_statistic_distribution("config/distribution/prep05_1_2meansderCURTAINS.yaml")
    # main plots ensismbling ===============================================================

    # comparison with old distributions ========================================

    # t_statistic_distribution(
    #     r"config\distribution\compare\compare_old_to_new0.5_1maxdev5.yaml"
    # )
    # t_statistic_distribution(r"config\distribution\compare\prep05_1_maxdev5_COMP.yaml")
    # t_statistic_distribution(r"config\distribution\compare\prep05_1_maxdev5_COMPR.yaml")
    # t_statistic_distribution(
    #     r"config\distribution\compare\prep05_1_maxdev5_0005_inits_copy.yaml"
    # )
    # t_statistic_distribution(
    #     r"config\distribution\compare\prep05_1_maxdev5CURTAINS_0005_inits.yaml"
    # )
    # comparison with old distributions ========================================

    # ========== no Aggregation TS =========================================
    # t_statistic_distribution(
    #     r"config\distribution\NA\prep05_1_NA-sumnorm-maxC-maxM.yaml"
    # )
    # t_statistic_distribution(
    #     r"config\distribution\NA\prep05_1_NA-sumnorm-chiC-chiM.yaml"
    # )
    # t_statistic_distribution(
    #     r"config\distribution\NA\prep05_1_NA-sumnorm-chiC-maxM.yaml"
    # )
    # t_statistic_distribution(
    #     r"config\distribution\NA\prep05_1_NA-sumnorm-maxC-chiM.yaml"
    # )

    # ========== no Aggregation TS =========================================

    # ========== max diff and dev TS's =========================================
    # t_statistic_distribution(
    #     r"config\distribution\prep05_1_maxdev5_MMD_CURTAINS_0005.yaml"
    # )
    # t_statistic_distribution(
    #     r"config\distribution\prep05_1_maxdev5_MMDiff_CURTAINS_0005.yaml"
    # )
    # t_statistic_distribution(
    #     r"config\distribution\prep05_1_maxdev5_MSD_CURTAINS_0005.yaml"
    # )
    # t_statistic_distribution(
    #     r"config\distribution\prep05_1_maxdev5_MSDiff_CURTAINS_0005.yaml"
    # )
    # t_statistic_distribution(r"config\distribution\prep05_1_maxdev5_MSD_CURTAINS.yaml")
    # ========== max diff and dev TS's =========================================

    # ========== With all the plots =========================================
    # t_statistic_distribution(
    #     [
    #         "config/distribution/prep05_1_maxdev5_0005.yaml",
    #         "config/distribution/plot.yaml",
    #     ]
    # )
    # t_statistic_distribution(
    #     [
    #         "config/distribution/prep05_1_maxdev5CURTAINS.yaml",
    #         "config/distribution/plot.yaml",
    #     ]
    # )

    # ========== With all the plots =========================================

    # t_statistic_distribution("config/distribution/prep0_0_LABkmeans_der.yaml")
    # t_statistic_distribution("config/distribution/compare/compare_old_to_new00.yaml")
    # t_statistic_distribution("config/distribution/compare/compare_old_to_new0.5_1.yaml")
    # t_statistic_distribution("config/distribution/prep0_0_LABmaxdev5CURTAINS.yaml")
    # t_statistic_distribution("config/distribution/prep05_1_LABmaxdev5CURTAINS.yaml")
    # start_time = time.time()
    # t_statistic_distribution("config/distribution/prep05_1_LABmaxdev5.yaml")
    # print("done in time --- %s seconds ---" % (time.time() - start_time))
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
