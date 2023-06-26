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
from cluster_scanning import ClusterScanning

logger = logging.getLogger()
logger.setLevel(logging.WARNING)


def load_counts_windows(path):
    res = pickle.load(open(path, "rb"))
    if isinstance(res, list) or isinstance(res, np.ndarray):
        return res
    else:
        return res["counts_windows"]


def score_sample(cfg, counts_windows_boot_load, do_wors_cases=True):

    print(len(os.listdir(counts_windows_boot_load)))
    files_list = os.listdir(counts_windows_boot_load)
    print(files_list)
    bres_files = [file for file in files_list if file.startswith("bres")]
    counts_windows_boot = []
    IDs_array = []
    for file in bres_files:
        counts_windows_boot.append(load_counts_windows(counts_windows_boot_load + file))
        IDs = ClusterScanning.IDstr_to_IDs(os.path.basename(file))
        IDs_array.append([IDs[0], IDs[2]])  # ignore the signal ID here
    IDs_array = np.stack(IDs_array)
    print(
        f"bootstrap indices go from {np.min(IDs_array[:,0])} to {np.max(IDs_array[:,0])}"
    )
    print(
        f"initialisation indices go from {np.min(IDs_array[:,1])} to {np.max(IDs_array[:,1])}"
    )

    binning = pickle.load(open(counts_windows_boot_load + "binning.pickle", "rb"))

    ensampbling = cfg.ensambling_num
    print("doing ensambling", ensampbling)
    if ensampbling == "desamble":
        tstat_array = []
        for i, counts_windows in enumerate(counts_windows_boot):
            # counts_windows = np.array(counts_windows) #DELETE THIS
            tstat_array.append(
                cs_performance_evaluation(
                    counts_windows=counts_windows,
                    binning=binning,
                    config_file_path=cfg.CSEconf,
                )
            )
            if i % 100 == 0:
                print(i)
            # if i > 1000:  # DELETE THIS
            #    break
        tstat_array = np.array(tstat_array)
        tstat_ensembled = tstat_array
        print("There are ", len(tstat_ensembled), " valid tstats")
        print("There are ", sum(tstat_ensembled == 0), " 0 stats")
        print("There are ", sum(tstat_ensembled > 0), " >0 stats")
    else:
        tstat_array = np.full(
            (
                np.max(IDs_array[:, 0]) + 1 - np.min(IDs_array[:, 0]),
                np.max(IDs_array[:, 1]) + 1,
            ),
            np.nan,
        )
        for i, counts_windows in enumerate(counts_windows_boot):
            # counts_windows = np.array(counts_windows) #DELETE THIS
            tstat_array[
                IDs_array[i, 0] - np.min(IDs_array[:, 0]), IDs_array[i, 1]
            ] = cs_performance_evaluation(
                counts_windows=counts_windows,
                binning=binning,
                config_file_path=cfg.CSEconf,
            )
            if i % 100 == 0:
                print(i)
            # if i > 1000:  # DELETE THIS
            #    break
        tstat_ensembled = []
        valid_bootstraps = np.sum(~np.isnan(tstat_array), axis=1) >= ensampbling
        for val in np.where(valid_bootstraps)[0]:
            if cfg.ensambling_type == "mean":
                tstat_ensembled.append(
                    np.mean(tstat_array[val][~np.isnan(tstat_array[val])][:ensampbling])
                )
            elif cfg.ensambling_type == "median":
                tstat_ensembled.append(
                    np.median(
                        tstat_array[val][~np.isnan(tstat_array[val])][:ensampbling]
                    )
                )
        tstat_ensembled = np.array(tstat_ensembled)

    # plot the worst cases
    if (
        cfg.evaluate_the_worst_cases
        and cfg.ensambling_num == "desamble"
        and do_wors_cases
    ):
        worst_cases = np.argsort(tstat_array)[-10:]
        bres_files = np.array(bres_files)
        print("worsdt_cases", bres_files[worst_cases], tstat_array[worst_cases])
        for i in worst_cases:
            counts_windows = pickle.load(
                open(counts_windows_boot_load + bres_files[i], "rb")
            )["counts_windows"]
            os.makedirs(
                counts_windows_boot_load
                + f"worst_cases{os.path.basename(cfg.CSEconf)[:-5]}/",
                exist_ok=True,
            )
            if isinstance(cfg.CSEconf, str):
                config_file_path = [cfg.CSEconf, "config/cs_eval/plotting.yaml"]
            else:
                config_file_path = cfg.CSEconf + ["config/cs_eval/plotting.yaml"]
            cs_performance_evaluation(
                counts_windows=counts_windows,
                binning=binning,
                config_file_path=config_file_path,
                ID=bres_files[i],
                path=counts_windows_boot_load
                + f"worst_cases{os.path.basename(cfg.CSEconf)[:-5]}/",
            )

    return tstat_ensembled


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
    postfix="",
    style="all",
    fig=None,
):

    arr = score_sample(cfg, path, do_wors_cases=False)
    ps = []
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
        if cfg.ensambling_num == "desamble":
            label += "\n p(<x>)<{:.4f} Z>{:.2f}".format(
                1 / len(tstat_list),
                p2Z(np.mean(1 / len(tstat_list))),
            )
        meanres_ps = 1 / len(tstat_list)
        upper_bound_mrps = True
    else:
        if cfg.ensambling_num == "desamble":
            label += "\n p(<x>)={:.4f} Z={:.2f}".format(
                meanres_ps, p2Z(np.mean(meanres_ps))
            )
        upper_bound_mrps = False

    label += " " + postfix

    # the actual plotting
    # plt.figure(fig)
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

    # set seed
    random.seed(a=cfg.seed, version=2)
    np.random.seed(cfg.seed)

    # Replace all these where needed
    output_path = cfg.output_path

    # initialise the main figure:
    plt.close("all")
    if cfg.contamination_style[0] == "U":
        fig = plt.figure(figsize=(8, 5))
        gs = fig.add_gridspec(2, hspace=0, height_ratios=[4, 1])
        (ax1, ax2) = gs.subplots(sharex=True)
        plt.sca(ax1)
    else:
        fig = plt.figure(figsize=(8, 3))

    # load the restarts
    if isinstance(cfg.counts_windows_boot_load, str):
        # if one location is given
        TS_list = score_sample(cfg, cfg.counts_windows_boot_load)
    else:
        # if multiple locations are given
        TS_list = []
        for counts_windows_boot_load in cfg.counts_windows_boot_load:
            TS_list += score_sample(cfg, counts_windows_boot_load)

    # plot the TS distribution
    # plt.figure(fig)
    if cfg.density:
        plt.hist(TS_list, bins=40, density=True, alpha=0.5)
    else:
        plt.hist(TS_list, bins=40, alpha=0.5)

    # plt.figure(fig)
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
        for c, path, col, postfix in zip(
            cfg.contaminations, cfg.cont_paths, cfg.colors, cfg.postfix
        ):
            results_1con = draw_contamination(
                cfg,
                c,
                path,
                col,
                TS_list,
                postfix=postfix,
                style=cfg.contamination_style,
                fig=fig,
            )
            results_1con["contaminations"] = [c]
            results = add_lists_in_dicts(results, results_1con)

    if cfg.contamination_style[0] == "U":
        handles, labels = ax2.get_legend_handles_labels()
        plt.sca(ax1)
        plt.legend(handles, labels, loc="center left", bbox_to_anchor=(1, 0.5))
        # plt.tight_layout(rect=[0, 0, 0.8, 1])
        plt.sca(ax2)
    else:
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        # plt.tight_layout(rect=[0, 0, 0.8, 1])
    plt.xlabel(cfg.xlabel)
    plt.xlim(
        min(TS_list) - 0.1 * (max(TS_list) - min(TS_list)),
        max(TS_list) + 0.1 * (max(TS_list) - min(TS_list)),
    )
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
    # main plots v4 avriated signal ===============================================
    # Generate plots for all methods:
    methods = [
        # "config/distribution/v4/prep05_1_maxdev5CURTAINS_1mean.yaml",
        # "config/distribution/v4/prep05_1_maxdev5CURTAINS_15mean.yaml",
        # "config/distribution/v4/prep05_1_maxdev3CURTAINS_15mean.yaml",
        # "config/distribution/v4/prep05_1_maxdev3CURTAINS_15med.yaml",
        # "config/distribution/v4/prep05_1_maxdev3_msdeCURTAINS_15mean.yaml",
        "config/distribution/v4/prep05_1_maxdev3_msde_4parCURTAINS_15mean.yaml",
    ]

    add_conf = "config/distribution/v4/bootstrap_sig_contam.yaml"

    methods = [[meth, add_conf] for meth in methods]

    for method in methods:
        t_statistic_distribution(method)

    # main plots v4 ===============================================================
    # methods = [
    #     "config/distribution/v4/prep05_1_maxdev5CURTAINS_1mean.yaml",
    #     "config/distribution/v4/prep05_1_maxdev5CURTAINS_15mean.yaml",
    #     "config/distribution/v4/prep05_1_maxdev3CURTAINS_15mean.yaml",
    #     "config/distribution/v4/prep05_1_maxdev3CURTAINS_15med.yaml",
    #     "config/distribution/v4/prep05_1_maxdev3_msdeCURTAINS_15mean.yaml",
    #     "config/distribution/v4/prep05_1_maxdev3_msdeCURTAINS_1mean.yaml",
    #     "config/distribution/v4/prep05_1_maxdev5CURTAINS_desamble.yaml"
    #     "config/distribution/v4/prep05_1_maxdev3CURTAINS_desamble.yaml"
    #     "config/distribution/v4/prep05_1_maxdev3_msdeCURTAINS_desamble.yaml",
    #     "config/distribution/v4/prep05_1_maxdev5CURTAINS_0002_15mean.yaml",
    #     "config/distribution/v4/prep05_1_maxdev3CURTAINS_0002_15mean.yaml",
    #     "config/distribution/v4/prep05_1_maxdev3CURTAINS_0002_15med.yaml"
    # ]

    # add_conf = "config/distribution/v4/restart_sig_contam.yaml"

    # methods = [[meth, add_conf] for meth in methods]

    # for method in methods:
    #     t_statistic_distribution(method)

    # methods = [
    #     "config/distribution/v4/prep05_1_maxdev5CURTAINS_15mean.yaml",
    #     "config/distribution/v4/prep05_1_maxdev3CURTAINS_15mean.yaml",
    #     "config/distribution/v4/prep05_1_maxdev3CURTAINS_15med.yaml"
    # ]

    # add_conf = "config/distribution/v4/restart_sig_contam_0002.yaml"

    # methods = [[meth, add_conf] for meth in methods]

    # for method in methods:
    #     t_statistic_distribution(method)

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
