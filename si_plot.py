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
from t_statistic_distribution import score_sample, load_counts_windows
logger = logging.getLogger()
logger.setLevel(logging.WARNING)



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
    origCSEconf = cfg.CSEconf
    
    cfg.evaluate_the_worst_cases = False
    # set seed
    random.seed(a=cfg.seed, version=2)
    np.random.seed(cfg.seed)

    BE_arr=[]
    SE_arr=[]
    conts=[0.005, 0.002, 0.0015, 0.001]
    if "contaminations" in config.get_dict().keys():
        BE_avg = []
        SE_avg = []
        BE_std = []
        SE_std = []
        SI_avg = []
        SI_std = []
        SFI_avg = []
        SFI_std = []
        for c, path in zip(cfg.contaminations, cfg.cont_paths):
            if c in conts:
                print(c)
                cfg.CSEconf = [origCSEconf, "config/cs_eval/BE.yaml"]
                BE = score_sample(cfg, path)
                BE_arr.append(BE)
                cfg.CSEconf = [origCSEconf, "config/cs_eval/SE.yaml"]
                SE = score_sample(cfg, path)
                SE_arr.append(SE)
                plt.figure(1)
                plt.scatter(BE, SE, s=5, label=f"{c}")
                plt.figure(2)
                plt.scatter(1/BE, SE/np.sqrt(BE), s=5, label=f"{c}")
                plt.figure(3)
                plt.scatter(SE, 1/BE, s=5, label=f"{c}")
                plt.figure(4)
                plt.scatter(1/BE, SE/BE, s=5, label=f"{c}")
                print(BE)
                print(SE)
                BE_avg.append(np.mean(BE))
                SE_avg.append(np.mean(SE))
                BE_std.append(np.std(BE))
                SE_std.append(np.std(SE))
                cfg.CSEconf = [origCSEconf, "config/cs_eval/SI.yaml"]
                SI = score_sample(cfg, path)
                SE_arr.append(SI)
                SI_avg.append(np.nanmean(SI))
                SI_std.append(np.nanstd(SI))
                SFI_avg.append(np.nanmean(SE/BE))
                SFI_std.append(np.nanstd(SE/BE))

    
    plt.figure(1)
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    plt.xlabel("$e_b$")
    plt.ylabel("$e_s$")
    plt.grid()
    plt.legend()
    plt.savefig("plots/SI/ROC.pdf")

    plt.figure(2)
    plt.xlabel("$1/e_b$")
    plt.ylabel("SI")
    plt.xscale("log")
    plt.ylim(0, 3)
    plt.xlim(1, 1e3)
    plt.grid()
    plt.legend()
    plt.savefig("plots/SI/SIC.pdf")

    plt.figure(3)
    plt.yscale("log")
    plt.ylabel("$1/e_b$")
    plt.xlabel("$e_s$")
    plt.xlim(0, 1)
    plt.ylim(1, 1e4)
    plt.grid()
    plt.legend()
    plt.savefig("plots/SI/ROC2.pdf")

    plt.figure(4)
    plt.xscale("log")
    plt.xlabel("$1/e_b$")
    plt.ylabel("$e_s/e_b$")
    #plt.xlim(0, 1)
    #plt.ylim(1, 1e4)
    plt.grid()
    plt.legend()
    plt.savefig("plots/SI/SEC.pdf")

    plt.figure(5)
    #print(cfg.contaminations*100000)
    #print(SI_avg)
    plt.errorbar(conts, SI_avg, yerr=SI_std, fmt="o")
    plt.savefig("plots/SI/SI_cont.pdf")

    plt.figure(6)
    #print(cfg.contaminations*100000)
    #print(SFI_avg)
    plt.errorbar(conts, SFI_avg, yerr=SFI_std, fmt="o")
    plt.savefig("plots/SI/SfI_cont.pdf")



if __name__ == "__main__":
    # main plots v4 avriated signal ===============================================
    # Generate plots for all methods:
    methods = [#"config/distribution/v4/prep05_1_maxdev3_msdeCURTAINS_desamble.yaml"
               "config/distribution/v4/prep05_1_maxdev3_msdeCURTAINS_15max.yaml"]
    
    add_conf = "config/distribution/v4/bootstrap_sig_contam.yaml"

    methods = [[meth, add_conf] for meth in methods]

    for method in methods:
        t_statistic_distribution(method)

