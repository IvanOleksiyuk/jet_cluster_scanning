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
from pathlib import Path
from matplotlib.ticker import MaxNLocator
import cluster_scanning
import utils.set_matplotlib_default as smd
from utils.config_utils import Config
from utils.binning_utils import default_binning
from utils.utils import (
    add_lists_in_dicts,
    p2Z,
    ensamble_means,
    p_value,
)
from cluster_scanning import ClusterScanning

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def process_binning_files(cfg, binning_files_aray, counts_windows_boot_load, binning, checkpoint_name=None, config_file_path=None, start_boot=0,  end_boot=None, checkpoint=None):
    # Handle the config file path
    if cfg is None:
        cfg = Config(config_file_path).get_dotmap()
    
    checkpoint = None
    if checkpoint_name is not None:
        if os.path.exists(counts_windows_boot_load+checkpoint_name+".npy"):
            logging.warning(f"Loading checkpoint from {counts_windows_boot_load+checkpoint_name}.npy")
            checkpoint = np.load(counts_windows_boot_load+checkpoint_name+".npy")
            logging.warning(f"checkpoint has {np.sum(np.logical_not(np.isnan(checkpoint)))} processed binnings")
        else:
            logging.warning(f"CHECKPOINT NOT FOUND {counts_windows_boot_load+checkpoint_name}")
    
    n_bootstraps = binning_files_aray.shape[0]
    n_ensambling = binning_files_aray.shape[1]
    tstat_array = np.full(
        (
            n_bootstraps,
            n_ensambling,
        ),
        np.nan,
    )
    if checkpoint is not None:
        tstat_array = update_values(checkpoint, tstat_array)
    if end_boot is None:
        end_boot = n_bootstraps
    for i in range(start_boot, end_boot):
        changes_made=0
        for j in range(n_ensambling):
            if not np.isnan(tstat_array[i, j]):
                continue
            if binning_files_aray[i, j]=="":
                continue
            if os.path.exists(counts_windows_boot_load+checkpoint_name):
                ID = binning_files_aray[i, j]
                ID = ID[5:]
                ID = ID[:-7]
                if os.path.exists(counts_windows_boot_load+checkpoint_name+f"/t_stat{ID}.npy"):
                    tstat_array[i, j] = float(np.load(counts_windows_boot_load+checkpoint_name+f"/t_stat{ID}.npy"))
            logger.debug(f"loading {counts_windows_boot_load + binning_files_aray[i, j]}")
            counts_windows = load_counts_windows(counts_windows_boot_load + binning_files_aray[i, j])
            tstat_array[i, j] = cs_performance_evaluation(
                counts_windows = counts_windows, 
                binning=binning,
                config_file_path=cfg.CSEconf,)
            changes_made+=1
        if i % 100 == 0:
            logger.info(f"{i} bottstraps processed")
            if checkpoint_name is not None:
                if changes_made>0:
                    np.save(counts_windows_boot_load+checkpoint_name, tstat_array)
                    logging.warning(f"checkpoint updated {counts_windows_boot_load+checkpoint_name}")

    if checkpoint_name is not None:
        np.save(counts_windows_boot_load+checkpoint_name, tstat_array)
        logging.warning(f"checkpoint saved {counts_windows_boot_load+checkpoint_name}")

    tstat_array = np.array(tstat_array)
    return tstat_array

def update_values(smaller_array, larger_array):
    # Get the shape of the smaller array
    smaller_shape = smaller_array.shape
    
    # Get the slices to select corresponding elements from the larger array
    slices = tuple(slice(0, dim) for dim in smaller_shape)
    
    # Update values in the larger array with corresponding values from the smaller array
    larger_array[slices] = smaller_array
    
    return larger_array


def load_counts_windows(path):
    res = pickle.load(open(path, "rb"))
    if isinstance(res, list) or isinstance(res, np.ndarray):
        return res
    else:
        return res["counts_windows"]



def get_filename_without_extension(file_path):
    filename_with_extension = os.path.basename(file_path)
    filename_without_extension = os.path.splitext(filename_with_extension)[0]
    return filename_without_extension

def score_sample(cfg, counts_windows_boot_load, do_wors_cases=True):
    print("searching binnings in:", counts_windows_boot_load)
    print("found binnings total:", len(os.listdir(counts_windows_boot_load)))
    print("CSE config", cfg.CSEconf)
    files_list = os.listdir(counts_windows_boot_load)
    bres_files = [file for file in files_list if file.startswith("bres")]
    ensambling = cfg.ensambling_num
    print("doing ensambling", ensambling)

    # Get the IDs of the files as a list
    counts_windows_boot = []
    IDs_array = []
    for file in bres_files:
        IDs = ClusterScanning.IDstr_to_IDs(os.path.basename(file))
        IDs_array.append([IDs[0], IDs[2]])  # ignore the signal ID here
    IDs_array = np.stack(IDs_array)

    # Find the span of the indices
    print(f"bootstrap indices go from {np.min(IDs_array[:,0])} to {np.max(IDs_array[:,0])}")
    print(f"initialisation indices go from {np.min(IDs_array[:,1])} to {np.max(IDs_array[:,1])}")

    binning = pickle.load(open(counts_windows_boot_load + "binning.pickle", "rb"))
    
    if ensambling == "desamble":
        tstat_array = []
        for i, file in enumerate(bres_files):
            print(f"loading {file}")
            logger.info(f"loading {file}")
            counts_windows_boot.append(load_counts_windows(counts_windows_boot_load + file))
            tstat_array.append(
                cs_performance_evaluation(
                    counts_windows=counts_windows_boot[-1],
                    binning=binning,
                    config_file_path=cfg.CSEconf,
                )
            )
            if i % 100 == 0:
                print(i)
            if hasattr(cfg, "max_i"):
                if i > cfg.max_i:  # DELETE THIS
                    break
        tstat_array = np.array(tstat_array)
        tstat_ensembled = tstat_array
        print("There are ", len(tstat_ensembled), " valid tstats")
        print("There are ", sum(tstat_ensembled == 0), " 0 stats")
        print("There are ", sum(tstat_ensembled > 0), " >0 stats")
    else:
        # create a 2D table of files and fill it with the file names
        tstat_files = np.full(
            (
                np.max(IDs_array[:, 0]) + 1,
                np.max(IDs_array[:, 1]) + 1,
            ),
            "",
            dtype='<U100',
        )
        for bres_file, indices in zip(bres_files, IDs_array):
            tstat_files[indices[0], indices[1]] = bres_file

        # Find the bootstraps that have a sufficient number of initialisations
        if hasattr(cfg, "small"):
            if cfg.small:
                tstat_files = tstat_files[:cfg.small]

        valid_bootstraps = np.sum(np.logical_not(tstat_files==""), axis=1) >= ensambling

        print(f"Number of available bootstraps {sum(valid_bootstraps)}")
        
        n_valid_bootstraps = sum(valid_bootstraps)

        if hasattr(cfg, "select_then_process"):
            # First select the bootstraps and then process them
            tstat_files_new = np.full(
                (
                    n_valid_bootstraps,
                    ensambling,
                ),
                "",
                dtype='<U100',
            )
            for i, val in enumerate(np.where(valid_bootstraps)[0][:n_valid_bootstraps]):
                tstat_files_new[i] = tstat_files[val][np.logical_not(tstat_files[val]=="")][:ensambling]
            tstat_array_select = process_binning_files(cfg, tstat_files_new, counts_windows_boot_load, binning, checkpoint_name=None)
        else:
            # Load tstat_array if it exists
            tstat_array_name = get_filename_without_extension(cfg.CSEconf)
            tstat_array = process_binning_files(cfg, tstat_files, counts_windows_boot_load, binning, checkpoint_name=tstat_array_name)
            tstat_array_select = np.full(
                (
                    n_valid_bootstraps,
                    ensambling,
                ),
                np.nan,
            )
            for i, val in enumerate(np.where(valid_bootstraps)[0]):
                tstat_array_select[i] = tstat_array[val][np.logical_not(tstat_files[val]=="")][:ensambling]
        
        if cfg.ensambling_type == "mean":
            tstat_ensembled= np.mean(tstat_array_select, axis=1)                
        elif cfg.ensambling_type == "median":
            tstat_ensembled=np.median(tstat_array_select, axis=1)
        elif cfg.ensambling_type == "max":
            tstat_ensembled=np.nanmax(tstat_array_select, axis=1)
        elif cfg.ensambling_type == "min":
            tstat_ensembled=np.nanmin(tstat_array_select, axis=1)

    if np.any(np.isnan(tstat_ensembled)):
        raise ValueError("There are NaNs in the tstat_ensembled")
    
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


def draw_contamination(
    cfg,
    c,
    path,
    col,
    tstat_list,
    postfix="",
    style="all",
    fig=None,
    max_ts=np.inf,
):

	arr = score_sample(cfg, path, do_wors_cases=False)
	ps = []
	for res in arr:
		ps.append(p_value(res, tstat_list))
	meanres_ps = p_value(np.mean(arr), tstat_list)
	mean_ps = np.mean(ps)
	c=int(c*1000000)
	label = r"$\epsilon$"+f"={c}"
	
	p_min=1/len(tstat_list)
	Z_max = p2Z(p_min)
	if hasattr(cfg, "labelling"):
		if cfg.labelling=="Z":
			add_p = False
			add_Z = True			
	else:
		add_p = True
		add_Z = True

	p_good = np.array(ps).copy()
	p_good[p_good < p_min] = p_min
	p_median = np.median(p_good)
	Z_median = p2Z(p_median)


	if p_median == p_min:
		if add_p:
			label += r", $p_{med} <$"+"{:.2e}".format(p_min)
		if add_Z:
			label += r", $Z_{med} >$"+"{:.2f}".format(Z_max)
		upper_bound_mps = True
	else:
		if add_p:
			label += r", $p_{med} =$"+"{:.2e}".format(p_median)	
		if add_Z:
			label += r", $Z_{med} =$"+"{:.2f}".format(Z_median)	
		upper_bound_mps = False

	# if mean_ps < p_min:
	# 	if add_p:
	# 		label += ", $<p> <${:.2e}".format(p_min)
	# 	if add_Z:
	# 		label += ", $<Z> <${:2f}".format(Z_max)
	# 	mean_ps = p_min
	# 	upper_bound_mps = True
	# else:
	# 	if add_p:
	# 		label += ", $<p> =${:.2e}".format(np.mean(ps))
	# 	if add_Z:
	# 		label += ", $<Z> =${:2f}".format(np.mean(p2Z(ps)))
	# 	upper_bound_mps = False

	if np.mean(meanres_ps) < p_min:
		if cfg.ensambling_num == "desamble":
			label += "\n p(<x>)<{:.4f} Z>{:.2f}".format(
				p_min,
				p2Z(np.mean(p_min)),
			)
		meanres_ps = p_min
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

	if col != "none" and np.quantile(arr, 0.25)<max(tstat_list)*1.1:
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


def t_statistic_distribution(config_file_path, from_results=False):
    """ script that calculates and plots the test statistic distribution for CS"""
    logging.info("==================================")
    logging.info("T statistic distribution")
    logging.info("==================================")
    config = Config(config_file_path)
    cfg = config.get_dotmap()

    # set seed
    random.seed(a=cfg.seed, version=2)
    np.random.seed(cfg.seed)

    # Replace all these where needed
    output_path = cfg.output_path

    logging.info(f"output_path {output_path}")
    
    # initialise the main figure:
    plt.close("all")
    if cfg.contamination_style[0] == "U":
        fig = plt.figure(figsize=(8, 5))
        gs = fig.add_gridspec(2, hspace=0, height_ratios=[4, 1])
        (ax1, ax2) = gs.subplots(sharex=True)
        plt.sca(ax1)
    else:
        fig = plt.figure(figsize=(8, 3))

    if from_results:
        logging.info("Skipping background evaluation by loading it from checkpoint")
        results_loaded = pickle.load(open(output_path + cfg.plot_name + "_results.pickle", "rb"))
        TS_list = results_loaded["TS_list"]
    else:
        # load the restarts
        if isinstance(cfg.counts_windows_boot_load, str):
            # if one location is given
            TS_list = score_sample(cfg, cfg.counts_windows_boot_load)
        else:
            # if multiple locations are given
            TS_list = []
            for counts_windows_boot_load in cfg.counts_windows_boot_load:
                TS_list += [score_sample(cfg, counts_windows_boot_load)]

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
        plt.ylabel("Pseudo-experiments")
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
    else:
        logging.info("NO CONTAMINATIONS FOUND TO PROCESS")
            
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
    plt.savefig(
        output_path + cfg.plot_name+".pdf",
        dpi=250,
        bbox_inches="tight",
    )
    plt.savefig(
        output_path + cfg.plot_name+".png",
        dpi=250,
        bbox_inches="tight",
    )
    print("saved "+output_path + cfg.plot_name)
    
    if not from_results:
        results["TS_list"] = TS_list
        pickle.dump(results, open(output_path + cfg.plot_name + "_results.pickle", "wb"))
    print("saved "+ output_path + cfg.plot_name + "_results.pickle")
    return results 


if __name__ == "__main__":
    
    # t_statistic_distribution(
	# 		["test/config/prep05_1_maxdev3_msdeCURTAINS_1mean.yaml",
	# 		"test/config/bootstrap_sig_contam_ideal.yaml",
	# 		"test/config/plot_path2.yaml",
	# 		"test/config/small.yaml"]
	# 	)	

    t_statistic_distribution(
			["config/distribution/responce/prep05_1_maxdev3_msders+3fitCURTAINS_15mean.yaml",
			"config/distribution/v4/bootstrap_sig_contam_old.yaml",
			"test/config/plot_path2.yaml"]
		)

    # t_statistic_distribution(
	# 		["config/distribution/responce/prep05_1_maxdev3_msdersCURTAINS_15mean_sig_reg.yaml",
	# 		"config/distribution/responce/bootstrap_sig_contam_all_sig_reg.yaml",
	# 		"test/config/plot_path2.yaml"]
	# 	)
 

    # t_statistic_distribution(
	# 		["config/distribution/responce/prep05_1_maxdev3_msdersCURTAINS_15mean_ignore_signal.yaml",
	# 		"config/distribution/v4/bootstrap_sig_contam_ignore_signal.yaml",
	# 		"test/config/plot_path2.yaml"]
	# 	)
    
    
    # t_statistic_distribution(
    #     ["config/distribution/v4/prep05_1_maxdev3_msdeCURTAINS_1mean.yaml",
    #      "config/distribution/v4/bootstrap_sig_contam_ideal.yaml",
    #      "config/distribution/v4/plot_path2.yaml",
    #      "config/distribution/v4/small.yaml"]
    # )

    
    # t_statistic_distribution(
    #     "config/distribution/v4/prep05_1_maxdev3_msdeCURTAINS_desamble_fake_boot_compare.yaml"
    # )
    # t_statistic_distribution(
    #     "config/distribution/v4/prep05_1_maxdev3_msdeCURTAINS_desamble_fake_boot.yaml"
    # )

    # main plots v4 avriated signal ===============================================
    # Generate plots for all methods:
    # methods = [
    #     # "config/distribution/v4/prep05_1_maxdev5CURTAINS_1mean.yaml",
    #     # "config/distribution/v4/prep05_1_maxdev5CURTAINS_15mean.yaml",
    #     # "config/distribution/v4/prep05_1_maxdev3CURTAINS_15mean.yaml",
    #     # "config/distribution/v4/prep05_1_maxdev3CURTAINS_15med.yaml",
    #     # "config/distribution/v4/prep05_1_maxdev3_msdeCURTAINS_15mean.yaml",
    #     "config/distribution/v4/prep05_1_maxdev3_msde_4parCURTAINS_15mean.yaml",
    # ]

    # add_conf = "config/distribution/v4/bootstrap_sig_contam.yaml"

    # methods = [[meth, add_conf] for meth in methods]

    # for method in methods:
    #     t_statistic_distribution(method)

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

# %%
