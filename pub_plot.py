import os
import subprocess
import shutil
from t_statistic_distribution import t_statistic_distribution
import logging

#This script is used to create all the plots for the publication
#The script hs two main parts: the run of the CS and other stuff and the creation of the plots

#The steps are:
#1. Create the data plots

logger = logging.getLogger()
logger.setLevel(logging.INFO)


directory = 'plots/fig/'

os.makedirs(directory, exist_ok=True)

steps = ['one_run'] #'dist', 'main', 'data', , 'main', 'SI', 'main', 'one_run', 

#data plots
if 'data' in steps:
	subprocess.run("python preproc/data_plotting.py config/pub_plot_path.yaml", check=True, shell=True)

#Run one CS experiment with 5K signal events and plot all the results sving them properly

if 'aux' in steps:
    pass

if 'one_run' in steps:
	print("doing the one plots from one CS run with evaluation")
	os.makedirs(directory+"algo", exist_ok=True)
	from one_cs_run import one_cs_run
	config_cs = [
			"config/general.yaml",
			"config/path.yaml",
			"config/v4/s0_0.5_1_MB_i1.yaml",
			"config/sig_frac/0.05.yaml",
			"config/multirun/0_0_0.yaml",
			"config/binning/CURTAINS.yaml",
			"config/tra_reg/3000_3100.yaml",
			"config/one_run_experiments.yaml",
			#"config/save_memory.yaml"
		]
	config_ev = [
		"config/cs_eval/maxdev3_msders.yaml",
		"config/cs_eval/plotting.yaml"]
	one_cs_run(config_cs, config_ev)
	run_location="char/one_run_experiments/k50MB2048_1iret0con0.05W3000_3100_w0.5s1Nboot/binnedW100s16ei30004600/"
	shutil.copyfile(run_location+"plots_b0_s0_i0/kmeans_ni_mjj_total.pdf", 
					directory+"algo/kmeans_ni_mjj_total.pdf")
	shutil.copyfile(run_location+"plots_b0_s0_i0/kmeans_ni_mjj_norm.pdf", 
					directory+"algo/kmeans_ni_mjj_norm.pdf")
	shutil.copyfile(run_location+"eval_b0_s0_i0/sumn-toatal.pdf", 
					directory+"algo/sumn-total.pdf")
	shutil.copyfile(run_location+"eval_b0_s0_i0/norm-toatal-sigmas.pdf",
					directory+"algo/norm_threshold.pdf")
	shutil.copyfile(run_location+"eval_b0_s0_i0/comb.pdf", 
					directory+"algo/comb.pdf")

#Create the distribution plots with vertical lines for the cuts
if 'dist' in steps:
	os.makedirs(directory+"dist", exist_ok=True)
	add_list=["config/distribution/v4/plot_path.yaml"] #, "config/distribution/v4/small.yaml"
	t_statistic_distribution(["config/distribution/pub/prep05_1_maxdev3_msdersCURTAINS_1mean-pub.yaml",
		"config/distribution/v4/bootstrap_sig_contam.yaml"]+add_list)
	t_statistic_distribution(["config/distribution/pub/prep05_1_maxdev3_msdersCURTAINS_15mean_ideal-pub.yaml",
		"config/distribution/v4/bootstrap_sig_contam_ideal.yaml"]+add_list)
	t_statistic_distribution(["config/distribution/pub/prep05_1_maxdev3_msdersCURTAINS_15mean-pub.yaml",
		"config/distribution/v4/bootstrap_sig_contam.yaml"]+add_list)
	# t_statistic_distribution(["config/distribution/v4/prep05_1_maxdev3_msdeCURTAINS_desamble-pub.yaml",
	# 	"config/distribution/v4/bootstrap_sig_contam.yaml"]+add_list)
	shutil.copyfile("plots/ts_distribs/V4prep05_1_maxdev3_msdersCURTAINS_15mean_ideal.pdf",
					directory+"dist/V4prep05_1_maxdev3_msdersCURTAINS_15mean_ideal.pdf")
	shutil.copyfile("plots/ts_distribs/V4prep05_1_maxdev3_msdersCURTAINS_15mean.pdf", 
					directory+"dist/V4prep05_1_maxdev3_msdersCURTAINS_15mean.pdf")
	# shutil.copyfile("plots/ts_distribs/V4prep05_1_maxdev3_msdersCURTAINS_desamble.pdf", 
	# 				directory+"dist/V4prep05_1_maxdev3_msdersCURTAINS_desamble.pdf")
	shutil.copyfile("plots/ts_distribs/V4prep05_1_maxdev3_msdersCURTAINS_1mean.pdf",
		 			directory+"dist/V4prep05_1_maxdev3_msdersCURTAINS_1mean.pdf")									

#Create the ROC, SF and SI plots

#Create all the main plot idealised 
if 'main' in steps:
	os.makedirs(directory+"main", exist_ok=True)
	subprocess.run("python significance_plot.py", check=True, shell=True)
	shutil.copyfile("plots/main/significances_idealised.pdf", 
			directory+"main/significances_idealised.pdf")
	shutil.copyfile("plots/main/significances_realistic.pdf", 
				directory+"main/significances_realistic.pdf")


#Create the mmain plot non-idealised 
