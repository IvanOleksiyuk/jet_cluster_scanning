import os
import subprocess
import shutil

directory = 'plots/pub_plot/'

os.makedirs(directory, exist_ok=True)

steps = ['dist' ] #'data', , , 'SI', 'main', 'one_run'

#data plots
if 'data' in steps:
	subprocess.run("python preproc/data_plotting.py config/pub_plot_path.yaml", check=True, shell=True)

#Run one CS experiment with 5K signal events and plot all the results sving them properly

if 'one_run' in steps:
	os.makedirs(directory+"algo", exist_ok=True)
	subprocess.run("python one_cs_run.py", check=True, shell=True)
	run_location="char/one_run_experiments/k50MB2048_1iret0con0.05W3000_3100_w0.5s1Nboot/binnedW100s16ei30004600/"
	shutil.copyfile(run_location+"plots_b0_s0_i0/kmeans_ni_mjj_total.png", 
					directory+"algo/kmeans_ni_mjj_total.png")
	shutil.copyfile(run_location+"plots_b0_s0_i0/kmeans_ni_mjj_norm.png", 
					directory+"algo/kmeans_ni_mjj_norm.png")
	shutil.copyfile(run_location+"eval_b0_s0_i0/sumn-toatal.png", 
					directory+"algo/sumn-total.png")
	shutil.copyfile(run_location+"eval_b0_s0_i0/norm-toatal-sigmas.png",
					directory+"algo/norm_threshold.png")
	shutil.copyfile(run_location+"eval_b0_s0_i0/comb.png", 
					directory+"algo/comb.png")

#Create the distribution plots with vertical lines for the cuts
if 'dist' in steps:
	from t_statistic_distribution import t_statistic_distribution
	t_statistic_distribution(["config/distribution/v4/prep05_1_maxdev3_msdeCURTAINS_15mean_ideal.yaml",
         "config/distribution/v4/bootstrap_sig_contam_ideal.yaml",
         "config/distribution/v4/plot_path.yaml"])
	t_statistic_distribution(["config/distribution/v4/prep05_1_maxdev3_msdeCURTAINS_15mean.yaml",
		"config/distribution/v4/bootstrap_sig_contam.yaml",
		"config/distribution/v4/plot_path.yaml"])

#Create the ROC, SF and SI plots

#Create all the main plot idealised 

#Create the mmain plot non-idealised 
