# jet_cluster_scanning

Ivan Oleksiyuk
ivan.oleksiyuk@gmail.com

These are my experiments concerning the anomaly detection using k-means to separate space of data instances (I mainly focuse on jet-images) into bins for bump hunting.

I am a beginer programmer so any constructive criticism is very wellcome.
THE PROJECT IS CURRENTLY FAR FROM RELEASE

DISCLAYMER: 
To use the project I recommend to creat a conda env with requiremnts from requirements.txt
all scripts in the root directory are intended to be executed from it 
all scripts in the job directory are intended to be executed from it
all tests are intended to be run via "pytest" command in the root directory
"SLURM" sections explain how to use this project with the cluster


User manual:
0 Preparation
	0.1 Clone the repository or a fork of it
	0.2 It is advisable to create a virtual environment (e.g. with conda) with python 3.7.15 and install requirements from requirements.txt
	0.3 Download the LHCO R&D dataset "events_anomalydetection.h5" from https://zenodo.org/record/4536377
	0.4 Tweak path.yaml according to your preferences 
	#0.5 ... tests not implemented yet
========================================================================================================================
1. Preprocessing 
	1.1 Run pyjet_clustering.py 
		to cluster jets and get some high-level bservables for LHCO events 
		input format 1.1M(events) x [700(particles)*3(p_t, eta, phi) + 1(label)] 
		ouput format 
	1.2	Run convert2images.py 
		to get a jet image dataset from the output of 1.1
	1.3 Run data_sort_mjj.py 
		to sort data by mjj and separate it into bkg and signal files used in the algorithms below 
	1.4 (Optionally) Run data_plott.py 
		to make some plots relevant for the data (some of them are in the publication)
========================================================================================================================
2.	Generate the H_0 runs
	Use cluster_scanning.py funcion with a base config and a bootstrap config 
	e.g. $python cluster_scanning.py config/s0_0.5_1_MB.yaml config/bootstrap/0_300.yaml
	to bootstrap resample a lot of background only samples and perform clustering on each resampling in order to later get a background hypothesis distribution of the 
	SLURM: If you are working on cluster I recommend using jobs/start_bootstrap_jobs.py script that will automatically start 5 jobs for 6h with 300 resamplings each (depending on the specifications of your cluster nodes I might not be able to train all 300 at once, you may restart the same script to continue training) e.g.
	cd jobs
	python start_bootstrap_jobs.py config/s0_0.5_1_MB.yaml config/tra_reg/3000_3100.yaml
========================================================================================================================
3. Genrate the signal contamination runs
	4.1	Use cluster_scanning.py funcion with base config and a sig_frac config 
	e.g. $python cluster_scanning.py config/s0_0.5_1_MB.yaml config/bootstrap/0_300.yaml
	to simmulare analysis in case of signal contamination
	4.2	Use cs_prformance_evaluateon.py with plotting function to visualise the process of test statistic evaluation for a givec cluster_scanning run
========================================================================================================================
4. (Optional) If you are not sattisified with binning that is performed by defolt after training you can rerun binning with different settings using binning.py
========================================================================================================================
5. 	Use t_statistic_distribution.py with configs
	... 
	to generate the null-hypothesis distribution of the test statistic and show the test statistic and the significance for the case of different levels of signal contaminations
========================================================================================================================
6. If you are benchmarking or searching for exclusion limits use significance_plot.py to plot the significances for different signal contaminations