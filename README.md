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

Short algorythm on how to use the prject:
0. 	extract jet constituents (e.g. LHCO) (not realised in this project (DIY if needed I guess)) and masses of the corresponding events in the other file 
1.	Use ... to turn (e.g. LHCO) jet constituent data into data vectors (I only provide scripts for jet images)
2.	Use ... to sort data vectors (e.g. flattened jet images) in the datasets with respect to the events invariant mass (e.g. jj_mass)
3.	Use cluster_scanning.py funcion with a base config and a bootstrap config 
e.g. $python cluster_scanning.py config/s0_0.5_1_MB.yaml config/bootstrap/0_300.yaml
to bootstrap resample a lot of background only samples and perform clustering on each resampling in order to later get a background hypothesis distribution of the 
SLURM: If you are working on cluster I recommend using jobs/start_bootstrap_jobs.py script that will automatically start 5 jobs for 6h with 300 resamplings each (depending on the specifications of your cluster nodes I might not be able to train all 300 at once, you may restart the same script to continue training)
4.1	Use cluster_scanning.py funcion with base config and a sig_frac config 
e.g. $python cluster_scanning.py config/s0_0.5_1_MB.yaml config/bootstrap/0_300.yaml
to simmulare analysis in case of signal contamination
4.2	Use cs_prformance_evaluateon.py with plotting function to visualise the process of test statistic evaluation for a givec cluster_scanning run
5. 	Use t_statistic_distribution.py with configs
	... 
to generate the null-hypothesis distribution of the test statistic and show the test statistic and the significance for the case of different levels of signal contaminations