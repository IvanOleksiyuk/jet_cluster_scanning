# jet_cluster_scanning

These are my experiments concerning the anomaly detection using k-means to separate space into bins for bump hunting.
Ivan Oleksiyuk
ivan.oleksiyuk@gmail.com

I am a beginer programmer so any constructive criticism is very wellcome.

Short algorythm on how to use the prject:
0. 	extrat jet constituents (e.g. LHCO) (not realised in this project (DIY if needed I guess)) and masses of the corresponding events in the other file 
1.	Use ... to turn (e.g. LHCO) jet constituent data into images
2.	Use ... to sort images in the datasets with respect to the events invariant mass (e.g. jj_mass)
3.	Use cluster_scanning.py funcion with configs 
	... 
to bootstrap resample the background only sample and perform ananlysis on each resampling in order to get a background hypothesis distribution of the 
4.1	Use cluster_scanning.py funcion with configs 
	... 
to simmulare analysis in case of signal contamination
4.2	Use cs_prformance_evaluateon.py with plotting function to visualise the process of test statistic evaluation for a givec cluster_scanning run
5. 	Use t_statistic_distribution.py with configs
	... 
to generate the null-hypothesis distribution of the test statistic and show the test statistic and the significance for the case of different levels of signal contaminations