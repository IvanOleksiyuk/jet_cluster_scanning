# This file is the script that is used to produce all the plots from the additional studies requested by the reviewers.

# Start with all the usual imports 
IOTEMP_RESPONCE_DIR=plots/responce


if [ ! -d "$IOTEMP_RESPONCE_DIR" ]; then
    # Directory does not exist, so create it
    mkdir -p "$IOTEMP_RESPONCE_DIR"
    echo "Directory created: $DIR"
else
    echo "Directory already exists: $DIR"
fi

############################################
# Training in the signal region 
############################################

# Run once and get the plots that we need:
python one_cs_run.py sig_reg

# Copy the plots needed for the responce
# TODO

#%% Run the multirun jobs with background only which will perform training clusters and labelling
	cd jobs
	python create_multirun_configs.py 0 4200 100 /home/users/o/oleksiyu/WORK/jet_cluster_scanning/config/multirun/background_only True config/save2responce.yaml
	python start_multirun_jobs.py config/v4/s0_0.5_1_MB_i1.yaml config/path.yaml config/tra_reg/sig_reg.yaml config/binning/CURTAINS.yaml -p config/multirun/i0_15.yaml config/save2responce.yaml
	cd ..

#%% Run the multirun jobs with signal which will perform training clusters and labelling
	cd jobs
	python start_sig_frac_jobs.py config/v4/s0_0.5_1_MB_i1.yaml config/path.yaml config/tra_reg/sig_reg.yaml config/binning/CURTAINS.yaml config/multirun/signal.yaml config/save2responce.yaml
	python start_sig_frac_jobs_extended.py config/v4/s0_0.5_1_MB_i1.yaml config/path.yaml config/tra_reg/sig_reg.yaml config/binning/CURTAINS.yaml config/multirun/signal.yaml config/save2responce.yaml
	cd ..

#%% Run the binning job to get the spectra
	python binning.py -d /home/users/o/oleksiyu/scratch/CS/responce/


############################################
# Training normally but ignoring signal in the training region
# Expectation insignificantly different from the default case
############################################

# Run once and get the plots that we need:
	python one_cs_run.py default_ignore_sig
	mv char/one_run_experiments/k50MB2048_1iret0con0.05W3000_3100_w0.5s1Nboot char/one_run_experiments/k50MB2048_1iret0con0.05W3000_3100_w0.5s1Nboot_ignore_sig

# Copy the plots needed for the responce
# TODO

# Run the multirun jobs with signal which will perform training clusters and labelling
	cd jobs
	python start_sig_frac_jobs.py config/v4/s0_0.5_1_MB_i1.yaml config/path.yaml config/tra_reg/3000_3100.yaml config/binning/CURTAINS.yaml config/multirun/signal.yaml config/ignore_signal.yaml config/save2responce.yaml
	python start_sig_frac_jobs_extended.py config/v4/s0_0.5_1_MB_i1.yaml config/path.yaml config/tra_reg/3000_3100.yaml config/binning/CURTAINS.yaml config/multirun/signal.yaml config/ignore_signal.yaml config/save2responce.yaml
	cd ..

############################################
# Evaluating with fit 
############################################

# Run once and get the plots that we need:
python one_cs_run.py default+fit
mv char/one_run_experiments/k50MB2048_1iret0con0.05W3000_3100_w0.5s1Nboot char/one_run_experiments/k50MB2048_1iret0con0.05W3000_3100_w0.5s1Nboot_fit

# Copy the plots needed for the responce
# TODO
