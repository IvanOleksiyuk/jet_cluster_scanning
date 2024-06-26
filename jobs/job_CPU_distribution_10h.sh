#!/bin/bash
### Rather simple job script for binning pf  all files in char/ and export them to export/ if needed
#SBATCH --partition=shared-cpu

### Job name
#SBATCH --job-name=DISTRIBUTION
 
### Output path for stdout and stderr
### %J is the job ID, %I is the array ID
#SBATCH --output=outputs/output_BINNING_%J.txt
 
### Request the time you need for execution. The full format is D-HH:MM:SS
### You must at least specify minutes OR days and hours and may add or
### leave out any other parameters
#SBATCH --time=600
 
### Request the amount of memory you need for your job. 
### You can specify this in either MB (1024M) or GB (4G).
#SBATCH --mem-per-cpu=32G
#SBATCH -c 1
 
### Load modules

module load GCC/10.2.0 CUDA/11.1.1 OpenMPI/4.0.5
module load Python/3.8.6
module load matplotlib/3.3.3
module load scikit-learn/0.23.2

### Go into main directory
cd ..

### Execute your application here
python t_statistic_distribution.py config/distribution/responce/prep05_1_maxdev3_msders+3fitCURTAINS_15mean.yaml config/distribution/v4/bootstrap_sig_contam_old.yaml test/config/plot_path2.yaml
#sh export_bres.sh 
### config/binning/CURTAINS.yaml