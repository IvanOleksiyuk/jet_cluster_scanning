#!/usr/local_rwth/bin/zsh

#SBATCH --account rwth0934
### Job name
#SBATCH --job-name=job3
 
### Output path for stdout and stderr
### %J is the job ID, %I is the array ID
#SBATCH --output=output_job3_%J.txt
 
### Request the time you need for execution. The full format is D-HH:MM:SS
### You must at least specify minutes OR days and hours and may add or
### leave out any other parameters
#SBATCH --time=700
 
### Request the amount of memory you need for your job. 
### You can specify this in either MB (1024M) or GB (4G).
#SBATCH --mem-per-cpu=32G
 
### Load modules
module load python
 
### Execute your application
python3 LHCO_sliding_clusters_bootstrapping1.py
