#!/bin/bash
#SBATCH --partition=shared-cpu

### Job name
#SBATCH --job-name=job3
 
### Output path for stdout and stderr
### %J is the job ID, %I is the array ID
#SBATCH --output=jobs/outputs/output_job3_%J.txt
 
### Request the time you need for execution. The full format is D-HH:MM:SS
### You must at least specify minutes OR days and hours and may add or
### leave out any other parameters
#SBATCH --time=360
 
### Request the amount of memory you need for your job. 
### You can specify this in either MB (1024M) or GB (4G).
#SBATCH --mem-per-cpu=32G
 
### Load modules
module load GCC/10.2.0 CUDA/11.1.1 OpenMPI/4.0.5
module load scikit-learn/0.23.2

### Execute your application
cd ..
python3 cluster_scanning.py "config/bootstrap_600_900_0.5_1.yaml"
