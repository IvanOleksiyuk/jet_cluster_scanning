import sys
import os
import shutil

# list of config files:
config_files = [
    "config/sig_frac/0.1.yaml",
    "config/sig_frac/0.05.yaml",
    "config/sig_frac/0.01.yaml",
]
jnames = ["0.1", "0.05", "0.01"]

# copy a job template to scripted folder
for jname, config_file in zip(jnames, config_files):
    path = "scripted/" + f"job_{jname}.sh"
    shutil.copy2("job_CPU_6h.sh", path)
    # Append a python run command to the job template
    with open(path, "a") as f:
        f.write("python3 cluster_scanning.py " + sys.argv[1:] + " " + config_file)
    # Submit the job to the cluster
    bashCommand = "sbatch " + path
    os.system(bashCommand)
    print("submitted job: ", jname)
