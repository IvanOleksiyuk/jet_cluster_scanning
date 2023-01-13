import sys
import os
import shutil

# list of config files:
config_files = [
    "config/restart/0_300.yaml",
    "config/restart/300_600.yaml",
    "config/restart/600_900.yaml",
    "config/restart/900_1200.yaml",
    "config/restart/1200_1500.yaml",
]
jnames = ["0_300", "300_600", "600_900", "900_1200", "1200_1500"]

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
