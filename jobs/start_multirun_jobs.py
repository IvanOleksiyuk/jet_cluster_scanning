import sys
import os
import shutil

# list of config files:
config_files = [
    "config/multirun/b0_300i0_5.yaml",
    "config/multirun/b300_600i0_5.yaml",
    "config/multirun/b600_900i0_5.yaml",
    "config/multirun/b900_1200i0_5.yaml",
    "config/multirun/b1200_1500i0_5.yaml",
    "config/multirun/b1500_1800i0_5.yaml",
    "config/multirun/b1800_2100i0_5.yaml",
]
jnames = ["0_300", "300_600", "600_900", "900_1200", "1200_1500", "1500_1800", "1800_2100"]

# copy a job template to scripted folder
for jname, config_file in zip(jnames, config_files):
    path = "scripted/" + f"job_{jname}.sh"
    shutil.copy2("job_CPU_6h.sh", path)
    # Append a python run command to the job template
    with open(path, "a") as f:
        f.write(
            "python3 cluster_scanning.py " + " ".join(sys.argv[1:]) + " " + config_file
        )
    # Submit the job to the cluster
    bashCommand = "sbatch " + path
    os.system(bashCommand)
    print("submitted job: ", jname)
