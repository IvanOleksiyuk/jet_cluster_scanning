import sys
import os
import shutil
from itertools import groupby

# list of config files:
multirun_config_dir = "../config/multirun/background_only"
print("Looking for config files in: " + multirun_config_dir)
config_files = []  # List to store absolute paths of files
# Walk through directory
for root, dirs, files in os.walk(multirun_config_dir):
    for file in files:
        # Construct absolute path
        file_path = os.path.join(root, file)
        abs_path = os.path.abspath(file_path)
        config_files.append(abs_path)
        print("Found config file: " + abs_path)
        
jnames = [os.path.basename(path).split("_")[0] for path in config_files]

if "-p" in sys.argv[1:]:
    listt = sys.argv[1:]
    switch = listt.index("-p")
    list_suf = listt[:switch]
    list_postfix = listt[switch + 1 :]
else:
    list_suf = sys.argv[1:]
    list_postfix = [""]

# copy a job template to scripted folder
for jname, config_file in zip(jnames, config_files):
    path = "scripted/" + f"job_{jname}.sh"
    shutil.copy2("job_CPU_6h.sh", path)
    # Append a python run command to the job template
    str_for_job = (
        "python3 cluster_scanning.py "
        + " ".join(list_suf)
        + " "
        + config_file
        + " "
        + " ".join(list_postfix)
    )
    with open(path, "a") as f:
        f.write(str_for_job)
    # Submit the job to the cluster
    bashCommand = "sbatch " + path
    os.system(bashCommand)
    print("submitted job: ", str_for_job)
