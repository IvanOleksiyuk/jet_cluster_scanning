import sys
import os
import shutil

# list of config files:
config_files = [
    # "config/multirun/b2100_2400i0.yaml",
    # "config/multirun/b2400_2700i0.yaml",
    "config/multirun/b2700_3000i0.yaml",
    # "config/multirun/b3000_3300i0.yaml",
    # "config/multirun/b3300_3600i0.yaml",
    # "config/multirun/b3600_3900i0.yaml",
    # "config/multirun/b3900_4200i0.yaml",
]
jnames = [
    # "2100_2400",
    # "2400_2700",
    "2700_3000",
    # "3000_3300",
    # "3300_3600",
    # "3600_3900",
    # "3900_4200",
]

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
    shutil.copy2("job_CPU_6h_8c.sh", path)
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
