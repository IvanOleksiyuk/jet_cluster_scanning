import sys
import os
import shutil
from itertools import groupby

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
jnames = [
    "0_300",
    "300_600",
    "600_900",
    "900_1200",
    "1200_1500",
    "1500_1800",
    "1800_2100",
]

if "-p" in sys.argv[1:]:
    listt = sys.argv[1:]
    listt2 = [list(group) for k, group in groupby(listt, lambda x: x == "-p") if not k]
    print(listt2)
    list_suf = listt2[0]
    list_postfix = listt2[1]
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
