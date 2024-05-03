import sys
import os
import shutil
from itertools import groupby

# list of config files:
config_files = [
    "config/sig_frac/0.05.yaml",
    "config/sig_frac/0.03.yaml",
    "config/sig_frac/0.02.yaml",
    "config/sig_frac/0.015.yaml",
    "config/sig_frac/0.01.yaml",
]
jnames = ["0.05", "0.03", "0.02", "0.015", "0.01"]

# copy a job template to scripted folder
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
