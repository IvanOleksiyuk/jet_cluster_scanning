import sys
import os
import shutil

# copy a job template to scripted folder
jnames = ["1", "05", "03", "025", "02", "01", "009", "008", "005", "011", "012", "013", "014", "015", "016", "017", "018", "019", "021", "022", "023", "024"]

b_batch_size= 30
for jname in jnames:
    exp_path = f"/home/users/o/oleksiyu/scratch/CS/v4/k50MB2048_1iret0con0.{jname}W3000_3100_w0.5s1Nboot/"
    command_str = f"python3 batch_eval_tstat.py {exp_path} binnedW100s16ei30004600/ config/cs_eval/maxdev3_msders_3fit.yaml maxdev3_msders_3fit 0 100 0 100 0 30"
    path = "scripted/" + f"job_{jname}.sh"
    shutil.copy2("job_CPU_4h.sh", path)
    # Append a python run command to the job template
    with open(path, "a") as f:
        f.write(
            command_str
        )
    # Submit the job to the cluster
    bashCommand = "sbatch " + path
    os.system(bashCommand)
    print("submitted job: ", jname)

# python3 batch_eval_tstat.py /home/users/o/oleksiyu/scratch/CS/v4/k50MB2048_1iret0con0W3000_3100_w0.5s1Nboot/ binnedW100s16ei30004600/ config/cs_eval/maxdev5.yaml maxdev5 0 10 0 1 0 30