import sys
import os
import shutil

# copy a job template to scripted folder

exp_path="/home/users/o/oleksiyu/scratch/CS/v4/k50MB2048_1iret0con0W3000_3100_w0.5s1Nboot/"

b_batch_size= 30
for b_batch in range(0, 100):
    jname = f"b{b_batch}x{b_batch_size}_eval"
    command_str = f"python3 batch_eval_tstat.py {exp_path} binnedW100s16ei30004600/ config/cs_eval/maxdev3_msders_3fit.yaml maxdev3_msders_3fit {b_batch*b_batch_size} {(b_batch+1)*b_batch_size} 0 1 0 30"
    path = "scripted/" + f"job_{jname}.sh"
    shutil.copy2("job_CPU_1h.sh", path)
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