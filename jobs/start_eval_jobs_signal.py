import sys
import os
import shutil

# copy a job template to scripted folder

exp_path="/home/users/o/oleksiyu/scratch/CS/v4/k50MB2048_1iret0con0W3000_3100_w0.5s1Nboot/"
["k50MB2048_1iret0con0.005W3000_3100_w0.5s1Nboot", 
"k50MB2048_1iret0con0.01W3000_3100_w0.5s1Nboot",
"k50MB2048_1iret0con0.008W3000_3100_w0.5s1Nboot",  
"k50MB2048_1iret0con0.021W3000_3100_w0.5s1Nboot",
"k50MB2048_1iret0con0.009W3000_3100_w0.5s1Nboot", 
"k50MB2048_1iret0con0.022W3000_3100_w0.5s1Nboot",
"k50MB2048_1iret0con0.011W3000_3100_w0.5s1Nboot",  
"k50MB2048_1iret0con0.023W3000_3100_w0.5s1Nboot",
"k50MB2048_1iret0con0.012W3000_3100_w0.5s1Nboot",  
"k50MB2048_1iret0con0.024W3000_3100_w0.5s1Nboot",
"k50MB2048_1iret0con0.013W3000_3100_w0.5s1Nboot",  
"k50MB2048_1iret0con0.025W3000_3100_w0.5s1Nboot",
"k50MB2048_1iret0con0.014W3000_3100_w0.5s1Nboot",  
"k50MB2048_1iret0con0.02W3000_3100_w0.5s1Nboot",
"k50MB2048_1iret0con0.015W3000_3100_w0.5s1Nboot",  
"k50MB2048_1iret0con0.03W3000_3100_w0.5s1Nboot",
"k50MB2048_1iret0con0.016W3000_3100_w0.5s1Nboot",  
"k50MB2048_1iret0con0.05W3000_3100_w0.5s1Nboot",
"k50MB2048_1iret0con0.017W3000_3100_w0.5s1Nboot",  
"k50MB2048_1iret0con0.1W3000_3100_w0.5s1Nboot",
"k50MB2048_1iret0con0.018W3000_3100_w0.5s1Nboot",  
"k50MB2048_1iret0con0.019W3000_3100_w0.5s1Nboot"]


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