import numpy as np
from utilities import latex_table, get_metric_from_char


char_list=["{:}m100s3c100r{:}KId0MinD", 
           "{:}m100s3c100r{:}KId0KNC5",  
           "{:}m100s3c100r{:}KId0logLr" ,
           "{:}m100s3c100r{:}KId0logLrh", 
           "{:}m100s3c100r{:}KId0logLrhn", 
           "{:}m100s3c100r{:}KId0logLmc",
           "{:}m100s3c100r{:}KId0logLmd"] #
labels_list=["MinD", "KNC5", "logLr", "logLrh", "logLrhn", "logLmc", "logLmd"]
datasets_list=["QCD+top", "top+QCD", "QCDl+DMl", "bg+Q200M100", "bg+Q50M10"]
preproc=["none", "none", "4rt", "none", "none"]
preproc_write=["", "", " 4rt", "", ""]
metric="AUC"

arr=np.full((len(char_list)+1, len(datasets_list)+1), "x", dtype='<U100')

arr[0][0]=""

for j in range(len(datasets_list)):
    arr[0][j+1]=datasets_list[j]+""+preproc_write[j]
  
for i in range(len(char_list)):
    arr[i+1][0]=labels_list[i]
    for j in range(len(datasets_list)):
        arr[i+1][j+1]=get_metric_from_char("char/"+char_list[i].format(datasets_list[j], preproc[j]), metric, form="{:.3f}")

print(arr)


with open('C:/ML_Dark_matter/knc/result_table.tex', 'w') as f:
    caption="AUC of taging with no contamination"
    f.write(latex_table(arr, caption=caption))
    

char_list=["{:}m100s3c100r{:}KId0MinD", 
           "{:}m100s3c100r{:}KId0KNC5",  
           "{:}m100s3c100r{:}KId0logLr" ,
           "{:}m100s3c100r{:}KId0logLrh", 
           "{:}m100s3c100r{:}KId0logLrhn", 
           "{:}m100s3c100r{:}KId0logLmc"] #
labels_list=["MinD", "KNC5", "logLr", "logLrh", "logLrhn", "logLmc"]
datasets_list=["QCD+top1000_+top", "top+QCD1000_+QCD", "QCDl+DMl1000_+DMl", "bg+Q200M1001000_+Q200M100", "bg+Q50M101000_+Q50M10"]
preproc=["none", "none", "4rt", "none", "none"]
datasets_names=["QCD+top", "top+QCD", "QCDl+DMl", "bg+Q200M100", "bg+Q50M10"]
metric="AUC"

arr=np.full((len(char_list)+1, len(datasets_list)+1), "x", dtype='<U100')

arr[0][0]=""

for j in range(len(datasets_list)):
    arr[0][j+1]=datasets_names[j]
  
for i in range(len(char_list)):
    arr[i+1][0]=labels_list[i]
    for j in range(len(datasets_list)):
        arr[i+1][j+1]=get_metric_from_char("char/"+char_list[i].format(datasets_list[j], preproc[j]), metric, form="{:.3f}")

print(arr)
    
with open('C:/ML_Dark_matter/knc/result_table_cont.tex', 'w') as f:
    caption="AUC of taging with contamination of 1\% of anomalies during triaining"
    f.write(latex_table(arr, caption=caption))