import numpy as np
from utilities import latex_table, get_metric_from_char

def create_table_of_kmeans_cluster_properties(datasets_list, preproc_list, preproc_write, metric_list, labels_list, savefile='tables/dim_table.tex', dataset_labels=None):
    char="{:}m100s3c100r{:}KId0logLds" 
    
    arr=np.full((len(metric_list)+1, len(datasets_list)+1), "x", dtype='<U100')
    
    arr[0][0]=""
    
    for j in range(len(datasets_list)):
        if dataset_labels==None:
            arr[0][j+1]=datasets_list[j]+""+preproc_write[j]
        else:
            arr[0][j+1]=dataset_labels[j]
      
    for i in range(len(metric_list)):
        arr[i+1][0]=labels_list[i]
        for j in range(len(datasets_list)):
            arr[i+1][j+1]=get_metric_from_char("char/"+char.format(datasets_list[j], preproc_list[j]), metric_list[i], form="{:.3g}")
    
    arr=arr.T
    print(arr)
    
    
    with open(savefile, 'w') as f:
        
        
        caption="Several properties of the set of 100 clusters found by k-means for different datasets."
        f.write(latex_table(arr, caption=caption))

if __name__ == "__main__":
    datasets_list=["QCD+top", "top+QCD", "QCDl+DMl", "QCDl+DMl", "QCDl+lptQ50M10", "QCDl+lptQ50M10"]
    preproc_list=["none", "none", "4rt", "none", "4rt", "none"]
    preproc_write=["", "", " 4rt",  "", " 4rt",  ""]
    metric_list=["mu_min", "mu_median", "mu_max", "sig/mu_max", "d_med"]
    labels_list=["$min(\\rho_i)$", "$med(\\rho_i)$", "$max(\\rho_i)$", "$max(\\sigma_i/\\rho_i)$", "$med(d_i(\\rho_i))$"]
    create_table_of_kmeans_cluster_properties(datasets_list, preproc_list, preproc_write, metric_list, labels_list, savefile='tables/dim_table.tex')