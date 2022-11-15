from cycler import cycler
import matplotlib
import matplotlib.pyplot as plt
import pickle
import numpy as np
def compare_anomaly_type_histograms(char1, char2, sufix1, sufix2, fig_name=None):
    n=9
    cmap=matplotlib.cm.get_cmap("autumn")
    cc = cycler(color=[cmap(i/n) for i in range(n)])
    plt.rc('axes', prop_cycle=cc)
    char_path=char1+"/res.pickle"
    res=pickle.load(open(char_path, "rb"))
    
    n_sigmas_inclusion_arr=res["n_sigmas_inclusion_arr"+sufix1]
    in_n_sigma_all=res["in_n_sigma_all"+sufix1]
    n_clusters_inclusion_arr=res["n_clusters_inclusion_arr"+sufix1]
    in_n_clusters_all=res["in_n_clusters_all"+sufix1]
    
    print(np.sum(in_n_sigma_all, axis=0))
    #print(res["means"][:10])
    #print(res["sigmas"][:10])
    
    in_n_clusters_all=np.array(in_n_clusters_all)
    in_n_sigma_all=np.array(in_n_sigma_all)
    plt.figure()
    for n_cluster_inclusion in n_clusters_inclusion_arr:
        plt.plot(in_n_clusters_all[:, n_cluster_inclusion-1], label="in "+(str)(n_cluster_inclusion-1)+" clusters", linestyle="--")
    for n_sigmas_inclusion in n_sigmas_inclusion_arr:
        plt.plot(in_n_sigma_all[:, n_sigmas_inclusion], label=(str)(n_sigmas_inclusion)+"sigma inclusion")
    
    cmap=matplotlib.cm.get_cmap("winter")
    char_path=char2+"/res.pickle"
    res=pickle.load(open(char_path, "rb"))
    
    n_sigmas_inclusion_arr=res["n_sigmas_inclusion_arr"+sufix2]
    in_n_sigma_all=res["in_n_sigma_all"+sufix2]
    n_clusters_inclusion_arr=res["n_clusters_inclusion_arr"+sufix2]
    in_n_clusters_all=res["in_n_clusters_all"+sufix2]

    print(np.sum(in_n_sigma_all, axis=0))
    #print(res["means"][:10])
    #print(res["sigmas"][:10])
    
    in_n_clusters_all=np.array(in_n_clusters_all)
    in_n_sigma_all=np.array(in_n_sigma_all)
    i=0
    for n_cluster_inclusion in n_clusters_inclusion_arr:
        plt.plot(in_n_clusters_all[:, n_cluster_inclusion-1], label="in "+(str)(n_cluster_inclusion-1)+" clusters", linestyle="--", color=cmap(i/n))
        i+=1
    for n_sigmas_inclusion in n_sigmas_inclusion_arr:
        plt.plot(in_n_sigma_all[:, n_sigmas_inclusion], label=(str)(n_sigmas_inclusion)+"sigma inclusion", color=cmap(i/n))
        i+=1
    
    ax = plt.gca()
    ax.set_title(char1+sufix1.replace('_', '\_')+"\n"+char2+sufix2.replace('_', '\_')) #
    if fig_name!=None:
        plt.savefig(fig_name)

if __name__ == "__main__":
    # possible_switches=[1, 2]
    switches=[0]
    plt.close("all")
    if 0 in switches:
         dataset="QCDf+Q50M10lf"
         reproc="none"
         compare_anomaly_type_histograms("char/"+dataset+"m100s3c100r"+reproc+"KId0logLds", "char/"+dataset+"m100s3c100r"+reproc+"KId0logLrhn", sufix1="_bg", sufix2="_bg")
         compare_anomaly_type_histograms("char/"+dataset+"m100s3c100r"+reproc+"KId0logLds", "char/"+dataset+"m100s3c100r"+reproc+"KId0logLrh0", sufix1="_bg", sufix2="_bg")
         compare_anomaly_type_histograms("char/"+dataset+"m100s3c100r"+reproc+"KId0logLds", "char/"+dataset+"m100s3c100r"+reproc+"KId0logLrhn", sufix1="_sg", sufix2="_sg")
         compare_anomaly_type_histograms("char/"+dataset+"m100s3c100r"+reproc+"KId0logLds", "char/"+dataset+"m100s3c100r"+reproc+"KId0logLrh0", sufix1="_sg", sufix2="_sg")
         
    if 1 in switches:
         dataset="QCDf+DMlf"
         reproc="none"
         compare_anomaly_type_histograms("char/"+dataset+"m100s3c100r"+reproc+"KId0logLds", "char/"+dataset+"m100s3c100r"+reproc+"KId0logLrhn", sufix1="_bg", sufix2="_bg")
         compare_anomaly_type_histograms("char/"+dataset+"m100s3c100r"+reproc+"KId0logLds", "char/"+dataset+"m100s3c100r"+reproc+"KId0logLrh0", sufix1="_bg", sufix2="_bg")
         compare_anomaly_type_histograms("char/"+dataset+"m100s3c100r"+reproc+"KId0logLds", "char/"+dataset+"m100s3c100r"+reproc+"KId0logLrhn", sufix1="_sg", sufix2="_sg")
         compare_anomaly_type_histograms("char/"+dataset+"m100s3c100r"+reproc+"KId0logLds", "char/"+dataset+"m100s3c100r"+reproc+"KId0logLrh0", sufix1="_sg", sufix2="_sg")

    if 2 in switches:
         dataset="QCDf+DMlf"
         reproc="4rt"
         compare_anomaly_type_histograms("char/"+dataset+"m100s3c100r"+reproc+"KId0logLds", "char/"+dataset+"m100s3c100r"+reproc+"KId0logLrhn", sufix1="_bg", sufix2="_bg")
         compare_anomaly_type_histograms("char/"+dataset+"m100s3c100r"+reproc+"KId0logLds", "char/"+dataset+"m100s3c100r"+reproc+"KId0logLrh0", sufix1="_bg", sufix2="_bg")
         compare_anomaly_type_histograms("char/"+dataset+"m100s3c100r"+reproc+"KId0logLds", "char/"+dataset+"m100s3c100r"+reproc+"KId0logLrhn", sufix1="_sg", sufix2="_sg")
         compare_anomaly_type_histograms("char/"+dataset+"m100s3c100r"+reproc+"KId0logLds", "char/"+dataset+"m100s3c100r"+reproc+"KId0logLrh0", sufix1="_sg", sufix2="_sg")
    
    if 3 in switches:
         dataset="QCDf+DMlf"
         reproc="4rt"
         compare_anomaly_type_histograms("char/"+dataset+"m100s3c100r"+reproc+"KId0logLds", "char/"+dataset+"m100s3c100r"+reproc+"KId0logLrhn", sufix1="_bg", sufix2="_bg")
         compare_anomaly_type_histograms("char/"+dataset+"m100s3c100r"+reproc+"KId0logLds", "char/"+dataset+"m100s3c100r"+reproc+"KId0logLrh0", sufix1="_bg", sufix2="_bg")
         compare_anomaly_type_histograms("char/"+dataset+"m100s3c100r"+reproc+"KId0logLds", "char/"+dataset+"m100s3c100r"+reproc+"KId0logLrhn", sufix1="_sg", sufix2="_sg")
         compare_anomaly_type_histograms("char/"+dataset+"m100s3c100r"+reproc+"KId0logLds", "char/"+dataset+"m100s3c100r"+reproc+"KId0logLrh0", sufix1="_sg", sufix2="_sg")
    
    if 4 in switches:
         dataset="toph5+QCDh5"
         reproc="none"
         compare_anomaly_type_histograms("char/"+dataset+"m100s3c100r"+reproc+"KId0logLds", "char/"+dataset+"m100s3c100r"+reproc+"KId0logLrhn", sufix1="_bg", sufix2="_bg")
         compare_anomaly_type_histograms("char/"+dataset+"m100s3c100r"+reproc+"KId0logLds", "char/"+dataset+"m100s3c100r"+reproc+"KId0logLrh0", sufix1="_bg", sufix2="_bg")
         compare_anomaly_type_histograms("char/"+dataset+"m100s3c100r"+reproc+"KId0logLds", "char/"+dataset+"m100s3c100r"+reproc+"KId0logLrhn", sufix1="_sg", sufix2="_sg")
         compare_anomaly_type_histograms("char/"+dataset+"m100s3c100r"+reproc+"KId0logLds", "char/"+dataset+"m100s3c100r"+reproc+"KId0logLrh0", sufix1="_sg", sufix2="_sg")

    if 5 in switches:
         dataset="QCDh5+toph5"
         reproc="none"
         compare_anomaly_type_histograms("char/"+dataset+"m100s3c100r"+reproc+"KId0logLds", "char/"+dataset+"m100s3c100r"+reproc+"KId0logLrhn", sufix1="_bg", sufix2="_bg")
         compare_anomaly_type_histograms("char/"+dataset+"m100s3c100r"+reproc+"KId0logLds", "char/"+dataset+"m100s3c100r"+reproc+"KId0logLrh0", sufix1="_bg", sufix2="_bg")
         compare_anomaly_type_histograms("char/"+dataset+"m100s3c100r"+reproc+"KId0logLds", "char/"+dataset+"m100s3c100r"+reproc+"KId0logLrhn", sufix1="_sg", sufix2="_sg")
         compare_anomaly_type_histograms("char/"+dataset+"m100s3c100r"+reproc+"KId0logLds", "char/"+dataset+"m100s3c100r"+reproc+"KId0logLrh0", sufix1="_sg", sufix2="_sg")
    
    if 1.1 in switches:
         dataset="QCDf+Q50M10lf"
         compare_anomaly_type_histograms("char/"+dataset+"m100s3c100rnoneKId0logLrh0", "char/QCDf+Q50M10lfm100s3c100rnoneKId0logLdis", sufix1="_bg", sufix2="_bg")
         compare_anomaly_type_histograms("char/"+dataset+"m100s3c100rnoneKId0logLrh0", "char/QCDf+Q50M10lfm100s3c100rnoneKId0logLdis", sufix1="_sg", sufix2="_sg")
         