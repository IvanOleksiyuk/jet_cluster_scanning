import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from utilities import get_metric_from_res
import os
mpl.rcParams.update(mpl.rcParamsDefault)

plt.close("all")

list_k_clusters=[1, 2, 5, 10, 20, 50, 100, 200]
list_sigma=[0, 0.5, 1, 2]
list_prepr_names=["none", "sqrt", "4rt"]
list_score_names=["MinD", "KNC5", "logLds", "logLrhn"]
list_combination=["mean", "max", "min"]
postpr=""

#np.array((len(list_k_clusters), len(list_sigma), len(list_prepr), len(list_score), len(list_combination)))
metric_name="AUC" #"eS@ieB100"
plot_min_pop=False 
ylims=None
os.makedirs("plots/param_scanQCDtop/"+metric_name, exist_ok=True)

for sigma in list_sigma:
    for prep in list_prepr_names:
        if plot_min_pop:
            fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [5, 1]}, figsize=(8, 8))
            plt.sca(ax0)
        else:
            plt.figure(figsize=(8, 8))
        metric=np.zeros((len(list_k_clusters)))
        min_pop=np.zeros((len(list_k_clusters)))
        
        for scr in list_score_names:       
            for i in range(len(list_k_clusters)):
                k=list_k_clusters[i]
                with open("char/MBQCDh5+toph52000_+toph5m{:}s{:}c100r{:}{:}KId0{:}/res.pickle".format(k, sigma, prep, postpr, scr), "rb") as f:
                    res=pickle.load(f)
                metric[i]=get_metric_from_res(res, metric_name)
        
            plt.plot(list_k_clusters, metric, ".-", alpha=0.5, label=scr+" one jet")

        
        plt.xscale('log')
        #plt.yscale('log')
        plt.ylim(ylims)
        plt.grid(b=True, which='major', color='grey')
        plt.grid(b=True, which='minor', color='lightgrey')
        plt.ylabel(metric_name)
        
        plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
        plt.title(metric_name+" sigma {:} prep {:} minibatch".format(sigma, prep))
        
        if plot_min_pop:
            plt.sca(ax1) 
            plt.plot(list_k_clusters, min_pop)
            plt.grid(b=True, which='major', color='grey')
            plt.grid(b=True, which='minor', color='lightgrey')
            plt.yscale('log')
            plt.axhline(2000, color="black")
        plt.xlabel("k")
        
        plt.savefig("plots/param_scanQCDtop/{:}/{:}sig{:}prep{:}minibatch.png".format(metric_name, metric_name, sigma, prep), bbox_inches="tight")