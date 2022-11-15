from sklearn.cluster import KMeans
import numpy as np
import random 
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import pickle
import reprocessing
from sklearn.metrics import roc_auc_score, roc_curve
from functools import partial
from plot_2d_array import plot_2d_array
from dataset_path_and_pref import dataset_path_and_pref, prepare_data

#prepare pyplot
#plt.close("all")
plt.rcParams.update({'font.size': 18})

def metric_auc(labels, scores):
    return roc_auc_score(labels, scores)

def metric_eps_B_inv(labels, scores, eps_s=0.3):
    fpr , tpr , thresholds = roc_curve(labels, scores)
    return 1/fpr[find_nearest_idx(tpr, eps_s)]

def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def comb_loss(losses, n=1):
    return np.mean(losses[:, :n], 1)

def eval_metric(bg_scores, sg_scores, metric):
    labels=np.concatenate((np.zeros(len(bg_scores)), np.ones(len(sg_scores))))
    return metric(labels, np.append(bg_scores, sg_scores))

def contamination_ROC_compare(MODE=1,
                              main_name="QCD+topm{:}s3c100rnoneKId0MinD", #main_name="+topm100s3c100rnoneKId0logLrhn"
                              k_arr=[1, 2, 4, 8, 16, 32, 64, 128, 256], 
                              label=""):
    if MODE==1:
        metric_func=metric_auc
        metric_name="AUC"
        metric_label="AUC"
    
    if MODE==2:
        metric_func=partial(metric_eps_B_inv, eps_s=0.3)
        metric_name="eps_B_S03"
        metric_label="1/\epsilon_B(\epsilon_S=0.3)"
    
    if MODE==3:
        metric_func=partial(metric_eps_B_inv, eps_s=0.1)
        metric_name="eps_B_S01"
        metric_label="1/\epsilon_B(\epsilon_S=0.1)"
        
    if MODE==4:
        metric_func=partial(metric_eps_B_inv, eps_s=0.2)
        metric_name="eps_B_S02"
        metric_label="1/\epsilon_B(\epsilon_S=0.2)"
        
    if MODE==5:
        metric_func=partial(metric_eps_B_inv, eps_s=0.5)
        metric_name="eps_B_S05"
        metric_label="1/\epsilon_B(\epsilon_S=0.5)"
    
    metric=[]
    for k in k_arr:
        char_path="char/"+main_name.format(k)+"/res.pickle"
        res=pickle.load(open(char_path, "rb"))
        if MODE==1:
            metric.append(res["AUC"])
        if MODE==2:
            eps_s=0.3
            metric.append(1/res["fpr"][find_nearest_idx(res["tps"], eps_s)])
        if MODE==3:
            eps_s=0.1
            metric.append(1/res["fpr"][find_nearest_idx(res["tps"], eps_s)])
        if MODE==4:
            eps_s=0.2
            metric.append(1/res["fpr"][find_nearest_idx(res["tps"], eps_s)])
        if MODE==5:
            eps_s=0.5
            metric.append(1/res["fpr"][find_nearest_idx(res["tps"], eps_s)])
    plt.plot(np.log2(k_arr), metric, label=label)
    
    
if __name__ == "__main__":
    switches=[1]
    if 0 in switches:
        MODE=2
        plt.figure(figsize=(12, 8))
        contamination_ROC_compare(MODE=MODE, main_name="2d10s+2d1s6mm{:}s0c100rnoneKId0MinD", k_arr=[1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32, 64, 128, 256], label="MinD")
        contamination_ROC_compare(MODE=MODE, main_name="2d10s+2d1s6mm{:}s0c100rnoneKId0KNC5", k_arr=[1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32, 64, 128, 256], label="KNC5")
        contamination_ROC_compare(MODE=MODE, main_name="2d10s+2d1s6mm{:}s0c100rnoneKId0logLrhn", k_arr=[1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32, 64, 128, 256], label="logLrhn")
        contamination_ROC_compare(MODE=MODE, main_name="2d10s+2d1s6m1000_+2d1s6mm{:}s0c100rnoneKId0MinD", k_arr=[1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32, 64, 128, 256], label="MinD 1%")
        contamination_ROC_compare(MODE=MODE, main_name="2d10s+2d1s6m1000_+2d1s6mm{:}s0c100rnoneKId0KNC5", k_arr=[1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32, 64, 128, 256], label="KNC5 1%")
        contamination_ROC_compare(MODE=MODE, main_name="2d10s+2d1s6m1000_+2d1s6mm{:}s0c100rnoneKId0logLrhn", k_arr=[1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32, 64, 128, 256], label="logLrhn 1%")
        #plt.axhline(y=58)
        plt.legend()
        plt.yscale("log")
        plt.grid()
        
    if 1 in switches:
        contamination_ROC_compare(MODE=1, main_name="top+QCDm{:}s3c10rnoneKId0MinD")
        contamination_ROC_compare(MODE=1, main_name="top+QCDm{:}s3c10rnoneKId0KNC5", k_arr=[8, 16, 32, 64, 100, 128, 256])
        contamination_ROC_compare(MODE=1, main_name="top+QCDm{:}s3c10rnoneKId0logLrhn", k_arr=[1, 2, 4, 8, 16, 32, 64, 100])
        plt.legend()
        plt.grid()
        #plt.savefig("plots/ROC_cont/"+main_name+".png", bbox_inches="tight")
        #contamination_ROC_compare(MODE=1, main_name="QCD+topm{:}s3c10rnoneKId0KNC5")
        
    if 2 in switches:
        MODE=1
        plt.figure(figsize=(12, 8))
        contamination_ROC_compare(MODE=MODE, main_name="2d10s+2d1s4mm{:}s0c100rnoneKId0MinD", k_arr=[1, 2, 4, 8, 16, 32, 64, 128, 256], label="MinD")
        contamination_ROC_compare(MODE=MODE, main_name="2d10s+2d1s4mm{:}s0c100rnoneKId0KNC5", k_arr=[1, 2, 4, 8, 16, 32, 64, 128, 256], label="KNC5")
        contamination_ROC_compare(MODE=MODE, main_name="2d10s+2d1s4mm{:}s0c100rnoneKId0logLrhn", k_arr=[1, 2, 4, 8, 16, 32, 64, 128, 256], label="logLrhn")
        #plt.axhline(y=58)
        plt.legend()
        #plt.yscale("log")
        plt.grid()
