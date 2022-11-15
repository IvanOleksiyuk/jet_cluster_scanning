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

def k_means_eval(
    fast_train="d",
    REVERSE=False,
    DATASET=1,
    SIGMA=3,
    CROP=10000,
    preproc=None,
    Id=0,
    MODE=1,
    TOP_DOWN=True,
    CUSTOM_MODEL_NAME=False, 
    ensemble=None, 
    k_arr=[1, 2, 4, 8, 16, 32, 64, 128, 256],
    log_k=True): #"models/Rm{:}s3c10KId2.pickle"
    
    random.seed(a=0, version=2)
    
    pref, pref2, tra_data_path, con_data_path, bg_val_data_path, sg_val_data_path = dataset_path_and_pref(DATASET, REVERSE)
    
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
        metric_name="eps_B_S01"
        metric_label="1/\epsilon_B(\epsilon_S=0.2)"
    
    X_tr=prepare_data(tra_data_path, preproc=preproc, SIGMA=SIGMA, crop=CROP)
    X_bg_val=prepare_data(bg_val_data_path, preproc=preproc, SIGMA=SIGMA)
    X_sg_val=prepare_data(sg_val_data_path, preproc=preproc, SIGMA=SIGMA)
    
    mean_dist=[np.inf]
    
    for i, k in enumerate(k_arr):
        if CUSTOM_MODEL_NAME:
            MODEl_path=CUSTOM_MODEL_NAME.format(k)
        else:
            MODEl_path="models/{:}m{:}s{:}c{:}r{:}KI{:}{:}.pickle".format(pref, k, SIGMA, CROP//1000, reprocessing.reproc_names(preproc), fast_train, Id)
        print(MODEl_path)
        kmeans=pickle.load(open(MODEl_path, "rb"))
        tr_losses = kmeans.transform(X_bg_val)
        tr_min_dists = np.min(tr_losses, axis=1)
        print(len(tr_min_dists))
        mean_dist.append(np.mean(tr_min_dists))
         
    mean_dist=np.array(mean_dist)
    #plt.plot(np.log2(k_arr), (1/mean_dist[1:]-1/mean_dist[:-1])/(1/mean_dist[:-1]))
    if log_k:
        plt.xlabel("$log_2(k)$")
    else:
        plt.xlabel("$k$")
    
    if log_k:
        plt.ylabel("$(<d_k>-<d_{k/2}>)/<d_{k/2}>$")
        plt.plot(np.log2(k_arr[1:]), (mean_dist[2:]-mean_dist[1:-1])/(mean_dist[1:-1]), label=pref+reprocessing.reproc_names(preproc))
    else:
        plt.ylabel("$(<d_k>-<d_{k-1}>)/<d_{k-1}>$")
        plt.plot(k_arr[1:], (mean_dist[2:]-mean_dist[1:-1])/(mean_dist[1:-1]), label=pref+reprocessing.reproc_names(preproc))

    
if __name__ == "__main__":
    switches=[1]
    if 1 in switches:
        plt.figure(figsize=(5, 4))
        k_means_eval(MODE=1, DATASET=1)
        k_means_eval(MODE=1, DATASET=1, REVERSE=True)
        k_means_eval(MODE=1, DATASET=2, preproc=reprocessing.reproc_4rt)
        k_means_eval(MODE=1, DATASET=3)
        k_means_eval(MODE=1, DATASET=4)
        plt.legend()
        plt.grid()
    if 2 in switches:
        plt.figure(figsize=(5, 4))
        plt.axhline(y=-0.05, linestyle="--", c="grey")
        a=np.linspace(2, 25, 1000)
        plt.plot(a, -1/a, c="grey")
        k_means_eval(MODE=1, DATASET=5, SIGMA=0, CROP=100000, log_k=False, k_arr=[i+1 for i in range(25)])
        k_means_eval(MODE=1, DATASET=5.1, SIGMA=0, CROP=100000, log_k=False, k_arr=[i+1 for i in range(25)])
        k_means_eval(MODE=1, DATASET=5.2, SIGMA=0, CROP=100000, log_k=False, k_arr=[i+1 for i in range(25)])
        k_means_eval(MODE=1, DATASET=5.3, SIGMA=0, CROP=100000, log_k=False, k_arr=[i+1 for i in range(10)])
        plt.xlim((0, 20))
        plt.legend()
        plt.grid()
    if 3 in switches:
        plt.figure(figsize=(5, 4))
        plt.axhline(y=-0.1, linestyle="--", c="grey")
        k_means_eval(MODE=1, DATASET=5, SIGMA=0, CROP=100000, k_arr=[1, 2, 4, 8, 16, 32, 64, 128, 256])
        k_means_eval(MODE=1, DATASET=5.1, SIGMA=0, CROP=100000, k_arr=[1, 2, 4, 8, 16])
        k_means_eval(MODE=1, DATASET=5.2, SIGMA=0, CROP=100000, k_arr=[1, 2, 4, 8, 16])
        k_means_eval(MODE=1, DATASET=5.3, SIGMA=0, CROP=100000, k_arr=[1, 2, 4, 8, 16, 32, 64])
        plt.legend()
        plt.grid()
        