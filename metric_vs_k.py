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

def sum_n_mins(losses, knc):
    losses_cop=np.copy(losses)
    losses_cop.sort(1)
    return np.mean(losses_cop[:, :knc], 1)

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
    SCORE_TYPE="MinD",
    knc=1): #"models/Rm{:}s3c10KId2.pickle"
    
    random.seed(a=0, version=2)
    
    pref, pref2, tra_data_path, con_data_path, bg_val_data_path, sg_val_data_path = dataset_path_and_pref(DATASET, REVERSE)
    
    if MODE==1:
        metric_func=metric_auc
        metric_name="AUC"
        metric_label="AUC"
    
    if MODE==2:
        metric_func=partial(metric_eps_B_inv, eps_s=0.3)
        metric_name="eps_B_S03"
        metric_label="$1/\epsilon_B(\epsilon_S=0.3)$"
    
    if MODE==3:
        metric_func=partial(metric_eps_B_inv, eps_s=0.1)
        metric_name="eps_B_S01"
        metric_label="$1/\epsilon_B(\epsilon_S=0.1)$"
        
    if MODE==4:
        metric_func=partial(metric_eps_B_inv, eps_s=0.2)
        metric_name="eps_B_S01"
        metric_label="$1/\epsilon_B(\epsilon_S=0.2)$"
    
    X_tr=prepare_data(tra_data_path, preproc=preproc, SIGMA=SIGMA, crop=CROP)
    X_bg_val=prepare_data(bg_val_data_path, preproc=preproc, SIGMA=SIGMA)
    X_sg_val=prepare_data(sg_val_data_path, preproc=preproc, SIGMA=SIGMA)
    
    metric=[]
    
    for i, k in enumerate(k_arr):
        if CUSTOM_MODEL_NAME:
            MODEl_path=CUSTOM_MODEL_NAME.format(k)
        else:
            MODEl_path="models/{:}m{:}s{:}c{:}r{:}KI{:}{:}.pickle".format(pref, k, SIGMA, CROP//1000, reprocessing.reproc_names(preproc), fast_train, Id)
            
        print(MODEl_path)
        kmeans=pickle.load(open(MODEl_path, "rb"))
        
        if SCORE_TYPE=="MinD": #minimal distance
            postf="MinD"
            tr_losses = kmeans.transform(X_tr)
            bg_losses = kmeans.transform(X_bg_val)
            sg_losses = kmeans.transform(X_sg_val)
            bg_scores = np.min(bg_losses, 1)
            sg_scores = np.min(sg_losses, 1)
            tr_scores = np.min(tr_losses, 1)
    
        if SCORE_TYPE=="KNC":
            postf="KNC"+str(knc)
            tr_losses = kmeans.transform(X_tr)
            bg_losses = kmeans.transform(X_bg_val)
            sg_losses = kmeans.transform(X_sg_val)
            bg_scores = sum_n_mins(bg_losses, knc)
            sg_scores = sum_n_mins(sg_losses, knc)
            tr_scores = sum_n_mins(tr_losses, knc)
            
        elif SCORE_TYPE=="Lrhn":
            postf="Lrhn"
            tr_scores, bg_scores, sg_scores, tr_losses, bg_losses, sg_losses = likelyhood_estimation(kmeans, crop, k, X_tr, X_bg_val, X_sg_val, half_gaussian_norm, log_likelyhood=False)
            

        metric.append(eval_metric(bg_scores, sg_scores, metric_func))
 
         
    metric=np.array(metric)
    #plt.plot(np.log2(k_arr), (1/mean_dist[1:]-1/mean_dist[:-1])/(1/mean_dist[:-1]))
    plt.plot(np.log2(k_arr),  metric)
    plt.xlabel("$log_2(k)$")
    plt.ylabel(metric_label)
    

    
if __name__ == "__main__":
    switches=[1]
    if 1 in switches:
        plt.figure(figsize=(10, 8))
        k_means_eval(MODE=1, DATASET=1)
        #k_means_eval(MODE=3, DATASET=1, REVERSE=True)
        k_means_eval(MODE=1, DATASET=2, preproc=reprocessing.reproc_4rt)
        k_means_eval(MODE=1, DATASET=3)
        k_means_eval(MODE=1, DATASET=4)
        
        k_means_eval(MODE=1, DATASET=1)
        #k_means_eval(MODE=3, DATASET=1, REVERSE=True)
        k_means_eval(MODE=1, DATASET=2, preproc=reprocessing.reproc_4rt)
        k_means_eval(MODE=1, DATASET=3)
        k_means_eval(MODE=1, DATASET=4)
        plt.legend()
        plt.grid()