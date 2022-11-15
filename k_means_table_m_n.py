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
plt.close("all")
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
    n_arr=[1, 2, 4, 8, 16, 32, 64, 128, 256],
    k_arr=[1, 2, 4, 8, 16, 32, 64, 128, 256]): #"models/Rm{:}s3c10KId2.pickle"
    
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
    
    X_bg_val=prepare_data(bg_val_data_path, preproc=preproc, SIGMA=SIGMA)
    X_sg_val=prepare_data(sg_val_data_path, preproc=preproc, SIGMA=SIGMA)
    
    metric=[[] for k in k_arr]
    
    for i, k in enumerate(k_arr):
        if ensemble is None:
            if CUSTOM_MODEL_NAME:
                MODEl_path=CUSTOM_MODEL_NAME.format(k)
            else:
                MODEl_path="models/{:}m{:}s{:}c{:}r{:}KI{:}{:}.pickle".format(pref, k, SIGMA, CROP//1000, reprocessing.reproc_names(preproc), fast_train, Id)
            print(MODEl_path)
            kmeans=pickle.load(open(MODEl_path, "rb"))
            bg_losses = kmeans.transform(X_bg_val)
            sg_losses = kmeans.transform(X_sg_val)
            bg_losses.sort(1)
            sg_losses.sort(1)
            for n in n_arr:
                if n<=k:
                    bg_scores=comb_loss(bg_losses, n)
                    sg_scores=comb_loss(sg_losses, n)
                    metric[i].append(eval_metric(bg_scores, sg_scores, metric_func))
                if n>k:
                    metric[i].append(np.nan)
                print(n, k)   
        else:
            bg_losses_list=[]
            sg_losses_list=[]     
            for Id in ensemble:
                MODEl_path="models/{:}m{:}s{:}c{:}r{:}KI{:}{:}.pickle".format(pref, k, SIGMA, CROP//1000, reprocessing.reproc_names(preproc), fast_train, Id)
                kmeans=pickle.load(open(MODEl_path, "rb"))
                bg_losses_list.append(kmeans.transform(X_bg_val))
                sg_losses_list.append(kmeans.transform(X_sg_val))
                bg_losses_list[-1].sort(1)
                sg_losses_list[-1].sort(1)
            for n in n_arr:
                if n<=k:
                    bg_scores=0
                    sg_scores=0
                    for j in range(len(ensemble)):
                        bg_scores+=comb_loss(bg_losses_list[j], n)
                        sg_scores+=comb_loss(sg_losses_list[j], n)
                    bg_scores/=len(ensemble)
                    sg_scores/=len(ensemble)
                    metric[i].append(eval_metric(bg_scores, sg_scores, metric_func))
                if n>k:
                    metric[i].append(np.nan)
                print(n, k)
                
    metric=np.array(metric)
    plot_2d_array(metric, TOP_DOWN)
    plt.title("{:}+{:} s{:} c{:}K r{:} I{:}{:}".format(pref, pref2, SIGMA, CROP//1000, reprocessing.reproc_names(preproc), fast_train, Id)+" "+metric_name)
    if TOP_DOWN:
        k_arr.reverse()
    plt.xticks(np.arange(len(n_arr))+0.5, n_arr)
    plt.yticks(np.arange(len(k_arr))+0.5, k_arr)
    plt.ylabel("k in k-means")
    plt.xlabel("k in kNN")
    plt.savefig("plots/{:}+{:}s{:}c{:}r{:}KI{:}{:}".format(pref, pref2, SIGMA, CROP//1000, reprocessing.reproc_names(preproc), fast_train, Id)+metric_name+".png", bbox_inches='tight')
    pickle.dump(metric, open("plots/data/{:}+{:}s{:}c{:}r{:}KI{:}{:}".format(pref, pref2, SIGMA, CROP//1000, reprocessing.reproc_names(preproc), fast_train, Id)+metric_name+".pickle", "wb"))
    if TOP_DOWN:
        k_arr.reverse()
    
if __name__ == "__main__":
    switches=[7]
    if 1 in switches:
        k_means_eval(MODE=1, DATASET=3)
        k_means_eval(MODE=2, DATASET=3)
    if 2 in switches:
        k_means_eval(MODE=1, DATASET=1, fast_train="f", ensemble=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        k_means_eval(MODE=2, DATASET=1, fast_train="f", ensemble=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        k_means_eval(MODE=3, DATASET=1, fast_train="f", ensemble=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    if 3 in switches:
        k_means_eval(MODE=1, DATASET=1, fast_train="s", ensemble=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        k_means_eval(MODE=2, DATASET=1, fast_train="s", ensemble=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        k_means_eval(MODE=3, DATASET=1, fast_train="s", ensemble=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
    if 4 in switches:
        n_arr=[1, 2, 4, 8, 16, 32, 64, 128]
        k_arr=[1, 2, 4, 8, 16, 32, 64, 128]
        k_means_eval(MODE=1, DATASET=1, fast_train="d", Id=0, n_arr=n_arr, k_arr=k_arr, REVERSE=True)
        k_means_eval(MODE=2, DATASET=1, fast_train="d", Id=0, n_arr=n_arr, k_arr=k_arr, REVERSE=True)
        k_means_eval(MODE=3, DATASET=1, fast_train="d", Id=0, n_arr=n_arr, k_arr=k_arr, REVERSE=True)
        k_means_eval(MODE=1, DATASET=1, fast_train="f", ensemble=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], n_arr=n_arr, k_arr=k_arr, REVERSE=True)
        k_means_eval(MODE=2, DATASET=1, fast_train="f", ensemble=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], n_arr=n_arr, k_arr=k_arr, REVERSE=True)
        k_means_eval(MODE=3, DATASET=1, fast_train="f", ensemble=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], n_arr=n_arr, k_arr=k_arr, REVERSE=True)
        k_means_eval(MODE=1, DATASET=1, fast_train="s", ensemble=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], n_arr=n_arr, k_arr=k_arr, REVERSE=True)
        k_means_eval(MODE=2, DATASET=1, fast_train="s", ensemble=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], n_arr=n_arr, k_arr=k_arr, REVERSE=True)
        k_means_eval(MODE=3, DATASET=1, fast_train="s", ensemble=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], n_arr=n_arr, k_arr=k_arr, REVERSE=True)

    if 5 in switches:
        n_arr=[1, 2, 4, 8, 16, 32, 64, 128]
        k_arr=[1, 2, 4, 8, 16, 32, 64, 128]
        k_means_eval(MODE=1, DATASET=2, fast_train="d", Id=0, n_arr=n_arr, k_arr=k_arr, preproc=reprocessing.reproc_4rt)
        k_means_eval(MODE=2, DATASET=2, fast_train="d", Id=0, n_arr=n_arr, k_arr=k_arr, preproc=reprocessing.reproc_4rt)
        k_means_eval(MODE=3, DATASET=2, fast_train="d", Id=0, n_arr=n_arr, k_arr=k_arr, preproc=reprocessing.reproc_4rt)
        k_means_eval(MODE=1, DATASET=2, fast_train="f", ensemble=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], n_arr=n_arr, k_arr=k_arr, preproc=reprocessing.reproc_4rt)
        k_means_eval(MODE=2, DATASET=2, fast_train="f", ensemble=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], n_arr=n_arr, k_arr=k_arr, preproc=reprocessing.reproc_4rt)
        k_means_eval(MODE=3, DATASET=2, fast_train="f", ensemble=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], n_arr=n_arr, k_arr=k_arr, preproc=reprocessing.reproc_4rt)
        k_means_eval(MODE=1, DATASET=2, fast_train="s", ensemble=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], n_arr=n_arr, k_arr=k_arr, preproc=reprocessing.reproc_4rt)
        k_means_eval(MODE=2, DATASET=2, fast_train="s", ensemble=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], n_arr=n_arr, k_arr=k_arr, preproc=reprocessing.reproc_4rt)
        k_means_eval(MODE=3, DATASET=2, fast_train="s", ensemble=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], n_arr=n_arr, k_arr=k_arr, preproc=reprocessing.reproc_4rt)
        
    if 6 in switches: # Topic 15.0
        n_arr=[1, 2, 4, 8, 16, 32, 64, 128, 256]
        k_arr=[1, 2, 4, 8, 16, 32, 64, 128, 256]
        k_means_eval(MODE=1, DATASET=3, fast_train="d", Id=0, n_arr=n_arr, k_arr=k_arr)
        k_means_eval(MODE=1, DATASET=4, fast_train="d", Id=0, n_arr=n_arr, k_arr=k_arr)
        k_means_eval(MODE=2, DATASET=3, fast_train="d", Id=0, n_arr=n_arr, k_arr=k_arr)
        k_means_eval(MODE=2, DATASET=4, fast_train="d", Id=0, n_arr=n_arr, k_arr=k_arr)
        
    if 7 in switches: # Topic 15.2
        n_arr=[1, 2, 4, 8, 16, 32, 64, 128, 256]
        k_arr=[1, 2, 4, 8, 16, 32, 64, 128, 256]
        k_means_eval(MODE=1, DATASET=3, fast_train="d", Id=0, n_arr=n_arr, k_arr=k_arr, preproc=reprocessing.reproc_sqrt)
        k_means_eval(MODE=1, DATASET=3, fast_train="d", Id=0, n_arr=n_arr, k_arr=k_arr, preproc=reprocessing.reproc_4rt)
        k_means_eval(MODE=1, DATASET=4, fast_train="d", Id=0, n_arr=n_arr, k_arr=k_arr, preproc=reprocessing.reproc_sqrt)
        k_means_eval(MODE=1, DATASET=4, fast_train="d", Id=0, n_arr=n_arr, k_arr=k_arr, preproc=reprocessing.reproc_4rt)