import matplotlib
from sklearn.cluster import KMeans
import numpy as np
import random 
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.ndimage import gaussian_filter
import pickle
import reprocessing
from sklearn.metrics import roc_auc_score, roc_curve
from functools import partial
from plot_2d_array import plot_2d_array

def metric_auc(labels, scores):
    return roc_auc_score(labels, scores)

def metric_eps_B_inv(labels, scores, eps_s=0.3):
    fpr , tpr , thresholds = roc_curve(labels, scores)
    return 1/fpr[find_nearest_idx(tpr, eps_s)]

plt.rcParams.update({'font.size': 18})
def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

random.seed(a=0, version=2)
 
plt.close("all")

REVERSE=True
DATASET=1
SIGMA=3
CROP=10000
preproc=reprocessing.reproc_none
Id=2
MODE=3
TOP_DOWN=True
CUSTOM_MODEL_NAME="models/Rm{:}s3c10KId2.pickle"

if DATASET==1:
    if REVERSE:
        pref="top"
    else:
        pref="QCD"
    bg_val_data_path="C:/bachelor work/Spyder/image_data_sets/Xval-40KQCD-pre3-2.pickle"
    sg_val_data_path="C:/bachelor work/Spyder/image_data_sets/Xval-40Ktop-pre3-2.pickle"
if DATASET==2:
    if REVERSE:
        pref="DMl"
    else:
        pref="QCDl"
    bg_val_data_path="C:/bachelor work/Spyder/image_data_sets/Xval-40KQCDl.pickle"
    sg_val_data_path="C:/bachelor work/Spyder/image_data_sets/Xval-40KDMl.pickle"
if REVERSE:
    bg_val_data_path, sg_val_data_path = sg_val_data_path, bg_val_data_path

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

def prepare_data(path, CROP=-1, preproc=None):    
    X = pickle.load(open(path, "rb"))
    X = X[:CROP]
    if preproc is not None:
        print("preprocessing active")
        X = preproc(X)
    X = gaussian_filter(X, sigma=[0, SIGMA, SIGMA, 0]) 
    X = X.reshape((X.shape[0], 1600))
    return X

X_bg_val=prepare_data(bg_val_data_path, preproc=preproc)
X_sg_val=prepare_data(sg_val_data_path, preproc=preproc)

def comb_loss(losses, n=1):
    return np.mean(losses[:, :n], 1)

def eval_metric(bg_losses, sg_losses, n, metric):
    bg_scores=comb_loss(bg_losses, n)
    sg_scores=comb_loss(sg_losses, n)
    labels=np.concatenate((np.zeros(len(bg_scores)), np.ones(len(sg_scores))))
    return metric(labels, np.append(bg_scores, sg_scores))

n=1
k_arr=[10, 50]

metric=[[] for k in k_arr]
plt.figure(figsize=(12, 12))
bins=np.linspace(0, 0.06, 80)

for i, k in enumerate(k_arr):
    cmap_bg = matplotlib.cm.get_cmap('autumn')
    cmap_sg = matplotlib.cm.get_cmap('winter')
    
    if CUSTOM_MODEL_NAME:
        MODEl_path=CUSTOM_MODEL_NAME.format(k)
    else:
        MODEl_path="models/{:}m{:}s{:}c{:}r{:}KId{:}.pickle".format(pref, k, SIGMA, CROP//1000, reprocessing.reproc_names(preproc), Id)
    print(MODEl_path)
    kmeans=pickle.load(open(MODEl_path, "rb"))
    bg_losses = kmeans.transform(X_bg_val)
    sg_losses = kmeans.transform(X_sg_val)
    if n==1:
        test_sg_labels=sg_losses.argmin(1)
        sg_min=sg_losses.min(1)
        most_sg_like=np.argmax(np.bincount(test_sg_labels))
        plt.hist(sg_min[test_sg_labels==most_sg_like], bins=bins, color=cmap_sg(i/len(k_arr)), alpha=0.2, label="most QCD like\ncluster")
    bg_losses.sort(1)
    sg_losses.sort(1)
    bg_score=comb_loss(bg_losses, n=n)
    sg_score=comb_loss(sg_losses, n=n)
    plt.hist(bg_score, bins=bins, histtype='step', label=k, color=cmap_bg(i/(len(k_arr))))
    plt.hist(sg_score, bins=bins, histtype='step', label=k, color=cmap_sg(i/len(k_arr)))

        
plt.xlabel("min_dist")
plt.ylabel("counts")
plt.legend()