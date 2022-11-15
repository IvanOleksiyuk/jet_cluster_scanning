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
import os  
from sklearn.manifold import TSNE

plt.close("all")

def gaussian(x, mean, sigma, weight):
    return weight/(sigma*(2*np.pi)**0.5)*np.exp(-(x-mean)**2/(2*sigma**2))

def half_gaussian(x, mean, sigma, weight):
    out= weight/(sigma*(2*np.pi)**0.5)*np.exp(-(x-mean)**2/(2*sigma**2))
    out[x<mean]=weight
    return weight/(sigma*(2*np.pi)**0.5)*np.exp(-(x-mean)**2/(2*sigma**2))

train_mode="d"
REVERSE=False
DATASET=1
SIGMA=3
crop=10000
preproc=None
Id=0
MODE=1
TOP_DOWN=True
CUSTOM_MODEL_NAME=False
ensemble=None 
k=10
n_clusters=k
SAVE_CHAR=True
DO_TSNE=True
DO_TSNE_CENTROIDS=True
TSNE_scores=True

random.seed(a=0, version=2)

pref, pref2, tra_data_path, con_data_path, bg_val_data_path, sg_val_data_path = dataset_path_and_pref(DATASET, REVERSE)

X_bg_val=prepare_data(bg_val_data_path, preproc=preproc, SIGMA=SIGMA)
X_sg_val=prepare_data(sg_val_data_path, preproc=preproc, SIGMA=SIGMA)
X_bg=prepare_data(tra_data_path, crop=crop, preproc=preproc, SIGMA=SIGMA)

MODEl_path="models/{:}m{:}s{:}c{:}r{:}KI{:}{:}.pickle".format(pref, k, SIGMA, crop//1000, reprocessing.reproc_names(preproc), train_mode, Id)
kmeans=pickle.load(open(MODEl_path, "rb"))
distances=kmeans.transform(X_bg)
means=[]
sigmas=[]
weights=[]

for i in range(k):
    dist=distances[kmeans.labels_==i, i]
    means.append(np.mean(dist))
    sigmas.append(np.std(dist))
    weights.append(len(dist)/crop)
    
dist_tr=kmeans.transform(X_bg)
dist_bg_val=kmeans.transform(X_bg_val)
dist_sg_val=kmeans.transform(X_sg_val)

bg_L=0
sg_L=0
cols=1

fig, ax = plt.subplots(k, cols, figsize=(cols*4, k*4.2), squeeze=False)

part_L_bg=np.zeros(dist_bg_val.shape)
part_L_sg=np.zeros(dist_sg_val.shape)
part_L_tr=np.zeros(dist_tr.shape)
bg_L, sg_L, tr_L=0, 0, 0
for i in range(k):
    part_L_tr[:, i]=gaussian(dist_tr[:, i], means[i], sigmas[i], weights[i])
    part_L_bg[:, i]=gaussian(dist_bg_val[:, i], means[i], sigmas[i], weights[i])
    part_L_sg[:, i]=gaussian(dist_sg_val[:, i], means[i], sigmas[i], weights[i])
    bg_L-=part_L_bg[:, i]
    sg_L-=part_L_sg[:, i]
    tr_L-=part_L_tr[:, i]

if DO_TSNE:
    
    tr_scores = -np.log(-tr_L)
    sg_scores = -np.log(-sg_L)
    max_score=max(np.max(tr_scores), np.max(sg_scores))
    min_score=min(np.min(tr_scores), np.min(sg_scores))
    
    tr_scores_nrm=(tr_scores-min_score)/(max_score-min_score)
    sg_scores_nrm=(sg_scores-min_score)/(max_score-min_score)
    
    n_sig=1000
    n_bg=1000
    random.seed(a=10, version=2)
    IDs_TSNE=np.random.randint(0, X_bg.shape[0]-1, n_bg, )
    IDs_TSNE_sig=np.random.randint(0, X_sg_val.shape[0]-1, n_sig, )
    if DO_TSNE_CENTROIDS:
        centoids=kmeans.cluster_centers_
        labels_TSNE=np.concatenate((kmeans.labels_[IDs_TSNE], -1*np.ones(n_sig), -2*np.ones(n_clusters)))
        Y = TSNE(n_components=2, n_iter=1000, random_state=0).fit_transform(np.concatenate([X_bg[IDs_TSNE], X_sg_val[IDs_TSNE_sig], centoids]))
    else:
        labels_TSNE=np.concatenate((kmeans.labels_[IDs_TSNE], -1*np.ones(n_sig)))
        Y = TSNE(n_components=2, n_iter=1000, random_state=0).fit_transform(np.concatenate([X_bg[IDs_TSNE], X_sg_val[IDs_TSNE_sig]]))
    plt.figure(figsize=(10, 10))
    u_labels = np.unique(kmeans.labels_)
    for i in u_labels:
        if n_clusters<=10:
            plt.scatter(Y[labels_TSNE==i, 0], Y[labels_TSNE==i, 1], label=i+1)
        else:
            plt.scatter(Y[labels_TSNE==i, 0], Y[labels_TSNE==i, 1])
    plt.scatter(Y[labels_TSNE==-1, 0], Y[labels_TSNE==-1, 1], label="signal", marker="x")
    if DO_TSNE_CENTROIDS:
        plt.scatter(Y[labels_TSNE==-2, 0], Y[labels_TSNE==-2, 1], label="centroids", marker="X", color="black", s=70)
    plt.legend()
    
    if TSNE_scores:
        plt.figure(figsize=(10, 10))
        plt.scatter(Y[labels_TSNE>=0, 0], Y[labels_TSNE>=0, 1], c=tr_scores_nrm[IDs_TSNE], cmap="turbo", marker="o")
        plt.scatter(Y[labels_TSNE==-1, 0], Y[labels_TSNE==-1, 1], c=sg_scores_nrm[IDs_TSNE_sig], label="signal", marker="x", cmap="turbo")
        if DO_TSNE_CENTROIDS:
            plt.scatter(Y[labels_TSNE==-2, 0], Y[labels_TSNE==-2, 1], label="centroids", marker="X", color="black", s=90)
        plt.legend()
        
        plt.figure(figsize=(10, 10))
        plt.scatter(Y[labels_TSNE>=0, 0], Y[labels_TSNE>=0, 1], c=tr_scores_nrm[IDs_TSNE], cmap="turbo", marker="o")
        if DO_TSNE_CENTROIDS:
            plt.scatter(Y[labels_TSNE==-2, 0], Y[labels_TSNE==-2, 1], label="centroids", marker="X", color="black", s=90)
        plt.legend()
        
        plt.figure(figsize=(10, 10))
        plt.scatter(Y[labels_TSNE==-1, 0], Y[labels_TSNE==-1, 1], c=sg_scores_nrm[IDs_TSNE_sig], label="signal", marker="x", cmap="turbo")
        if DO_TSNE_CENTROIDS:
            plt.scatter(Y[labels_TSNE==-2, 0], Y[labels_TSNE==-2, 1], label="centroids", marker="X", color="black", s=90)
        plt.legend()
    
    print("done TSNE")

bins=np.linspace(0, 150, 100)
for i in range(k):
    plt.sca(ax[i][0])
    plt.yticks([])
    plt.hist(-np.log(part_L_bg[:, i]), bins=bins, histtype='step')
    plt.hist(-np.log(part_L_sg[:, i]), bins=bins, histtype='step')

plt.figure()
_, bins, _, = plt.hist(bg_L, histtype='step', label='bg', bins=50, density=True)
plt.hist(sg_L, histtype='step', label='sig', bins=bins, density=True)
plt.xlabel("-L")

plt.figure()
_, bins, _, = plt.hist(bg_L, histtype='step', label='bg', bins=50, density=True)
plt.hist(sg_L, histtype='step', label='sig', bins=bins, density=True)
plt.xlabel("-L")
plt.ylim(ymin=0, ymax=0.1)

bg_logL=-np.log(-bg_L)
sg_logL=-np.log(-sg_L)

max_bg_logL=max(bg_logL[np.isfinite(bg_logL)])
max_sg_logL=max(sg_logL[np.isfinite(sg_logL)])
max_logL=max(max_bg_logL, max_sg_logL)
"""
bins=np.linspace(min(bg_logL), max_bg_logL, 50)
bg_logL[np.logical_not(np.isfinite(bg_logL))]=max_logL*1.1
sg_logL[np.logical_not(np.isfinite(sg_logL))]=max_logL*1.1

plt.figure()
_, bins, _, = plt.hist(bg_logL, histtype='step', label='bg', bins=40, density=True)
plt.hist(sg_logL, histtype='step', label='sig', bins=bins, density=True)
plt.xlabel("-log(L)")

plt.figure()
_, bins, _, = plt.hist(bg_logL, histtype='step', label='bg', bins=40, density=True)
plt.hist(sg_logL, histtype='step', label='sig', bins=bins, density=True)
plt.xlabel("-log(L)")
plt.ylim(ymin=0, ymax=0.005)


labels=np.concatenate((np.zeros(len(X_bg_val)), np.ones(len(X_sg_val))))
auc = roc_auc_score(labels, np.append(bg_L, sg_L))
plt.legend(title=f'AUC: {auc:.2f}')
plt.show()
fpr , tpr , thresholds = roc_curve(labels, np.append(bg_L, sg_L))
plt.figure()
plt.grid()
plt.plot(tpr, 1/fpr)
plt.ylim(ymin=1, ymax=1000)
plt.plot(np.linspace(0, 1, 1000), 1/np.linspace(0, 1, 1000), color="gray")
plt.yscale("log")
plt.legend(title=f'AUC: {auc:.2f}')

plt.figure()
plt.grid()
sic_max=np.max(np.nan_to_num(tpr/fpr**0.5)[fpr!=0]) #some
plt.plot(tpr, tpr/fpr**0.5, label="max {0:.2f}".format(sic_max))
plt.legend()
    
if SAVE_CHAR:
    path="char/{:}m{:}s{:}c{:}r{:}KI{:}{:}/".format(pref, k, SIGMA, crop//1000, reprocessing.reproc_names(preproc), train_mode, Id)+"L1/"
    os.makedirs(path, exist_ok=True)
    k=0
    k+=1
    plt.figure(k)
    plt.savefig(path+"distrib_of_likelyhoods.png", bbox_inches="tight")
    k+=1
    plt.figure(k)
    plt.savefig(path+"distL.png", bbox_inches="tight")
    k+=1
    plt.figure(k)
    plt.savefig(path+"distL_sk.png", bbox_inches="tight")
    k+=1
    plt.figure(k)
    plt.savefig(path+"distlogL.png", bbox_inches="tight")
    k+=1
    plt.figure(k)
    plt.savefig(path+"distlogL_sk.png", bbox_inches="tight")
    k+=1
    plt.figure(k)
    plt.savefig(path+"ROC.png", bbox_inches="tight")
    k+=1
    plt.figure(k)
    plt.savefig(path+"SIC.png", bbox_inches="tight")
"""
   


