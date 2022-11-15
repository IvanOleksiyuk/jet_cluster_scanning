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
def gaussian(x, mean, sigma, weight):
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
k_arr=[1, 2, 4, 8, 16, 32, 64, 128, 256]
SAVE_CHAR=True

random.seed(a=0, version=2)

pref, pref2, tra_data_path, con_data_path, bg_val_data_path, sg_val_data_path = dataset_path_and_pref(DATASET, REVERSE)

X_bg_val=prepare_data(bg_val_data_path, preproc=preproc, SIGMA=SIGMA)
X_sg_val=prepare_data(sg_val_data_path, preproc=preproc, SIGMA=SIGMA)
X_bg=prepare_data(tra_data_path, crop=crop, preproc=preproc, SIGMA=SIGMA)

for k in k_arr

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
    
dist_bg_val=kmeans.transform(X_bg_val)
dist_sg_val=kmeans.transform(X_sg_val)

bg_L=0
sg_L=0

for i in range(k):
   bg_L=-gaussian(dist_bg_val[:, i], means[i], sigmas[i], weights[i])
   sg_L=-gaussian(dist_sg_val[:, i], means[i], sigmas[i], weights[i])

_, bins, _, = plt.hist(bg_L, histtype='step', label='bg', bins=50, density=True)
plt.hist(sg_L, histtype='step', label='sig', bins=bins, density=True)
plt.xlabel("-L")
#plt.yscale("log")

bg_logL=-np.log(-bg_L)
sg_logL=-np.log(-sg_L)

max_bg_logL=max(bg_logL[np.isfinite(bg_logL)])
max_sg_logL=max(sg_logL[np.isfinite(sg_logL)])
max_logL=max(max_bg_logL, max_sg_logL)

bins=np.linspace(min(bg_logL), max_bg_logL, 50)
bg_logL[np.logical_not(np.isfinite(bg_logL))]=max_logL*1.1
sg_logL[np.logical_not(np.isfinite(sg_logL))]=max_logL*1.1

plt.figure()
_, bins, _, = plt.hist(bg_logL, histtype='step', label='bg', bins=40, density=True)
plt.hist(sg_logL, histtype='step', label='sig', bins=bins, density=True)
plt.xlabel("-log(L)")
#plt.yscale("log")

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
    plt.figure(1)
    plt.savefig(path+"distL.png", bbox_inches="tight")
    plt.figure(2)
    plt.savefig(path+"distlogL.png", bbox_inches="tight")
    plt.figure(3)
    plt.savefig(path+"ROC.png", bbox_inches="tight")
    plt.figure(4)
    plt.savefig(path+"SIC.png", bbox_inches="tight")

   


