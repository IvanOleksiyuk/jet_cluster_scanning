import numpy as np
import random
import h5py
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans, MiniBatchKMeans
import time
import os
import reprocessing
import pickle
from scipy.ndimage import gaussian_filter
import copy
import set_matplotlib_default as smd
#Hyperparameters
ID=0
k=50
steps=40
W=4000/steps
MiniBatch=""
signal_fraction=0.05
window="26w60w"
Mjjmin_arr=np.linspace(2000, 6000-W, steps)
Mjjmax_arr=Mjjmin_arr+W

save_path="char/0kmeans_scan/ideal{:}k{:}{:}con{:}W{:}ste{:}ID{:}/".format(window, k, MiniBatch, signal_fraction, W, steps, ID)
print(save_path)
HP={"k": k,
    "MiniBatch": MiniBatch,
    "signal_fraction": signal_fraction,
    "W": W,
    "steps": steps, 
    "ID": ID,
    "Mjjmin_arr": Mjjmin_arr,
    "Mjjmax_arr": Mjjmax_arr}

os.makedirs(save_path, exist_ok=True)

random.seed(a=ID, version=2)
start_time_glb = time.time()
data_path="../../../hpcwork/rwth0934/LHCO_dataset/processed_io/"

mjj_bg=np.load(data_path+"mjj_bkg_sort.npy")
mjj_sg=np.load(data_path+"mjj_sig_sort.npy")


num_true=int(np.rint(signal_fraction*len(mjj_sg)))
print(num_true)
allowed=np.concatenate((np.zeros(len(mjj_sg)-num_true, dtype=bool), np.ones(num_true, dtype=bool)))
np.random.shuffle(allowed)

def Mjj_slise(Mjjmin, Mjjmax):
    print("loading window", Mjjmin, Mjjmax)
    indexing_bg=np.logical_and(mjj_bg>=Mjjmin, mjj_bg<=Mjjmax)
    indexing_bg=np.where(indexing_bg)[0]
    
    indexing_sg=np.logical_and(mjj_sg>=Mjjmin, mjj_sg<=Mjjmax)
    indexing_sg=np.where(indexing_sg)[0]
    
    print(len(indexing_bg), "bg events found")
    print(len(indexing_sg), "sg events found")

    return len(indexing_bg), len(indexing_sg)

bg=[]
sg=[]
for i in range(len(Mjjmin_arr)):
    bg_, sg_=Mjj_slise(Mjjmin_arr[i], Mjjmax_arr[i])
    bg.append(bg_)
    sg.append(sg_)

bg=np.array(bg)
sg=np.array(sg)
sg=sg*signal_fraction
    
window_centers=(Mjjmin_arr+Mjjmax_arr)/2

#visual bg and bg with signal
plt.figure()
ax=plt.gca()
ax.set_yscale('log')
plt.xlabel(r"$m_{jj}$ of window center")
plt.ylabel(r"$N(m_{jj})$ in a 100 GeV window")
plt.step(window_centers, bg, label=r"$\epsilon = 0$", where='mid') 
plt.step(window_centers, bg+sg, label=r"$\epsilon = 0.005$", where='mid')
plt.legend()
plt.savefig(save_path+"bg_withsg.png", bbox_inches="tight")

plt.figure()
plt.plot(window_centers, sg/np.sqrt(bg+sg), label="chisq/ndim {:}".format(np.mean((sg/np.sqrt(bg+sg))**2)))
plt.legend()
plt.savefig(save_path+"significance_of_signal.png")

#print(sg)
print(np.sum((sg/np.sqrt(bg+sg))**2))
print(np.mean((sg/np.sqrt(bg+sg))**2))
print((np.sum((sg/np.sqrt(bg+sg))**2)-len(sg))/np.sqrt(2*len(sg)))