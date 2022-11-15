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
import resource

def make_bootstrap_array(n):
    np.sort(np.random.randint(0, 1000000, (1000000,)))
    a=np.arange(n)
    return np.bincount(np.random.choice(a, (n,)), minlength=n)
    

start_time_all = time.time()
#Hyperparameters
ID=0
W=100
k=50
retrain=0
steps=201
reproc=reprocessing.reproc_4rt
reproc_name=reprocessing.reproc_names(reproc)
MiniBatch=""
smearing=2
signal_fraction=0.05
window="26w60w"
Mjjmin_arr=np.linspace(2600, 6000-W, steps)
Mjjmax_arr=Mjjmin_arr+W

save_path="char/0kmeans_scan/{:}k{:}{:}ret{:}con{:}W{:}ste{:}rew{:}sme{:}ID{:}/".format(window, k, MiniBatch, retrain, signal_fraction, W, steps, reproc_name, smearing, ID)
print(save_path)
HP={"k": k,
    "MiniBatch": MiniBatch,
    "retrain": retrain,
    "signal_fraction": signal_fraction,
    "W": W,
    "steps": steps,
    "reproc_name": reproc_name, 
    "smearing": smearing, 
    "ID": ID,
    "Mjjmin_arr": Mjjmin_arr,
    "Mjjmax_arr": Mjjmax_arr}

os.makedirs(save_path, exist_ok=True)

random.seed(a=ID, version=2)
start_time_glb = time.time()
data_path="../../../hpcwork/rwth0934/LHCO_dataset/processed_io/"

mjj_bg=np.load(data_path+"mjj_bkg_sort.npy")
mjj_sg=np.load(data_path+"mjj_sig_sort.npy")

im_bg_f=h5py.File(data_path+'v2JetImSort_bkg.h5', 'r')
im_sg_f=h5py.File(data_path+'v2JetImSort_sig.h5', 'r')

print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
  
im_bg=im_bg_f['data'][:]
im_sg=im_sg_f['data'][:]

print("loading done --- %s seconds ---" % (time.time() - start_time_all))
print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
 
im_bg=im_bg.reshape((len(im_bg)*2, 40, 40))
im_sg=im_sg.reshape((len(im_sg)*2, 40, 40))

print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


if reproc!=None:
    batch=1000
    for i in range(len(im_bg)//batch):    
        im_bg[i*batch:(i+1)*batch] = reproc(im_bg[i*batch:(i+1)*batch])
    if (i+1)*batch<len(im_bg):
        im_bg[(i+1)*batch:] = reproc(im_bg[(i+1)*batch:])    
        
    for i in range(len(im_sg)//batch):    
        im_sg[i*batch:(i+1)*batch] = reproc(im_sg[i*batch:(i+1)*batch])
    if (i+1)*batch<len(im_sg):
        im_sg[(i+1)*batch:] = reproc(im_sg[(i+1)*batch:]) 

print("reweignhting done --- %s seconds ---" % (time.time() - start_time_all))
print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

if smearing!=0:
    smear=lambda x: gaussian_filter(x, sigma=[0, smearing, smearing])
    for i in range(len(im_bg)//batch):    
        im_bg[i*batch:(i+1)*batch] = smear(im_bg[i*batch:(i+1)*batch])
    if (i+1)*batch<len(im_bg):
        im_bg[(i+1)*batch:] = smear(im_bg[(i+1)*batch:])    
        
    for i in range(len(im_sg)//batch):    
        im_sg[i*batch:(i+1)*batch] = smear(im_sg[i*batch:(i+1)*batch])
    if (i+1)*batch<len(im_sg):
        im_sg[(i+1)*batch:] = smear(im_sg[(i+1)*batch:]) 

print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
print("smearign done --- %s seconds ---" % (time.time() - start_time_all))
 
im_bg=im_bg.reshape((len(im_bg)//2, 2, 40, 40))
im_sg=im_sg.reshape((len(im_sg)//2, 2, 40, 40))

print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

num_true=int(np.rint(signal_fraction*len(im_sg)))
print(num_true)
allowed=np.concatenate((np.zeros(len(im_sg)-num_true, dtype=bool), np.ones(num_true, dtype=bool)))
np.random.shuffle(allowed)

def Mjj_slise(Mjjmin, Mjjmax, bootstrap_bg=None):
    print("loading window", Mjjmin, Mjjmax)
    indexing_bg=np.logical_and(mjj_bg>=Mjjmin, mjj_bg<=Mjjmax)
    indexing_bg=np.where(indexing_bg)[0]
    
    indexing_sg=np.logical_and(mjj_sg>=Mjjmin, mjj_sg<=Mjjmax)
    indexing_sg=np.where(indexing_sg)[0]
    
    print(len(indexing_bg), "bg events found")
    print(len(indexing_sg), "sg events found")
    start_time = time.time()
    if bootstrap_bg is None:
        bg=im_bg[indexing_bg[0]:indexing_bg[-1]]
    else:
        bg=np.repeat(im_bg, bootstrap_bg[indexing_bg[0]:indexing_bg[-1]])
    sg=im_sg[indexing_sg[0]:indexing_sg[-1]]
    sg=sg[allowed[indexing_sg[0]:indexing_sg[-1]]]
    print("only", len(sg), "sg events taken")
    print("load --- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    data = np.concatenate((bg, sg))
    data = data.reshape((len(data)*2, 40, 40))
    print("concat --- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    data = data.reshape((len(data), 40*40))
    print("reshape --- %s seconds ---" % (time.time() - start_time))
    return data

def perform_scan_no_retrain(bootstrap=None):
    if MiniBatch:
        kmeans=MiniBatchKMeans(k)
    else:
        kmeans=KMeans(k)
    
    sideband=Mjj_slise(Mjjmin_arr[0], Mjjmax_arr[0])
    
    counts_windows=[]
    if retrain:
        kmeans_all=[]
    
    init='k-means++'
    for i in range(len(Mjjmin_arr)):
        data=Mjj_slise(Mjjmin_arr[i], Mjjmax_arr[i], bootstrap)
        if retrain:
            kmeans.fit(data)
            kmeans_all.append(copy.deepcopy(kmeans))
        predictions=kmeans.predict(data)
        counts_windows.append([np.sum(predictions==j) for j in range(k)])
        if retrain:
            if retrain==2:
                init=kmeans.cluster_centers_
            else:
                init=kmeans_all[0].cluster_centers_
            if MiniBatch:
                kmeans=MiniBatchKMeans(k, init=init)
            else:
                kmeans=KMeans(k, init=init)
        print("window {:.2f}-{:.2f}, N={:}".format(Mjjmin_arr[i], Mjjmax_arr[i], len(data)))
        print("eval_done --- %s seconds ---" % (time.time() - start_time_glb))
    if not retrain:
        kmeans_all.append(kmeans)
    return counts_windows, kmeans_all

kmeans_all_boot=[]
counts_windows_boot=[]
n_bootstraps=10
for l in range(n_bootstraps):
    bootstrap=make_bootstrap_array(len(im_bg))
    np.random.shuffle(allowed)
    sideband=Mjj_slise(Mjjmin, Mjjmax, bootstrap_bg=None)
    if MiniBatch:
        kmeans=MiniBatchKMeans(k)
    else:
        kmeans=KMeans(k)
    kmeans.fit()
    bg=np.repeat(im_bg, bootstrap)
    sg=im_sg[allowed]    
    counts_windows, kmeans_all = perform_scan(bootstrap=bootstrap)
    counts_windows_boot.append(counts_windows)
    kmeans_all_boot.append(kmeans_all)
res={}
res["counts_windows_boot"]=counts_windows_boot
res["HP"]=HP
res["kmeans_all_boot"]=kmeans_all_boot
pickle.dump(res, open(save_path+"res.pickle", "wb"))

im_bg_f.close()
im_sg_f.close()

res=pickle.load(open(save_path+"res.pickle", "rb"))
kmeans_all=res["kmeans_all"]

print("all done --- %s seconds ---" % (time.time() - start_time_all))
