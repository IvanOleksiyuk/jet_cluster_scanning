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
steps=34
reproc=reprocessing.reproc_sqrt
reproc_name=reprocessing.reproc_names(reproc)
MiniBatch=""
smearing=1
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

im_bg=im_bg_f['data']
im_sg=im_sg_f['data']


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
    if reproc!=None:
        data = reproc(data)
    if smearing!=0:
        data = gaussian_filter(data, sigma=[0, smearing, smearing]) 
    data = data.reshape((len(data), 40*40))
    print("reproc --- %s seconds ---" % (time.time() - start_time))
    return data


print("load data --- %s seconds ---" % (time.time() - start_time_glb))


if MiniBatch:
    kmeans=MiniBatchKMeans(k)
else:
    kmeans=KMeans(k)


if not retrain:
    first_tr_data=Mjj_slise(Mjjmin_arr[0], Mjjmax_arr[0])
    print("first_tr_data", first_tr_data.shape)
    kmeans.fit(first_tr_data)
    print("k-means done --- %s seconds ---" % (time.time() - start_time_glb))
    
    plt.figure()
    plt.imshow(np.reshape(kmeans.cluster_centers_[0],(40, 40)))
    plt.savefig("plots/test/cluster_0.png")

counts_windows=[]
if retrain:
    kmeans_all=[]

init='k-means++'
for i in range(len(Mjjmin_arr)):
    data=Mjj_slise(Mjjmin_arr[i], Mjjmax_arr[i])
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
    
min_allowed_count=100
min_min_allowed_count=10
window_centers=(Mjjmin_arr+Mjjmax_arr)/2
counts_windows=np.array(counts_windows)
plt.figure()
plt.grid()
for j in range(k):
    plt.plot(window_centers, counts_windows[:, j])
plt.xlabel("m_jj")
plt.ylabel("n points from window")
plt.savefig(save_path+"kmeans_ni_mjj_total.png")
smallest_cluster_count_window=np.min(counts_windows, axis=1)
for i in range(len(window_centers)):
    if smallest_cluster_count_window[i]<min_allowed_count:
        if smallest_cluster_count_window[i]<min_min_allowed_count:
            plt.axvline(window_centers[i], color="black", alpha=0.6)
        else:
            plt.axvline(window_centers[i], color="black", alpha=0.3)
        
plt.savefig(save_path+"kmeans_ni_mjj_total_statAllowed.png")
    
partials_windows=np.zeros(counts_windows.shape)
for i in range(len(Mjjmin_arr)):
    partials_windows[i, :]=counts_windows[i, :]/np.sum(counts_windows[i, :])

plt.figure()
plt.grid()
for j in range(k):
    plt.plot((Mjjmin_arr+Mjjmax_arr)/2, partials_windows[:, j])
plt.xlabel("m_jj")
plt.ylabel("fraction of points in window")
plt.savefig(save_path+"kmeans_xi_mjj_total.png")

countmax_windows=np.zeros(counts_windows.shape)
for i in range(k):
    countmax_windows[:, i]=counts_windows[:, i]/np.max(counts_windows[:, i])

plt.figure()
plt.grid()
for j in range(k):
    plt.plot((Mjjmin_arr+Mjjmax_arr)/2, countmax_windows[:, j])
    
conts_bg=[]
conts_sg=[]
for Mjjmin, Mjjmax in zip(Mjjmin_arr, Mjjmax_arr):
    conts_bg.append(np.sum(np.logical_and(mjj_bg>=Mjjmin, mjj_bg<=Mjjmax)))
    conts_sg.append(np.sum(np.logical_and(np.logical_and(mjj_sg>=Mjjmin, mjj_sg<=Mjjmax), allowed)))                
conts_bg=np.array(conts_bg)
conts_sg=np.array(conts_sg)
conts=conts_bg+conts_sg
plt.plot((Mjjmin_arr+Mjjmax_arr)/2, conts/np.max(conts), "--")
plt.xlabel("m_jj")
plt.ylabel("n points from window/max(...)")
plt.savefig(save_path+"kmeans_xi_mjj_maxn.png")

for i in range(len(window_centers)):
    if smallest_cluster_count_window[i]<min_allowed_count:
        if smallest_cluster_count_window[i]<min_min_allowed_count:
            plt.axvline(window_centers[i], color="black", alpha=0.6)
        else:
            plt.axvline(window_centers[i], color="black", alpha=0.3)
        
plt.savefig(save_path+"kmeans_xi_mjj_maxn_statAllowed.png")

res={}
res["counts_windows"]=counts_windows
res["partials_windows"]=partials_windows
res["HP"]=HP
if retrain:
    res["kmeans_all"]=kmeans_all
else:
    res["kmeans"]=kmeans
    
pickle.dump(res, open(save_path+"res.pickle", "wb"))

im_bg_f.close()
im_sg_f.close()

res=pickle.load(open(save_path+"res.pickle", "rb"))
kmeans_all=res["kmeans_all"]

if retrain:
    diffs=[]
    for i in range(steps-1):
        diffs.append(np.sum((kmeans_all[i+1].cluster_centers_-kmeans_all[0].cluster_centers_)**2, axis=1)**0.5)
    diffs=np.array(diffs)
    plt.figure()
    plt.grid()
    mjj_arr=(Mjjmin_arr[:-1]+Mjjmax_arr[:-1])/4+(Mjjmin_arr[1:]+Mjjmax_arr[1:])/4
    for j in range(k):
        plt.plot(mjj_arr, diffs[:, j])
    plt.xlabel("m_jj")
    plt.ylabel("|mean-mean_0|")
    plt.savefig(save_path+"kmeans_cluster_abs_init_change.png")

if retrain:
    diffs=[]
    for i in range(steps-1):
        diffs.append(np.sum((kmeans_all[i+1].cluster_centers_-kmeans_all[i].cluster_centers_)**2, axis=1)**0.5)
    diffs=np.array(diffs)
    plt.figure()
    plt.grid()
    mjj_arr=(Mjjmin_arr[:-1]+Mjjmax_arr[:-1])/4+(Mjjmin_arr[1:]+Mjjmax_arr[1:])/4
    for j in range(k):
        plt.plot(mjj_arr, diffs[:, j])
    plt.xlabel("m_jj")
    plt.ylabel("|d mean/d mjj|")
    plt.savefig(save_path+"kmeans_cluster_abs_deriv.png")

if retrain:
    diffs=[]
    for i in range(steps-2):
        diffs.append(np.sum((-2*kmeans_all[i+1].cluster_centers_+kmeans_all[i].cluster_centers_+kmeans_all[i+2].cluster_centers_)**2, axis=1)**0.5)
    diffs=np.array(diffs)
    plt.figure()
    plt.grid()
    mjj_arr=(Mjjmin_arr[1:-1]+Mjjmax_arr[1:-1])/2
    for j in range(k):
        plt.plot(mjj_arr, diffs[:, j])
    plt.xlabel("m_jj")
    plt.ylabel("|d^2 mean/d mjj^2|")
    plt.savefig(save_path+"kmeans_cluster_abs_2deriv.png")

print("all done --- %s seconds ---" % (time.time() - start_time_all))