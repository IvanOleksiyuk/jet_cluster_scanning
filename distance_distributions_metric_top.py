import pickle 
import numpy as np
import scipy as sp
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
import random
from scipy.ndimage import gaussian_filter
import reprocessing as rp
from cycler import cycler
from utilities import get_metric_from_res

pix=40
path="/media/ivan/Windows/datasets/image_data_sets/h5/top_samples/top-img-bkg.h5"
store = pd.HDFStore(path,  'r')
X=store.select("table", stop=2000)
X=X.values
X=X[:, :1600]
X=X.reshape(-1, 40, 40, 1)
store.close()
Y=X
SIGMA_array=[0, 0.5, 1, 2, 4, 8, 16]
reprocessings=[rp.reproc_none, rp.reproc_sq, rp.reproc_sqrt, rp.reproc_4rt, rp.reproc_heavi]
repr_names=["none", "sqrt", "4rt"]

random.seed(a=0, version=2)

#Y=np.random.rand(*Y.shape)
WD_arr=[]

for repr_name, reprocessing in zip(repr_names, reprocessings):
    plt.figure()
    ax=plt.gca()
    colormap = plt.get_cmap('turbo')
    custom_cycler = (cycler(color=[colormap(k) for k in np.linspace(0, 1, len(SIGMA_array))]))
    ax.set_prop_cycle(custom_cycler)
    
    for SIGMA in SIGMA_array[:4]:
        x = reprocessing(Y)
        x = gaussian_filter(x, sigma=[0, SIGMA, SIGMA, 0]) 
        x = rp.boundMSE(x)
        
        x=np.reshape(x, (x.shape[0], -1))
        
        dists=[]
        for i in range(20000):
            n=dists.append(np.sum((random.choice(x)-random.choice(x))**2))
        
        num=50
        hist, _ = np.histogram(dists, bins=np.linspace(0, 2, num+1), density=True)
        ds = [i for i in range(len(hist))]
        WD=sp.stats.wasserstein_distance(ds, ds, hist, np.ones(num)*1/2)
        WD_arr.append(WD)
        plt.hist(dists, bins=np.linspace(0, 2, num+1), histtype='step', density=True, label="s={:}, WD={:.3f}".format(SIGMA, WD))
    
    plt.axhline(0.5, color="gray", linestyle=":")
    
    plt.ylim(-0.1, 3)
    plt.legend()
    plt.title("{:} {:}pix.png".format(repr_name, pix))
    plt.savefig("plots/dist_distributions_QCDtop/{:}{:}pixtop.png".format(repr_name, pix), bbox_inches="tight")
    
    
list_k_clusters=[1, 2, 5, 10, 20, 50, 100, 200]
list_sigma=[0, 0.5, 1, 2]
list_prepr_names=["none", "sqrt", "4rt"]
list_score_names=["MinD", "KNC5", "logLds", "logLrhn"]
list_combination=["mean", "max", "min"]
markers=[".", "x", "+", "*"]
postpr="None"
metric_name="AUC" #"ieB@eS0.2"
plot_min_pop=False 
ylims=None
scr=list_score_names[1]
k=list_k_clusters[3]

plt.figure()

plt.xlabel("Wasserstein distance in prep+norm+sqrt")
plt.ylabel(metric_name+"with prep+nothing")
comb="one-jet"
i=0
for prep in list_prepr_names:
    for sigma in list_sigma:
        plt.axvline(WD_arr[i], color="lightgrey")
        i+=1

for j, scr in enumerate(list_score_names):
    for k in [20, 50, 100]:
        metric=[]
        for prep in list_prepr_names:
            for sigma in list_sigma:
                with open("char/MBQCDh5+toph52000_+toph5m{:}s{:}c100r{:}{:}KId0{:}/res.pickle".format(k, sigma, prep, postpr, scr), "rb") as f:
                    res=pickle.load(f)
                if comb=="one-jet":
                    metric.append(get_metric_from_res(res, metric_name))
        pass
        plt.plot(WD_arr, metric, markers[j], label=str(k)+" "+scr)
 
    
ax=plt.gca()
i=0

for prep in list_prepr_names:
    for sigma in list_sigma:
        ax.annotate(prep+str(sigma), (WD_arr[i], metric[i]))
        i+=1
      
plt.legend()
plt.title(comb)
plt.show()

