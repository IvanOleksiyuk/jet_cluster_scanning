import pickle 
import numpy as np
import scipy as sp
import scipy.stats
import matplotlib.pyplot as plt
import random
from scipy.ndimage import gaussian_filter
import reprocessing as rp
from cycler import cycler
from utilities import get_metric_from_res
import random 

random.seed(a=0, version=2)

pix=40
Y=pickle.load(open("/home/ivan/datasets/2K_BG{:}.pickle".format(pix), "rb"))
SIGMA_array=[0, 0.5, 1, 2, 4, 8, 16]
reprocessings=[rp.reproc_none, rp.reproc_sq, rp.reproc_sqrt, rp.reproc_4rt, rp.reproc_heavi]
repr_names=["none",  "sqrt", "4rt", "sq", "heavi"]

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
        x = gaussian_filter(x, sigma=[0, SIGMA, SIGMA]) 
        #x = rp.boundMSE(x)
        
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
    plt.savefig("plots/dist_distributions_NoBounding/{:}{:}pix.png".format(repr_name, pix), bbox_inches="tight")
    
plt.show()




