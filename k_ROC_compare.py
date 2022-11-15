import numpy as np
import matplotlib.pyplot as plt
import pickle 
from cycler import cycler


def contamination_ROC_compare(pref1="QCD",
                              pref2="top",
                              crop=100000,
                              cont=0,
                              main_name="+topm{:}s3c100rnoneKId0MinD", #main_name="+topm100s3c100rnoneKId0logLrhn"
                              k_list=[10, 100]):


    plt.figure(figsize=(10, 8))
    plt.grid()
    plt.ylim(ymin=1, ymax=10000)
    plt.plot(np.linspace(0, 1, 1000), 1/np.linspace(0, 1, 1000), color="gray")
    plt.yscale("log")
    
    ax=plt.gca()
    colormap = plt.get_cmap('turbo')
    custom_cycler = (cycler(color=[colormap(k) for k in np.linspace(0, 1, len(k_list))]))
    ax.set_prop_cycle(custom_cycler)
    
    for k in k_list:
        if cont>0:
            pref=pref1+"+"+pref2+"_"
        else:
            pref=pref1
        cahr_name=pref+main_name.format(k)
        char_path="char/"+cahr_name+"/res.pickle"
        res=pickle.load(open(char_path, "rb"))
        plt.plot(res['tps'], 1/res['fpr'], label="k={:}, AUC={:.3f}".format(k, res['AUC']))
    plt.legend()
    plt.savefig("plots/ROC_k/"+main_name.format(0)+".png", bbox_inches="tight")
    
if __name__ == "__main__":
    switches=[1]
    if 0 in switches:
        contamination_ROC_compare(pref1="2d10s",
                                  pref2="2d1s4m",
                                  main_name="+2d1s4mm{:}s0c100rnoneKId0MinD")
    if 1 in switches:
        contamination_ROC_compare(pref1="bg",
                                  pref2="Q50M10",
                                  main_name="+Q50M10m{:}s3c100rnoneKId0MinD")

