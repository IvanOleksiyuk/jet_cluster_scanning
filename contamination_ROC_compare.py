import numpy as np
import matplotlib.pyplot as plt
import pickle 
from cycler import cycler
import set_matplotlib_default

def contamination_ROC_compare(pref1="QCD",
                              pref2="top",
                              crop=100000,
                              main_name="+topm100s3c100rnoneKId0MinD", #main_name="+topm100s3c100rnoneKId0logLrhn"
                              cont_list=[0]):


    plt.figure(figsize=(5, 4))
    plt.grid(which='both', alpha=0.5)
    plt.ylim(ymin=1, ymax=1000)
    plt.plot(np.linspace(0, 1, 1000), 1/np.linspace(0, 1, 1000), color="gray")
    plt.yscale("log")
    
    ax=plt.gca()
    colormap = plt.get_cmap('jet')
    custom_cycler = (cycler(color=[colormap(k) for k in np.linspace(0, 1, len(cont_list))]))
    ax.set_prop_cycle(custom_cycler)
    
    for cont in cont_list:
        if cont>0:
            pref=pref1+"+"+pref2+"{:}".format(cont)+"_"
        else:
            pref=pref1
        cahr_name=pref+main_name
        char_path="char/"+cahr_name+"/res.pickle"
        res=pickle.load(open(char_path, "rb"))
        plt.plot(res['tpr'], 1/res['fpr'], label="S/B={:}, AUC={:.3f}".format(cont/crop, res['AUC']))
    plt.legend()
    plt.savefig("plots/ROC_cont/"+main_name+".png", bbox_inches="tight")
    
if __name__ == "__main__":
    switches=[1]
    if 0 in switches:
        contamination_ROC_compare(pref1="QCD",
                                  pref2="top",
                                  main_name="+topm100s3c100rnoneKId0MinD")
    if 1 in switches:
        contamination_ROC_compare(pref1="top",
                                  pref2="QCD",
                                  main_name="+QCDm100s3c100rnoneKId0MinD") 
    if 2 in switches:
        contamination_ROC_compare(pref1="QCDl",
                                  pref2="DMl",
                                  main_name="+DMlm100s3c100r4rtKId0MinD")
    if 3 in switches:
        contamination_ROC_compare(pref1="bg",
                                  pref2="Q200M100",
                                  main_name="+Q200M100m100s3c100rnoneKId0MinD")
    if 4 in switches:
        contamination_ROC_compare(pref1="bg",
                                  pref2="Q50M10",
                                  main_name="+Q50M10m100s3c100rnoneKId0MinD")
    if 5 in switches:
        contamination_ROC_compare(pref1="QCD",
                                  pref2="top",
                                  main_name="+topm100s3c100rnoneKId0logLrhn")
    if 6 in switches:
        contamination_ROC_compare(pref1="top",
                                  pref2="QCD",
                                  main_name="+QCDm100s3c100rnoneKId0logLrhn")
    if 7 in switches:
        contamination_ROC_compare(pref1="QCDl",
                                  pref2="DMl",
                                  main_name="+DMlm100s3c100r4rtKId0logLrhn")
    if 8 in switches:
        contamination_ROC_compare(pref1="bg",
                                  pref2="Q200M100",
                                  main_name="+Q200M100m100s3c100rnoneKId0logLrhn")
    if 9 in switches:
        contamination_ROC_compare(pref1="bg",
                                  pref2="Q50M10",
                                  main_name="+Q50M10m100s3c100rnoneKId0logLrhn")
