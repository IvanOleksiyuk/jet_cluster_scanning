import numpy as np
import matplotlib.pyplot as plt
import pickle 
from cycler import cycler


def contamination_ROC_compare(names, 
                              methods, 
                              plot_name):

    plt.figure(figsize=(10, 8))
    plt.grid()
    plt.ylim(ymin=1, ymax=1000)
    plt.plot(np.linspace(0, 1, 1000), 1/np.linspace(0, 1, 1000), color="gray")
    plt.yscale("log")
    
    #ax=plt.gca()
    #colormap = plt.get_cmap('jet')
    #custom_cycler = (cycler(color=[colormap(k) for k in np.linspace(0, 1, len(names))]))
    #ax.set_prop_cycle(custom_cycler)
    
    for name, method in zip(names, methods):
        char_path=name+"/res.pickle"
        res=pickle.load(open(char_path, "rb"))
        plt.plot(res['tps'], 1/res['fpr'], label="AUC={:.3f}, {:}".format(res['AUC'], method))
    
    plt.legend()
    plt.savefig("plots/ROC_comp/"+plot_name+".png", bbox_inches="tight")
    
    
if __name__ == "__main__":
    switches=[5, 6]
    #Non-contaminated
    if 0 in switches:
        contamination_ROC_compare(names=["char/QCD+topm100s3c100rnoneKId0MinD", 
                                         "char/QCD+topm100s3c100rnoneKId0KNC5", 
                                         "char/QCD+topm100s3c100rnoneKId0logLrhn", 
                                         "char/QCD+topm100s3c100rnoneKId0logLmc",
                                         "char/MoGQCD+topm100s3c100rnoneKIbl0",
                                         "char/MoGQCD+topm100s3c100rnoneKIdia0"], 
                                  methods=["MinD", "KNC5", "logLhnr", "logLmc", "Mogbl", "Mogdia"], 
                                  plot_name="QCD_top")
    if 1 in switches:
        contamination_ROC_compare(names=["char/top+QCDm100s3c100rnoneKId0MinD", 
                                         "char/top+QCDm100s3c100rnoneKId0KNC5", 
                                         "char/top+QCDm100s3c100rnoneKId0logLrhn", 
                                         "char/top+QCDm100s3c100rnoneKId0logLmc",
                                         "char/MoGtop+QCDm100s3c100rnoneKIbl0",
                                         "char/MoGtop+QCDm100s3c100rnoneKIdia0"], 
                                  methods=["MinD", "KNC5", "logLhnr", "logLmc", "Mogbl", "Mogdia"], 
                                  plot_name="top_QCD")
    if 2 in switches:
        contamination_ROC_compare(names=["char/QCDl+DMlm100s3c100r4rtKId0MinD", 
                                         "char/QCDl+DMlm100s3c100r4rtKId0KNC5", 
                                         "char/QCDl+DMlm100s3c100r4rtKId0logLrhn", 
                                         "char/QCDl+DMlm100s3c100r4rtKId0logLmc",
                                         "char/MoGQCDl+DMlm100s3c100r4rtKIbl0",
                                         "char/MoGQCDl+DMlm100s3c100r4rtKIdia0"], 
                                  methods=["MinD", "KNC5", "logLhnr", "logLmc", "Mogbl", "Mogdia"], 
                                  plot_name="DMl")

    if 3 in switches:
        contamination_ROC_compare(names=["char/bg+Q200M100m100s3c100rnoneKId0MinD", 
                                         "char/bg+Q200M100m100s3c100rnoneKId0KNC5", 
                                         "char/bg+Q200M100m100s3c100rnoneKId0logLrhn", 
                                         "char/bg+Q200M100m100s3c100rnoneKId0logLmc",
                                         "char/MoGbg+Q200M100m100s3c100rnoneKIbl0",
                                         "char/MoGbg+Q200M100m100s3c100rnoneKIdia0"], 
                                  methods=["MinD", "KNC5", "logLhnr", "logLmc", "Mogbl", "Mogdia"],  
                                  plot_name="Q200M100")
    if 4 in switches:
        contamination_ROC_compare(names=["char/bg+Q50M10m100s3c100rnoneKId0MinD", 
                                         "char/bg+Q50M10m100s3c100rnoneKId0KNC5", 
                                         "char/bg+Q50M10m100s3c100rnoneKId0logLrhn", 
                                         "char/bg+Q50M10m100s3c100rnoneKId0logLmc",
                                         "char/MoGbg+Q50M10m100s3c100rnoneKIbl0",
                                         "char/MoGbg+Q50M10m100s3c100rnoneKIdia0"], 
                                  methods=["MinD", "KNC5", "logLhnr", "logLmc", "Mogbl", "Mogdia"],  
                                  plot_name="Q50M10")

    #contaminated
    if 5 in switches:
        dataset="QCD+top1000_+top"
        contamination_ROC_compare(names=["char/{:}m100s3c100rnoneKId0MinD".format(dataset), 
                                         "char/{:}m100s3c100rnoneKId0KNC5".format(dataset), 
                                         "char/{:}m100s3c100rnoneKId0logLrhn".format(dataset), 
                                         "char/{:}m100s3c100rnoneKId0logLmc".format(dataset),
                                         "char/{:}m100s3c100rnoneKIbl0".format(dataset),
                                         "char/{:}m100s3c100rnoneKIdia0".format(dataset)], 
                                  methods=["MinD", "KNC5", "logLhnr", "logLmc", "Mogbl", "Mogdia"], 
                                  plot_name="QCD_top_cont")
    if 6 in switches:
        dataset="top+QCD1000_+QCD"
        contamination_ROC_compare(names=["char/{:}m100s3c100rnoneKId0MinD".format(dataset), 
                                         "char/{:}m100s3c100rnoneKId0KNC5".format(dataset), 
                                         "char/{:}m100s3c100rnoneKId0logLrhn".format(dataset), 
                                         "char/{:}m100s3c100rnoneKId0logLmc".format(dataset),
                                         "char/{:}m100s3c100rnoneKIbl0".format(dataset),
                                         "char/{:}m100s3c100rnoneKIdia0".format(dataset)], 
                                  methods=["MinD", "KNC5", "logLhnr", "logLmc", "Mogbl", "Mogdia"], 
                                  plot_name="top_QCD_cont")
    if 7 in switches:
        dataset=""
        contamination_ROC_compare(names=["char/{:}m100s3c100rnoneKId0MinD".format(dataset), 
                                         "char/{:}m100s3c100rnoneKId0KNC5".format(dataset), 
                                         "char/{:}m100s3c100rnoneKId0logLrhn".format(dataset), 
                                         "char/{:}m100s3c100rnoneKId0logLmc".format(dataset),
                                         "char/{:}m100s3c100rnoneKIbl0".format(dataset),
                                         "char/{:}m100s3c100rnoneKIdia0".format(dataset)], 
                                  methods=["MinD", "KNC5", "logLhnr", "logLmc", "Mogbl", "Mogdia"], 
                                  plot_name="")

    if 8 in switches:
        dataset=""
        contamination_ROC_compare(names=["char/{:}m100s3c100rnoneKId0MinD".format(dataset), 
                                         "char/{:}m100s3c100rnoneKId0KNC5".format(dataset), 
                                         "char/{:}m100s3c100rnoneKId0logLrhn".format(dataset), 
                                         "char/{:}m100s3c100rnoneKId0logLmc".format(dataset),
                                         "char/{:}m100s3c100rnoneKIbl0".format(dataset),
                                         "char/{:}m100s3c100rnoneKIdia0".format(dataset)], 
                                  methods=["MinD", "KNC5", "logLhnr", "logLmc", "Mogbl", "Mogdia"], 
                                  plot_name="")
    if 9 in switches:
        dataset=""
        contamination_ROC_compare(names=["char/{:}m100s3c100rnoneKId0MinD".format(dataset), 
                                         "char/{:}m100s3c100rnoneKId0KNC5".format(dataset), 
                                         "char/{:}m100s3c100rnoneKId0logLrhn".format(dataset), 
                                         "char/{:}m100s3c100rnoneKId0logLmc".format(dataset),
                                         "char/{:}m100s3c100rnoneKIbl0".format(dataset),
                                         "char/{:}m100s3c100rnoneKIdia0".format(dataset)], 
                                  methods=["MinD", "KNC5", "logLhnr", "logLmc", "Mogbl", "Mogdia"], 
                                  plot_name="")