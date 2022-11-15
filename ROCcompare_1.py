import numpy as np
import matplotlib.pyplot as plt
import pickle 
from cycler import cycler
import set_matplotlib_default as smd

def contamination_ROC_compare(names, 
                              methods, 
                              plot_name,
                              box_label,
                              output_folder="plots/ROC_comp/",
                              names_inn=None,
                              DVAE=None):

    inn_color="rosybrown"
    dvae_color="lightsteelblue"
    inn_linewidth=1
    dvae_linewidth=1
    inn_linestyle=":"
    dvae_linestyle=":"
    
    plt.figure(figsize=(5, 5))
    plt.grid( which='both', alpha=0.5 )
    plt.ylim(ymin=1, ymax=1000)
    plt.plot(np.linspace(0, 1, 1000), 1/np.linspace(0, 1, 1000), color="grey", linestyle="--")
    plt.yscale("log")
    
    if names_inn!=None:
        for i, name_inn in enumerate(names_inn):
            roc = np.load(name_inn)
            if i==0:
                plt.plot(roc[:,1], 1/(roc[:,0]+1e-16), color=inn_color, label="INN with reweightings", linewidth=inn_linewidth, linestyle=inn_linestyle)
            else:
                plt.plot(roc[:,1], 1/(roc[:,0]+1e-16), color=inn_color, linewidth=inn_linewidth, linestyle=inn_linestyle)
    
    if DVAE=="Aachen":
        fpr_test0 = []
        tpr_test0 = []
        auc_test0 = []
        
        dlab = "Aachen"
        num_topics = "2"
        for scaling in ["1","2"]:
            for befp in ["1.0","0.5","0.25"]:
                tpr_test0.append( np.loadtxt("results/Effic/"+dlab+"eff_t"+str(num_topics)+"s"+str(scaling)+"b"+str(befp))[0] )
                fpr_test0.append( np.loadtxt("results/Effic/"+dlab+"eff_t"+str(num_topics)+"s"+str(scaling)+"b"+str(befp))[1] )
        
        auc_test0 = [ 0.52, 0.58, 0.52, 0.62, 0.64, 0.65 ]
        
        labels_test0 = [ "$\\beta=$"+str(n) for n in [1.0,0.5,0.25] ] + [ "$\\beta=$"+str(n) for n in [1.0,0.5,0.25] ]
        labels_test0 = [ labels_test0[i]+str(" \n AUC$=$"+str(np.round(auc_test0[i],2))) for i in range(6) ]
        i=0
        plt.plot( tpr_test0[i], np.nan_to_num(1/fpr_test0[i]), label="DVAE with reweightings", color=dvae_color, linewidth=dvae_linewidth, linestyle=dvae_linestyle )

        for i in range(1, 3):
            plt.plot( tpr_test0[i], np.nan_to_num(1/fpr_test0[i]), color=dvae_color, linewidth=dvae_linewidth, linestyle=dvae_linestyle )
        for i in range(3):
            plt.plot( tpr_test0[i+3], np.nan_to_num(1/fpr_test0[i+3]), color=dvae_color, linewidth=dvae_linewidth, linestyle=dvae_linestyle)

    if DVAE=="Heidelberg":
        fpr_test1 = []
        tpr_test1 = []
        auc_test1 = []    
    
        dlab = "Heid"
        num_topics = "2"
        for scaling in ["1","2"]:
            for befp in ["1.0","0.5","0.25"]:
                tpr_test1.append( np.loadtxt("results/Effic/"+dlab+"eff_t"+str(num_topics)+"s"+str(scaling)+"b"+str(befp))[0] )
                fpr_test1.append( np.loadtxt("results/Effic/"+dlab+"eff_t"+str(num_topics)+"s"+str(scaling)+"b"+str(befp))[1] )
        
        auc_test1 = [ 0.61, 0.58, 0.54, 0.74, 0.76, 0.74 ]
            

        labels_test1 = [ "$\\beta=$"+str(n) for n in [1.0,0.5,0.25] ] + [ "$\\beta=$"+str(n) for n in [1.0,0.5,0.25] ]
        labels_test1 = [ labels_test1[i]+str(" \n AUC$=$"+str(np.round(auc_test1[i],2))) for i in range(6) ]
        i=0
        plt.plot( tpr_test1[i], np.nan_to_num(1/fpr_test1[i]), color=dvae_color, linewidth=dvae_linewidth, linestyle=dvae_linestyle, label="DVAE with reweightings" )
        for i in range(1, 3):
            plt.plot( tpr_test1[i], np.nan_to_num(1/fpr_test1[i]), color=dvae_color, linewidth=dvae_linewidth, linestyle=dvae_linestyle )
        for i in range(3):
            plt.plot( tpr_test1[i+3], np.nan_to_num(1/fpr_test1[i+3]), color=dvae_color, linewidth=dvae_linewidth, linestyle=dvae_linestyle )

    
    axs=plt.gca()
    axs.grid( which='both', alpha=0.5 )
    
    axs.set_xlabel( "$\epsilon_s$", fontproperties=smd.axislabelfont )
    axs.set_ylabel( "$\epsilon_b^{-1}$", fontproperties=smd.axislabelfont )
    
    axs.set_yscale('log')
    
    axs.set_xticks( [ np.round(i*0.2,1) for i in range(6) ] )
    axs.set_xticklabels( [ np.round(i*0.2,1) for i in range(6) ] , fontproperties=smd.tickfont )
    
    axs.set_ylim((1,1000))
    axs.set_xlim((0, 1))
    
    axs.set_yticks( [1,10,100,1000] )
    axs.set_yticklabels( [ "$10^{}$".format(i) for i in [0,1,2,3] ] , fontproperties=smd.tickfont, va='top' )
    
    #ax=plt.gca()
    #colormap = plt.get_cmap('jet')
    #custom_cycler = (cycler(color=[colormap(k) for k in np.linspace(0, 1, len(names))]))
    #ax.set_prop_cycle(custom_cycler)
    
    for name, method in zip(names, methods):
        char_path=name+"/res.pickle"
        res=pickle.load(open(char_path, "rb"))
        plt.plot(res['tpr'], 1/res['fpr'], label="AUC={:.3f}, {:}".format(res['AUC'], method))
    axs.legend( loc="upper right", prop=smd.labelfont )
    
    axs.text( 0.03, 1.24,  box_label, va='bottom', ha='left', fontproperties=smd.tickfont, bbox=dict(facecolor='white', alpha=0.8) )
    

    
    plt.savefig(output_folder+plot_name+".png", bbox_inches="tight")
    
    
if __name__ == "__main__":
    switches=[0, 1, 2, 3, 4]
    #Non-contaminated
    num=2
    if 0 in switches:
        dataset="QCD+top"
        contamination_ROC_compare(names=["char/{:}m100s3c100rnoneKId0MinD".format(dataset), 
                                         "char/{:}m100s3c100rnoneKId0KNC5".format(dataset),
                                         "char/{:}m100s3c100rnoneKId0logLmdui".format(dataset),
                                         "char/{:}m100s3c100rnoneKId0logLrhn".format(dataset), 
                                         "char/{:}m100s3c100rnoneKId0logLrh0nw".format(dataset),
                                         "char/{:}m100s3c100rnoneKId0logLrh1nw".format(dataset),
                                         "char/{:}m100s3c100rnoneKId0logLrh0".format(dataset),
                                         "char/MoG{:}m100s3c100rnoneKIbl0".format(dataset)],
                                  methods=["MinD", "KNC5", "MLLED", "MLL1D", "MLL0DNPN", "MLL1DNPN", "MLL0D", "GMMLL"], 
                                  plot_name=(str)(num)+"QCD+top")
    if 1 in switches:
        dataset="top+QCD"
        contamination_ROC_compare(names=["char/{:}m100s3c100rnoneKId0MinD".format(dataset), 
                                         "char/{:}m100s3c100rnoneKId0KNC5".format(dataset),
                                         "char/{:}m100s3c100rnoneKId0logLmdui".format(dataset),
                                         "char/{:}m100s3c100rnoneKId0logLrhn".format(dataset), 
                                         "char/{:}m100s3c100rnoneKId0logLrh0nw".format(dataset),
                                         "char/{:}m100s3c100rnoneKId0logLrh1nw".format(dataset),
                                         "char/{:}m100s3c100rnoneKId0logLrh0".format(dataset),
                                         "char/MoG{:}m100s3c100rnoneKIbl0".format(dataset)],
                                  methods=["MinD", "KNC5", "MLLED", "MLL1D", "MLL0DNPN", "MLL1DNPN", "MLL0D", "GMMLL"], 
                                  plot_name=(str)(num)+"top+QCD")
    if 2 in switches:
        dataset="QCDl+DMl"
        contamination_ROC_compare(names=["char/{:}m100s3c100r4rtKId0MinD".format(dataset), 
                                         "char/{:}m100s3c100r4rtKId0KNC5".format(dataset),
                                         "char/{:}m100s3c100r4rtKId0logLmdui".format(dataset),
                                         "char/{:}m100s3c100r4rtKId0logLrhn".format(dataset), 
                                         "char/{:}m100s3c100r4rtKId0logLrh0nw".format(dataset),
                                         "char/{:}m100s3c100r4rtKId0logLrh1nw".format(dataset),
                                         "char/{:}m100s3c100r4rtKId0logLrh0".format(dataset),
                                         "char/MoG{:}m100s3c100r4rtKIbl0".format(dataset)],
                                  methods=["MinD", "KNC5", "MLLED", "MLL1D", "MLL0DNPN", "MLL1DNPN", "MLL0D", "GMMLL"], 
                                  plot_name=(str)(num)+"QCDl+DMl")

    if 3 in switches:
        dataset="bg+Q200M100"
        contamination_ROC_compare(names=["char/{:}m100s3c100rnoneKId0MinD".format(dataset), 
                                         "char/{:}m100s3c100rnoneKId0KNC5".format(dataset),
                                         "char/{:}m100s3c100rnoneKId0logLmdui".format(dataset),
                                         "char/{:}m100s3c100rnoneKId0logLrhn".format(dataset), 
                                         "char/{:}m100s3c100rnoneKId0logLrh0nw".format(dataset),
                                         "char/{:}m100s3c100rnoneKId0logLrh1nw".format(dataset),
                                         "char/{:}m100s3c100rnoneKId0logLrh0".format(dataset),
                                         "char/MoG{:}m100s3c100rnoneKIbl0".format(dataset)],
                                  methods=["MinD", "KNC5", "MLLED", "MLL1D", "MLL0DNPN", "MLL1DNPN", "MLL0D", "GMMLL"],  
                                  plot_name=(str)(num)+"bg+Q200M100")
    if 4 in switches:
        dataset="bg+Q50M10"
        contamination_ROC_compare(names=["char/{:}m100s3c100rnoneKId0MinD".format(dataset), 
                                         "char/{:}m100s3c100rnoneKId0KNC5".format(dataset),
                                         "char/{:}m100s3c100rnoneKId0logLmdui".format(dataset),
                                         "char/{:}m100s3c100rnoneKId0logLrh1".format(dataset), 
                                         "char/{:}m100s3c100rnoneKId0logLrh0nw".format(dataset),
                                         "char/{:}m100s3c100rnoneKId0logLrh1nw".format(dataset),
                                         "char/{:}m100s3c100rnoneKId0logLrh0".format(dataset),
                                         "char/MoG{:}m100s3c100rnoneKIbl0".format(dataset)],
                                  methods=["MinD", "KNC5", "MLLED", "MLL1D", "MLL0DNPN", "MLL1DNPN", "MLL0D", "GMMLL"], 
                                  plot_name=(str)(num)+"bg+Q50M10")

    #contaminated
    if 5 in switches:
        dataset="QCD+top1000_+top"
        contamination_ROC_compare(names=["char/{:}m100s3c100rnoneKId0MinD".format(dataset), 
                                         "char/{:}m100s3c100rnoneKId0KNC5".format(dataset), 
                                         "char/{:}m100s3c100rnoneKId0logLrhn".format(dataset), 
                                         "char/{:}m100s3c100rnoneKId0logLmc".format(dataset),
                                         "char/{:}m100s3c100rnoneKIbl0".format(dataset),
                                         "char/MoG{:}m100s3c100rnoneKIdia0".format(dataset)], 
                                  methods=["MinD", "KNC5", "logLhnr", "logLmc", "Mogbl", "Mogdia"], 
                                  plot_name="QCD_top_cont")
    if 6 in switches:
        dataset="top+QCD1000_+QCD"
        contamination_ROC_compare(names=["char/{:}m100s3c100rnoneKId0MinD".format(dataset), 
                                         "char/{:}m100s3c100rnoneKId0KNC5".format(dataset),
                                         "char/{:}m100s3c100rnoneKId0logLmdu".format(dataset),
                                         "char/{:}m100s3c100rnoneKId0logLrhn".format(dataset), 
                                         "char/{:}m100s3c100rnoneKId0logLrh0nv".format(dataset),
                                         "char/{:}m100s3c100rnoneKId0logLrh1nv".format(dataset),
                                         "char/{:}m100s3c100rnoneKId0loglogLrh0".format(dataset),
                                         "char/MoG{:}m100s3c100rnoneKIbl0".format(dataset)],
                                  methods=["MinD", "KNC5", "logLmdu", "logLhnr", "logLrh0nv", "logLrh1nv", "logLrh0", "Mogbl"], 
                                  plot_name="top_QCD_cont")
    if 7 in switches:
        dataset=""
        contamination_ROC_compare(names=["char/{:}m100s3c100rnoneKId0MinD".format(dataset), 
                                         "char/{:}m100s3c100rnoneKId0KNC5".format(dataset), 
                                         "char/{:}m100s3c100rnoneKId0logLrhn".format(dataset), 
                                         "char/{:}m100s3c100rnoneKId0logLmc".format(dataset),
                                         "char/{:}m100s3c100rnoneKIbl0".format(dataset),
                                         "char/MoG{:}m100s3c100rnoneKIdia0".format(dataset)], 
                                  methods=["MinD", "KNC5", "logLhnr", "logLmc", "Mogbl", "Mogdia"], 
                                  plot_name="")

    if 8 in switches:
        dataset=""
        contamination_ROC_compare(names=["char/{:}m100s3c100rnoneKId0MinD".format(dataset), 
                                         "char/{:}m100s3c100rnoneKId0KNC5".format(dataset), 
                                         "char/{:}m100s3c100rnoneKId0logLrhn".format(dataset), 
                                         "char/{:}m100s3c100rnoneKId0logLmc".format(dataset),
                                         "char/{:}m100s3c100rnoneKIbl0".format(dataset),
                                         "char/MoG{:}m100s3c100rnoneKIdia0".format(dataset)], 
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