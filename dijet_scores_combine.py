import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def dijet_scores_combine_ROC_SIC_one(bg_scores, sg_scores, comb_method="+", res=None):
    labels=np.zeros(len(bg_scores)//2+len(sg_scores)//2)
    labels[len(bg_scores)//2:]=1
    if comb_method=="+":
        comb_jets_scr = lambda scr : scr[0::2]+scr[1::2]
    elif comb_method=="mean":
        comb_jets_scr = lambda scr : (scr[0::2]+scr[1::2])/2
    elif comb_method=="max":
        comb_jets_scr = lambda scr : np.maximum(scr[0::2], scr[1::2])
    elif comb_method=="min":
        comb_jets_scr = lambda scr : np.minimum(scr[0::2], scr[1::2])
    
    scores = comb_jets_scr(np.concatenate((bg_scores, sg_scores)))
    auc = roc_auc_score(labels, scores)
    
    fpr , tpr , thresholds = roc_curve(labels, scores)
    plt.figure("ROC")
    plt.plot(tpr, 1/fpr, label=f'AUC {comb_method}: {auc:.3f}')

    plt.figure("dist_dijet_"+comb_method, figsize=(10, 10))
    _, bins, _, = plt.hist(comb_jets_scr(bg_scores), histtype='step', label='bg', bins=40, density=True)
    plt.hist(comb_jets_scr(sg_scores), histtype='step', label='sig', bins=bins, density=True)
    plt.legend(title=f'AUC: {auc:.3f}')    

    plt.figure("SIC")
    sic_max=np.max(np.nan_to_num(tpr/fpr**0.5)[fpr!=0]) #some
    plt.plot(tpr, tpr/fpr**0.5, label="max {0:.2f}".format(sic_max))
    
    if res!=None:
        res["AUC_{:}".format(comb_method)]=auc
        res["fpr_{:}".format(comb_method)]=fpr
        res["tpr_{:}".format(comb_method)]=tpr
    

def dijet_scores_combine_ROC_SIC(bg_scores, sg_scores, comb_method="+", res=None):
    if hasattr(comb_method, '__iter__'):
        for cm in comb_method:
            dijet_scores_combine_ROC_SIC_one(bg_scores, sg_scores, comb_method=cm, res=res)
    else:
        dijet_scores_combine_ROC_SIC_one(bg_scores, sg_scores, comb_method=comb_method, res=res)
    


