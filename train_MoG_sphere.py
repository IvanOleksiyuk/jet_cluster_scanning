import numpy as np
import random 
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from scipy.ndimage import gaussian_filter
import pickle 
import reprocessing
from sklearn.metrics import roc_auc_score, roc_curve
import os
from dataset_path_and_pref import dataset_path_and_pref, prepare_data
import copy

def train_MoG_sphere(DI, pref, k, reg_covar, SIGMA, crop, cont, preproc, Id, train_mode, data=None, MINI_BATCH=False):
    if data is None:
        X_tr=prepare_data(DI["tra_data_path"], field=DI["tra_data_field"], crop=crop, preproc=preproc, SIGMA=SIGMA)
        if cont>0:
            X_cont=prepare_data(DI["con_data_path"], field=DI["con_data_field"], crop=cont, preproc=preproc, SIGMA=SIGMA)
            X_tr=np.concatenate((X_tr, X_cont))
    else:
        X_tr=data
    if train_mode=="s":
        MoG_sphere=GaussianMixture(n_components=k, covariance_type="spherical", random_state=Id, n_init=1, max_iter=10, init_params="random").fit(X_tr)
    if train_mode=="d":
        MoG_sphere=GaussianMixture(n_components=k, covariance_type="spherical", random_state=Id, n_init=1).fit(X_tr)
    if train_mode=="dia":
        MoG_sphere=GaussianMixture(n_components=k, covariance_type="diag", random_state=Id, n_init=1, reg_covar=10**reg_covar).fit(X_tr)
    if train_mode=="bl":
        MoG_sphere=GaussianMixture(n_components=k, covariance_type="spherical", random_state=Id, n_init=1, reg_covar=10**reg_covar).fit(X_tr)
    model_path="models_MoG_sphere/{:}m{:}reg{:}s{:}c{:}r{:}KI{:}{:}.pickle".format(pref, reg_covar, k, SIGMA, crop//1000, reprocessing.reproc_names(preproc), train_mode, Id)
    pickle.dump(MoG_sphere, open(model_path, "wb"))
    if not (train_mode in ["d", "s", "dia", "bl"]):
        print("invalid train mode!")
    return X_tr, MoG_sphere
    
def train_or_load_MoG_sphere(DI, pref, k, reg_covar, SIGMA, crop, cont, preproc, Id, train_mode, data=None, return_data=False, MINI_BATCH=False):
    model_path="models_MoG_sphere/{:}m{:}reg{:}s{:}c{:}r{:}KI{:}{:}.pickle".format(pref, reg_covar, k, SIGMA, crop//1000, reprocessing.reproc_names(preproc), train_mode, Id)
    if os.path.isfile(model_path):
        print("loading trained model", model_path)
        MoG_sphere=pickle.load(open(model_path, "rb"))
        if data is None:
            X_tr=prepare_data(DI["tra_data_path"], field=DI["tra_data_field"], crop=crop, preproc=preproc, SIGMA=SIGMA)
            if cont>0:
                X_cont=prepare_data(DI["con_data_path"], field=DI["con_data_field"], crop=cont, preproc=preproc, SIGMA=SIGMA)
                X_tr=np.concatenate((X_tr, X_cont))
            return X_tr, MoG_sphere
        else:
            return data, MoG_sphere
    else:
        print("training a new model", model_path)
        X_tr, MoG_sphere =train_MoG_sphere(DI, pref, k, reg_covar, SIGMA, crop, cont, preproc, Id, train_mode=train_mode, data=data, MINI_BATCH=MINI_BATCH)
        return X_tr, MoG_sphere

def MoG_sphere_process(dataset=1,
                    n_clusters=10,
                    SIGMA=3,
                    crop=10000,
                    cont=0,
                    reg_covar=-6,
                    preproc=None,
                    Id=0,                         
                    characterise=False,
                    SAVE_CHAR=True,
                    REVERSE=False,
                    DO_TSNE=True,
                    DO_TSNE_CENTROIDS=False, 
                    train_mode="d", #explanation below
                    data=None, #training data that was alredy loaded and preprocessed by the prevous iteration of the algorythm and so can be reused
                    full_mean_diffs=False,
                    non_smeared_mean=False, 
                    TSNE_scores=True, 
                    SCORE_TYPE="KNC", 
                    MINI_BATCH=False, 
                    return_scores=False, 
                    return_data=True): 
# To train_mode
# we are not so interested in the clustering itself at this point thus we dont 
# really require the best and fully convergent clustering, repeating clustering 
# with 10 initialisations is thus a waist of resources (it may be better to 
# build an ensemble out of such 10 instead of picking one of 10 with best clustering) 
# "d" - default as it is default in scikit
# "s" - stochastic train with only 1 initialisation for only 10 steps and random initialisation 
    plt.close("all")
    k=n_clusters
    random.seed(a=10, version=2)
    
    DI = dataset_path_and_pref(dataset, REVERSE)
        
    if cont>0:
        DI["pref"]=DI["pref"]+"+"+DI["pref2"]+"{:}".format(cont)+"_"
    
    hyp=(DI["pref"], k, reg_covar, SIGMA, crop, cont, preproc, Id)
    
    X_tr, MoG_sphere = train_or_load_MoG_sphere(DI, *hyp, train_mode=train_mode, data=data, MINI_BATCH=MINI_BATCH)
    
    cov=copy.deepcopy(MoG_sphere.covariances_)
    cov=np.array(cov)
    cov.sort()
    print(np.sqrt(cov))
    print(cov)
    print(np.median(cov))
    
    if return_scores or characterise:
        X_bg_val=prepare_data(DI["bg_val_data_path"], field=DI["bg_val_data_field"], preproc=preproc, SIGMA=SIGMA)
        X_sg_val=prepare_data(DI["sg_val_data_path"], field=DI["sg_val_data_field"], preproc=preproc, SIGMA=SIGMA)
        tr_scores = -MoG_sphere.score_samples(X_tr)
        bg_scores = -MoG_sphere.score_samples(X_bg_val)
        sg_scores = -MoG_sphere.score_samples(X_sg_val)
        #centroid_scores = -MoG_sphere.score_samples(MoG_sphere.means_)
        
    if characterise:
        #%%
        if DO_TSNE:
            max_score=np.quantile(np.concatenate([tr_scores, sg_scores]), 0.99)#max(np.max(tr_scores), np.max(sg_scores))
            min_score=np.quantile(np.concatenate([tr_scores, sg_scores]), 0.01)#min(np.min(tr_scores), np.min(sg_scores))
            
            tr_scores_nrm=tr_scores#-min_score)/(max_score-min_score)
            sg_scores_nrm=sg_scores#-min_score)/(max_score-min_score)
            
            n_sig=1000
            n_bg=1000
            random.seed(a=10, version=2)
            IDs_TSNE=np.random.randint(0, X_tr.shape[0]-1, n_bg, )
            IDs_TSNE_sig=np.random.randint(0, X_sg_val.shape[0]-1, n_sig, )
            if DO_TSNE_CENTROIDS:
                centoids=MoG_sphere.means_
                labels_TSNE=np.concatenate((np.ones(n_bg), -1*np.ones(n_sig), -2*np.ones(n_clusters)))
                Y = TSNE(n_components=2, n_iter=1000, random_state=0).fit_transform(np.concatenate([X_tr[IDs_TSNE], X_sg_val[IDs_TSNE_sig], centoids]))
            else:
                labels_TSNE=np.concatenate((np.ones(n_bg), -1*np.ones(n_sig)))
                Y = TSNE(n_components=2, n_iter=1000, random_state=0).fit_transform(np.concatenate([X_tr[IDs_TSNE], X_sg_val[IDs_TSNE_sig]]))
            
            if TSNE_scores:
                plt.figure(figsize=(10, 10))
                plt.scatter(Y[labels_TSNE>=0, 0], Y[labels_TSNE>=0, 1], c=tr_scores_nrm[IDs_TSNE], cmap="turbo", marker="o", vmin=min_score, vmax=max_score)
                plt.scatter(Y[labels_TSNE==-1, 0], Y[labels_TSNE==-1, 1], c=sg_scores_nrm[IDs_TSNE_sig], label="signal", marker="x", cmap="turbo", vmin=min_score, vmax=max_score)
                if DO_TSNE_CENTROIDS:
                    plt.scatter(Y[labels_TSNE==-2, 0], Y[labels_TSNE==-2, 1], label="centroids", marker="X", color="black", s=90)
                plt.legend()
                
                plt.figure(figsize=(10, 10))
                plt.scatter(Y[labels_TSNE>=0, 0], Y[labels_TSNE>=0, 1], c=tr_scores_nrm[IDs_TSNE], cmap="turbo", marker="o", vmin=min_score, vmax=max_score)
                if DO_TSNE_CENTROIDS:
                    plt.scatter(Y[labels_TSNE==-2, 0], Y[labels_TSNE==-2, 1], label="centroids", marker="X", color="black", s=90)
                plt.legend()
                
                plt.figure(figsize=(10, 10))
                plt.scatter(Y[labels_TSNE==-1, 0], Y[labels_TSNE==-1, 1], c=sg_scores_nrm[IDs_TSNE_sig], label="signal", marker="x", cmap="turbo", vmin=min_score, vmax=max_score)
                if DO_TSNE_CENTROIDS:
                    plt.scatter(Y[labels_TSNE==-2, 0], Y[labels_TSNE==-2, 1], label="centroids", marker="X", color="black", s=90)
                plt.legend()
                
            print("done TSNE")
        
        labels=np.concatenate((np.zeros(len(X_bg_val)), np.ones(len(X_sg_val))))
        auc = roc_auc_score(labels, np.append(bg_scores, sg_scores))
        
        fpr , tpr , thresholds = roc_curve(labels, np.append(bg_scores, sg_scores))
        plt.figure()
        plt.grid()
        plt.plot(tpr, 1/fpr)
        plt.ylim(ymin=1, ymax=1000)
        plt.plot(np.linspace(0, 1, 1000), 1/np.linspace(0, 1, 1000), color="gray")
        plt.yscale("log")
        plt.legend(title=f'AUC: {auc:.3f}')
        
        plt.figure(figsize=(10, 10))
        _, bins, _, = plt.hist(bg_scores, histtype='step', label='bg', bins=40, density=True)
        plt.hist(sg_scores, histtype='step', label='sig', bins=bins, density=True)
        #plt.hist(centroid_scores, histtype='step', label='centr', bins=bins, density=True)
        plt.legend(title=f'AUC: {auc:.3f}')
        
        plt.figure()
        plt.grid()
        sic_max=np.max(np.nan_to_num(tpr/fpr**0.5)[fpr!=0]) #some
        plt.plot(tpr, tpr/fpr**0.5, label="max {0:.2f}".format(sic_max))
        plt.legend()
        
        if SAVE_CHAR:
            path="char/MoG{:}+{:}m{:}s{:}c{:}r{:}KI{:}{:}/".format(DI["pref"], DI["pref2"], k, SIGMA, crop//1000, reprocessing.reproc_names(preproc), train_mode, Id)
            os.makedirs(path, exist_ok=True)
            k=0
            if DO_TSNE:
                k+=1
                plt.figure(k)
                plt.savefig(path+"TSNE_scores.png", bbox_inches="tight")
                k+=1
                plt.figure(k)
                plt.savefig(path+"TSNE_scores_tra.png", bbox_inches="tight")
                k+=1
                plt.figure(k)
                plt.savefig(path+"TSNE_scores_sig.png", bbox_inches="tight")
            k+=1
            plt.figure(k)
            plt.savefig(path+"ROC.png", bbox_inches="tight")
            k+=1
            plt.figure(k)
            plt.savefig(path+"dist.png", bbox_inches="tight")
            k+=1
            plt.figure(k)
            plt.savefig(path+"SIC.png", bbox_inches="tight")
            
            res={}
            res["fpr"]=fpr
            res["tpr"]=tpr
            res["AUC"]=auc
            pickle.dump(res, open(path+"res.pickle", "wb"))
            print(path+"res.pickle")
        
if __name__ == "__main__":
    # possible_switches=[1, 2]
    switches=[0]
    
    if 0 in switches:
        cont=0
        train_mode="bl"
        reg_covar=-11

        MoG_sphere_process(dataset="1h5",
                    n_clusters=100,
                    cont=cont,
                    SIGMA=3,
                    crop=100000,
                    preproc=None, 
                    Id=0,
                    reg_covar=reg_covar,
                    REVERSE=False,                                 
                    characterise=True,
                    SAVE_CHAR=True,
                    DO_TSNE=True,
                    DO_TSNE_CENTROIDS=True, 
                    train_mode=train_mode, 
                    data=None)
    
    if 0.1 in switches:
        cont=0
        train_mode="bl"
        reg_covar=-11

        MoG_sphere_process(dataset=1,
                    n_clusters=100,
                    cont=cont,
                    SIGMA=3,
                    crop=100000,
                    preproc=None, 
                    Id=0,
                    reg_covar=reg_covar,
                    REVERSE=False,                                 
                    characterise=True,
                    SAVE_CHAR=True,
                    DO_TSNE=True,
                    DO_TSNE_CENTROIDS=True, 
                    train_mode=train_mode, 
                    data=None)
        
        MoG_sphere_process(dataset=1,
                    n_clusters=100,
                    cont=cont,
                    SIGMA=3,
                    crop=100000,
                    preproc=None, 
                    Id=0,
                    reg_covar=reg_covar,
                    REVERSE=True,                                 
                    characterise=True,
                    SAVE_CHAR=True,
                    DO_TSNE=True,
                    DO_TSNE_CENTROIDS=True, 
                    train_mode=train_mode, 
                    data=None)
        
        MoG_sphere_process(dataset=2,
                    n_clusters=100,
                    cont=cont,
                    SIGMA=3,
                    crop=100000,
                    preproc=reprocessing.reproc_4rt,
                    Id=0,
                    reg_covar=reg_covar,
                    REVERSE=False,                                 
                    characterise=True,
                    SAVE_CHAR=True,
                    DO_TSNE=True,
                    DO_TSNE_CENTROIDS=True, 
                    train_mode=train_mode, 
                    data=None)
        
        MoG_sphere_process(dataset=2,
                    n_clusters=100,
                    cont=cont,
                    SIGMA=3,
                    crop=100000,
                    preproc=reprocessing.reproc_4rt,
                    Id=0,
                    reg_covar=reg_covar,
                    REVERSE=False,                                 
                    characterise=True,
                    SAVE_CHAR=True,
                    DO_TSNE=True,
                    DO_TSNE_CENTROIDS=True, 
                    train_mode=train_mode, 
                    data=None)
        
        MoG_sphere_process(dataset=3,
                    n_clusters=100,
                    cont=cont,
                    SIGMA=3,
                    crop=100000,
                    preproc=None, 
                    Id=0,
                    reg_covar=reg_covar,
                    REVERSE=False,                                 
                    characterise=True,
                    SAVE_CHAR=True,
                    DO_TSNE=True,
                    DO_TSNE_CENTROIDS=True, 
                    train_mode=train_mode, 
                    data=None)

        MoG_sphere_process(dataset=4,
                    n_clusters=100,
                    cont=cont,
                    SIGMA=3,
                    crop=100000,
                    preproc=None, 
                    Id=0,
                    reg_covar=reg_covar,
                    REVERSE=False,                                 
                    characterise=True,
                    SAVE_CHAR=True,
                    DO_TSNE=True,
                    DO_TSNE_CENTROIDS=True, 
                    train_mode=train_mode, 
                    data=None)
        """       
    if 1 in switches:
        k_list=[2]
        for k in k_list:
            MoG_sphere_process(dataset=5,
                        n_clusters=k,
                        SIGMA=0,
                        crop=100000,
                        cont=1000,
                        preproc=None, #reprocessing.reproc_4rt,
                        Id=0,
                        reg_covar=-6,
                        REVERSE=False,                                 
                        characterise=True,
                        SAVE_CHAR=True,
                        DO_TSNE=True,
                        DO_TSNE_CENTROIDS=True, 
                        train_mode="full", 
                        data=None)
        """