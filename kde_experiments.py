from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.neighbors import KernelDensity
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.ndimage import gaussian_filter
import pickle
import reprocessing
from sklearn.metrics import roc_auc_score, roc_curve
import os
from sklearn_extra.cluster import KMedoids
from dataset_path_and_pref import dataset_path_and_pref, prepare_data

"""
def gaussian_mult(x, mean, sigma, weight):
    s=sigma/(2**0.5)
    a=mean**2/s**2
    print(a)
    print(s)
    return weight/((s*(2*np.pi)**0.5)**a)*np.exp(-(x)**2/(2*s**2))
"""

def gaussian_mult_c(x, sigma_0, weight, d=1):
    return weight/((sigma_0*(2*np.pi)**0.5)**d)*np.exp(-(x)**2/(2*sigma_0**2))

def gaussian_mult(x, mean, sigma, weight, d=1):
    s=mean
    return weight/((s*(2*np.pi)**0.5)**d)*np.exp(-(x)**2/(2*s**2))

def exponential_slope_parameterless(x, mean, sigma, weight):
    return np.exp(-x)*weight

def gaussian(x, mean, sigma, weight):
    return weight/(sigma*(2*np.pi)**0.5)*np.exp(-(x-mean)**2/(2*sigma**2))

def half_gaussian(x, mean, sigma, weight, smear=1):
    sigma_=sigma*smear
    out = 1/(sigma_*(2*np.pi)**0.5)*np.exp(-(x-mean)**2/(2*sigma_**2))
    out[x<mean]=np.max(out)
    return weight*out

def half_gaussian_norm(x, mean, sigma, weight, smear=1):
    sigma_=sigma*smear
    out = 1/(sigma_*(2*np.pi)**0.5)*np.exp(-(x-mean)**2/(2*sigma_**2))
    out[x<mean]=np.max(out)
    out/=(0.5+mean*np.max(out))
    return weight*out

def sum_n_mins(losses, knc):
    losses_cop=np.copy(losses)
    losses_cop.sort(1)
    return np.mean(losses_cop[:, :knc], 1)

def likelyhood_estimation(kmeans, crop, k, X_tr, X_bg_val, X_sg_val, density_function, log_likelyhood=True):
    dist_tr = kmeans.transform(X_tr)
    dist_bg_val = kmeans.transform(X_bg_val)
    dist_sg_val = kmeans.transform(X_sg_val)
    means=[]
    sigmas=[]
    weights=[]
    for i in range(k):
        dist=dist_tr[kmeans.labels_==i, i]
        means.append(np.mean(dist))
        sigmas.append(np.std(dist))
        weights.append(len(dist)/crop)
    """
    print("means")
    print(means)
    print("weights")
    print(np.array(weights))
    plt.figure(figsize=(10, 10))
    plt.scatter(means, np.array(weights)*crop)
    plt.xlabel("r")
    plt.ylabel("n")
    plt.yscale("log")
    plt.grid()
    """
    part_L_bg=np.zeros(dist_bg_val.shape)
    part_L_sg=np.zeros(dist_sg_val.shape)
    part_L_tr=np.zeros(dist_tr.shape)
    bg_L, sg_L, tr_L=0, 0, 0
    for i in range(k):
        part_L_tr[:, i]=density_function(dist_tr[:, i], means[i], sigmas[i], weights[i])
        part_L_bg[:, i]=density_function(dist_bg_val[:, i], means[i], sigmas[i], weights[i])
        part_L_sg[:, i]=density_function(dist_sg_val[:, i], means[i], sigmas[i], weights[i])
        bg_L-=part_L_bg[:, i]
        sg_L-=part_L_sg[:, i]
        tr_L-=part_L_tr[:, i]
    if log_likelyhood:
        tr_losses = -np.log(part_L_tr)
        bg_losses = -np.log(part_L_bg)
        sg_losses = -np.log(part_L_sg)
        bg_scores = -np.log(-bg_L)
        sg_scores = -np.log(-sg_L)
        tr_scores = -np.log(-tr_L)
        bg_scores[np.logical_not(np.isfinite(bg_scores))]=-np.inf
        sg_scores[np.logical_not(np.isfinite(sg_scores))]=-np.inf
        tr_scores[np.logical_not(np.isfinite(tr_scores))]=-np.inf
        max_score=max(np.max(tr_scores), np.max(sg_scores), np.max(bg_scores))
        #replacing inf with max*1.1 has no effect on the tagging performance (inf was 0 befor -log)
        bg_scores[np.logical_not(np.isfinite(bg_scores))]=max_score*1.1
        sg_scores[np.logical_not(np.isfinite(sg_scores))]=max_score*1.1
        tr_scores[np.logical_not(np.isfinite(tr_scores))]=max_score*1.1

        bg_losses[np.logical_not(np.isfinite(bg_losses))]=-np.inf
        sg_losses[np.logical_not(np.isfinite(sg_losses))]=-np.inf
        tr_losses[np.logical_not(np.isfinite(tr_losses))]=-np.inf
        max_losses=max(np.max(tr_losses), np.max(sg_losses), np.max(bg_losses))
        #replacing inf with max*1.1 has no effect on the tagging performance (inf was 0 befor -log)
        bg_losses[np.logical_not(np.isfinite(bg_losses))]=max_losses*1.1
        sg_losses[np.logical_not(np.isfinite(sg_losses))]=max_losses*1.1
        tr_losses[np.logical_not(np.isfinite(tr_losses))]=max_losses*1.1
    else:
        tr_losses = part_L_tr
        bg_losses = part_L_bg
        sg_losses = part_L_sg
        bg_scores = bg_L
        sg_scores = sg_L
        tr_scores = tr_L
    return tr_scores, bg_scores, sg_scores, tr_losses, bg_losses, sg_losses

def train_k_means(tra_data_path, con_data_path, pref, k, SIGMA, crop, cont, preproc, Id, train_mode, data=None, MINI_BATCH=False):
    if data is None:
        X_tr=prepare_data(tra_data_path, crop=crop, preproc=preproc, SIGMA=SIGMA)
        if cont>0:
            X_cont=prepare_data(con_data_path, crop=cont, preproc=preproc, SIGMA=SIGMA)
            X_tr=np.concatenate((X_tr, X_cont))
    else:
        X_tr=data
    if train_mode=="s":
        if MINI_BATCH:
            kmeans=MiniBatchKMeans(n_clusters=k, random_state=Id, n_init=1, max_iter=10, init="random").fit(X_tr)
        else:
            kmeans=KMeans(n_clusters=k, random_state=Id, n_init=1, max_iter=10, init="random").fit(X_tr)
    if train_mode=="f":
        if MINI_BATCH:
            kmeans=MiniBatchKMeans(
                n_clusters=k, 
                random_state=Id, 
                n_init=1).fit(X_tr)
        else:
            kmeans=KMeans(n_clusters=k, random_state=Id, n_init=1).fit(X_tr)
    if train_mode=="d":
        if MINI_BATCH:
            kmeans=MiniBatchKMeans(n_clusters=k, random_state=Id).fit(X_tr)
        else:
            kmeans=KMeans(n_clusters=k, random_state=Id).fit(X_tr)
    if train_mode=="med":
        kmeans=KMedoids(n_clusters=k, random_state=Id).fit(X_tr)  
    model_path="models/{:}m{:}s{:}c{:}r{:}KI{:}{:}.pickle".format(pref, k, SIGMA, crop//1000, reprocessing.reproc_names(preproc), train_mode, Id)
    pickle.dump(kmeans, open(model_path, "wb"))
    if not (train_mode in ["d", "f", "s"]):
        print("invalid train mode!")
    return X_tr, kmeans
    
def train_or_load_k_means(tra_data_path, con_data_path, pref, k, SIGMA, crop, cont, preproc, Id, train_mode, data=None, return_data=False, MINI_BATCH=False):
    model_path="models/{:}m{:}s{:}c{:}r{:}KI{:}{:}.pickle".format(pref, k, SIGMA, crop//1000, reprocessing.reproc_names(preproc), train_mode, Id)
    if os.path.isfile(model_path):
        print("loading trained model", model_path)
        kmeans=pickle.load(open(model_path, "rb"))
        if data is None:
            X_tr=prepare_data(tra_data_path, crop=crop, preproc=preproc, SIGMA=SIGMA)
            if cont>0:
                X_cont=prepare_data(con_data_path, crop=cont, preproc=preproc, SIGMA=SIGMA)
                X_tr=np.concatenate((X_tr, X_cont))
            return X_tr, kmeans
        else:
            return data, kmeans
    else:
        print("training a new model", model_path)
        X_tr, kmeans = train_k_means(tra_data_path, con_data_path, pref, k, SIGMA, crop, cont, preproc, Id, train_mode=train_mode, data=data, MINI_BATCH=MINI_BATCH)
        return X_tr, kmeans

def k_means_process(dataset=1,
                    n_clusters=10,
                    SIGMA=3,
                    crop=10000,
                    cont=0,
                    preproc=None,
                    Id=0,
                    knc=1, 
                    smear=1,                                
                    characterise=False,
                    SAVE_CHAR=True,
                    REVERSE=False,
                    DO_TSNE=True,
                    DO_TSNE_CENTROIDS=False, 
                    train_mode="d",#explanation below
                    data=None,#training data that was alredy loaded and preprocessed by the prevous iteration of the algorythm and so can be reused
                    full_mean_diffs=False,
                    non_smeared_mean=False,
                    TSNE_scores=True,
                    SCORE_TYPE="KNC",
                    MINI_BATCH=False,
                    return_scores=False,
                    return_data=True, 
                    images=True):
# To train_mode
# we are not so interested in the clustering itself at this point thus we dont
# really require the best and fully convergent clustering, repeating clustering
# with 10 initialisations is thus a waist of resources (it may be better to
# build an ensemble out of such 10 instead of picking one of 10 with best clustering)
# "d" - default as it is default in scikit
# "f" - fast (train only with one initialisation)
# "s" - stochastic train with only 1 initialisation for only 10 steps and random initialisation
    plt.close("all")
    random.seed(a=10, version=2)
    
    pref, pref2, tra_data_path, con_data_path, bg_val_data_path, sg_val_data_path = dataset_path_and_pref(dataset, REVERSE)
    
    X_tr=prepare_data(tra_data_path, crop=crop, preproc=preproc, SIGMA=SIGMA)
    print("starting with", len(X_tr))
    kde = KernelDensity(kernel='gaussian', bandwidth=0.005).fit(X_tr)
    print("fit done")
    tr_scores = -kde.score_samples(X_tr)
    print("training points scores sampled")
    
    if return_scores or characterise:
        X_bg_val=prepare_data(bg_val_data_path, crop=2000, preproc=preproc, SIGMA=SIGMA)
        X_sg_val=prepare_data(sg_val_data_path, crop=2000, preproc=preproc, SIGMA=SIGMA)
        
        sg_scores=-kde.score_samples(X_sg_val)
        print("sg scores sampled")
        bg_scores=-kde.score_samples(X_bg_val)
        print("bg scores sampled")
        
    if characterise:
        #%%
        if False:
            max_score=np.quantile(np.concatenate([tr_scores, sg_scores]), 0.99)#max(np.max(tr_scores), np.max(sg_scores))
            min_score=np.quantile(np.concatenate([tr_scores, sg_scores]), 0.01)#min(np.min(tr_scores), np.min(sg_scores))
            
            tr_scores_nrm=tr_scores#-min_score)/(max_score-min_score)
            sg_scores_nrm=sg_scores#-min_score)/(max_score-min_score)
            
            
            n_sig=1000
            n_bg=1000
            random.seed(a=10, version=2)
            IDs_TSNE=np.random.randint(0, X_tr.shape[0]-1, n_bg, )
            IDs_TSNE_sig=np.random.randint(0, X_sg_val.shape[0]-1, n_sig, )
            
            if X_tr.shape[1]==2:
                Y =np.concatenate([X_tr[IDs_TSNE], X_sg_val[IDs_TSNE_sig]])
            else:
                Y = TSNE(n_components=2, n_iter=1000, random_state=0).fit_transform(np.concatenate([X_tr[IDs_TSNE], X_sg_val[IDs_TSNE_sig]]))
            plt.figure(figsize=(10, 10))

            labels_TSNE=np.concatenate((np.ones(n_bg), -1*np.ones(n_sig), -2*np.ones(n_clusters)))
            plt.legend()
            if X_tr.shape[1]==2:
                    ax=plt.gca()
                    ax.axis('equal')
            
            if TSNE_scores:
                plt.figure(figsize=(12, 3))
                plt.scatter(Y[labels_TSNE>=0, 0], Y[labels_TSNE>=0, 1], c=tr_scores_nrm[IDs_TSNE],  cmap="turbo", marker="o", vmin=min_score, vmax=max_score, s=4)
                plt.colorbar()
                plt.scatter(Y[labels_TSNE==-1, 0], Y[labels_TSNE==-1, 1], c=sg_scores_nrm[IDs_TSNE_sig],  marker="x", cmap="turbo", vmin=min_score, vmax=max_score, s=4)
                if X_tr.shape[1]==2:
                    ax=plt.gca()
                    ax.axis('equal')
                    plt.xlim((-30, 30))
                    plt.ylim((-5, 10))
                if DO_TSNE_CENTROIDS:
                    plt.scatter(Y[labels_TSNE==-2, 0], Y[labels_TSNE==-2, 1], label="centroids", marker="X", color="black", s=90)
                plt.legend()

                
                plt.figure(figsize=(12, 10))
                plt.scatter(Y[labels_TSNE>=0, 0], Y[labels_TSNE>=0, 1], c=tr_scores_nrm[IDs_TSNE], label="training", cmap="turbo", marker="o", vmin=min_score, vmax=max_score)
                if DO_TSNE_CENTROIDS:
                    plt.scatter(Y[labels_TSNE==-2, 0], Y[labels_TSNE==-2, 1], label="centroids", marker="X", color="black", s=90)
                plt.legend()
                
                plt.figure(figsize=(12, 10))
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
        plt.legend(title=f'AUC: {auc:.3f}')
        
        plt.figure()
        plt.grid()
        sic_max=np.max(np.nan_to_num(tpr/fpr**0.5)[fpr!=0]) #some
        plt.plot(tpr, tpr/fpr**0.5, label="max {0:.2f}".format(sic_max))
        plt.legend()
                
        #%% Save results:
        if SAVE_CHAR:
            path="char/kde"
            res={}
            res["fpr"]=fpr
            res["tps"]=tpr
            res["AUC"]=auc
            pickle.dump(res, open(path+"res.pickle", "wb"))
            print(path+"res.pickle")
    if return_scores:
        if return_data:
            return X_tr, tr_scores, bg_scores, sg_scores
        else:
            return tr_scores, bg_scores, sg_scores
    else:
        if return_data:
            return X_tr
    
if __name__ == "__main__":
    # possible_switches=[1, 2]
    switches=[0]
    
    if 0 in switches:
        k_means_process(dataset=4,
                    n_clusters=10,
                    SIGMA=3,
                    crop=5000,
                    preproc=None,
                    Id=0,
                    knc=1,                                 
                    characterise=True,
                    SAVE_CHAR=True,
                    REVERSE=False,
                    DO_TSNE=True,
                    DO_TSNE_CENTROIDS=True, 
                    train_mode="d", 
                    data=None)

