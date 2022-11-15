from sklearn.cluster import KMeans
import numpy as np
import random 
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.ndimage import gaussian_filter
import pickle
import reprocessing
from sklearn.metrics import roc_auc_score, roc_curve
import os
from dataset_path_and_pref import dataset_path_and_pref, prepare_data

plt.close("all")
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

def gaussian(x, mean, sigma, weight):
    return weight/(sigma*(2*np.pi)**0.5)*np.exp(-(x-mean)**2/(2*sigma**2))

def half_gaussian(x, mean_, sigma_, weight, smear=1):
    mean=mean_#-sigma_*(smear-1)
    sigma=sigma_*smear
    out = weight/(sigma*(2*np.pi)**0.5)*np.exp(-(x-mean)**2/(2*sigma**2))
    out[x<mean]=np.max(out)
    return out

def sum_n_mins(losses, knc):
    losses_cop=np.copy(losses)
    losses_cop.sort(1)
    return np.mean(losses_cop[:, :knc], 1)

def train_k_means(tra_data_path, pref, k, SIGMA, crop, preproc, Id, train_mode, data=None):
    if data is None:
        X_bg=prepare_data(tra_data_path, crop=crop, preproc=preproc, SIGMA=SIGMA)
    else:
        X_bg=data
    if train_mode=="s":
        kmeans=KMeans(n_clusters=k, random_state=Id, n_init=1, max_iter=10, init="random").fit(X_bg)
        pickle.dump(kmeans, open("models/{:}m{:}s{:}c{:}r{:}KIs{:}.pickle".format(pref, k, SIGMA, crop//1000, reprocessing.reproc_names(preproc), Id), "wb"))
    if train_mode=="f":
        kmeans=KMeans(n_clusters=k, random_state=Id, n_init=1).fit(X_bg)
        pickle.dump(kmeans, open("models/{:}m{:}s{:}c{:}r{:}KIf{:}.pickle".format(pref, k, SIGMA, crop//1000, reprocessing.reproc_names(preproc), Id), "wb"))
    if train_mode=="d":
        kmeans=KMeans(n_clusters=k, random_state=Id).fit(X_bg)
        pickle.dump(kmeans, open("models/{:}m{:}s{:}c{:}r{:}KId{:}.pickle".format(pref, k, SIGMA, crop//1000, reprocessing.reproc_names(preproc), Id), "wb"))
    if not (train_mode in ["d", "f", "s"]):
        print("invalid train mode!")
    return X_bg, kmeans
    
def train_or_laod_k_means(tra_data_path, pref, k, SIGMA, crop, preproc, Id, train_mode, data=None, return_data=False):
    model_path="models/{:}m{:}s{:}c{:}r{:}KI{:}{:}.pickle".format(pref, k, SIGMA, crop//1000, reprocessing.reproc_names(preproc), train_mode, Id)
    if os.path.isfile(model_path):
        print("loading trained model", model_path)
        kmeans=pickle.load(open(model_path, "rb"))
        if data is None:
            X_bg=prepare_data(tra_data_path, crop=crop, preproc=preproc, SIGMA=SIGMA)
            return X_bg, kmeans
        else:
            return data, kmeans
    else:
        print("training a new model", model_path)
        X_bg, kmeans = train_k_means(tra_data_path, pref, k, SIGMA, crop, preproc, Id, train_mode=train_mode, data=data)
        return X_bg, kmeans

def k_means_process(dataset=1,
                    n_clusters=10,
                    SIGMA=3,
                    crop=10000,
                    preproc=None,
                    Id=0,
                    knc=1, 
                    smear=1,                                
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
                    SCORE_TYPE="KNC"): 
# To train_mode
# we are not so interested in the clustering itself at this point thus we dont 
# really require the best and fully convergent clustering, repeating clustering 
# with 10 initialisations is thus a waist of resources (it may be better to 
# build an ensemble out of such 10 instead of picking one of 10 with best clustering) 
# "d" - default as it is default in scikit
# "f" - fast (train only with one initialisation)
# "s" - stochastic train with only 1 initialisation for only 10 steps and random initialisation 
 
    k=n_clusters
    random.seed(a=10, version=2)
    
    pref, pref2, tra_data_path, con_data_path, bg_val_data_path, sg_val_data_path = dataset_path_and_pref(dataset, REVERSE)
    
    hyp=(pref, k, SIGMA, crop, preproc, Id)
    
    X_bg, kmeans = train_or_laod_k_means(tra_data_path, *hyp, train_mode=train_mode, data=data)

    if characterise:
        X_bg_val=prepare_data(bg_val_data_path, preproc=preproc, SIGMA=SIGMA)
        X_sg_val=prepare_data(sg_val_data_path, preproc=preproc, SIGMA=SIGMA)
 
        
        if SCORE_TYPE=="MinD": #minimal distance
            postf="MinD"
            tr_losses = kmeans.transform(X_bg)
            bg_losses = kmeans.transform(X_bg_val)
            sg_losses = kmeans.transform(X_sg_val)
            bg_scores = np.min(bg_losses, 1)
            sg_scores = np.min(sg_losses, 1)
            tr_scores = np.min(tr_losses, 1) 
    
        if SCORE_TYPE=="KNC":
            postf="KNC"+str(knc)
            tr_losses = kmeans.transform(X_bg)
            bg_losses = kmeans.transform(X_bg_val)
            sg_losses = kmeans.transform(X_sg_val)
            bg_scores = sum_n_mins(bg_losses, knc)
            sg_scores = sum_n_mins(sg_losses, knc)
            tr_scores = sum_n_mins(tr_losses, knc)

        elif SCORE_TYPE=="logLmc":
            postf="logLmc"
            dist_tr = kmeans.transform(X_bg)
            dist_bg_val = kmeans.transform(X_bg_val)
            dist_sg_val = kmeans.transform(X_sg_val)
            sigma_0s=[]
            weights=[]
            for i in range(k):
                dist=dist_tr[kmeans.labels_==i, i]
                sigma_0s.append(np.sqrt(np.mean(dist**2)))
                weights.append(len(dist)/crop)
            part_L_bg=np.zeros(dist_bg_val.shape)
            part_L_sg=np.zeros(dist_sg_val.shape)
            part_L_tr=np.zeros(dist_tr.shape)
            bg_L, sg_L, tr_L=0, 0, 0
            for i in range(k):
                part_L_tr[:, i]=gaussian_mult_c(dist_tr[:, i], sigma_0s[i], weights[i])
                part_L_bg[:, i]=gaussian_mult_c(dist_bg_val[:, i], sigma_0s[i], weights[i])
                part_L_sg[:, i]=gaussian_mult_c(dist_sg_val[:, i], sigma_0s[i], weights[i])
                bg_L-=part_L_bg[:, i]
                sg_L-=part_L_sg[:, i]
                tr_L-=part_L_tr[:, i]
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

        elif SCORE_TYPE=="logLm":
            postf="logLm"
            dist_tr = kmeans.transform(X_bg)
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
            part_L_bg=np.zeros(dist_bg_val.shape)
            part_L_sg=np.zeros(dist_sg_val.shape)
            part_L_tr=np.zeros(dist_tr.shape)
            bg_L, sg_L, tr_L=0, 0, 0
            for i in range(k):
                part_L_tr[:, i]=gaussian_mult(dist_tr[:, i], means[i], sigmas[i], weights[i])
                part_L_bg[:, i]=gaussian_mult(dist_bg_val[:, i], means[i], sigmas[i], weights[i])
                part_L_sg[:, i]=gaussian_mult(dist_sg_val[:, i], means[i], sigmas[i], weights[i])
                bg_L-=part_L_bg[:, i]
                sg_L-=part_L_sg[:, i]
                tr_L-=part_L_tr[:, i]
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
        
        elif SCORE_TYPE=="Lrm":
            postf="Lrm"
            dist_tr = kmeans.transform(X_bg)
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
            part_L_bg=np.zeros(dist_bg_val.shape)
            part_L_sg=np.zeros(dist_sg_val.shape)
            part_L_tr=np.zeros(dist_tr.shape)
            bg_L, sg_L, tr_L=0, 0, 0
            for i in range(k):
                part_L_tr[:, i]=gaussian_mult(dist_tr[:, i], means[i], sigmas[i], weights[i])
                part_L_bg[:, i]=gaussian_mult(dist_bg_val[:, i], means[i], sigmas[i], weights[i])
                part_L_sg[:, i]=gaussian_mult(dist_sg_val[:, i], means[i], sigmas[i], weights[i])
                bg_L-=part_L_bg[:, i]
                sg_L-=part_L_sg[:, i]
                tr_L-=part_L_tr[:, i]
            tr_losses = part_L_tr
            bg_losses = part_L_bg
            sg_losses = part_L_sg
            bg_scores = bg_L
            sg_scores = sg_L
            tr_scores = tr_L
        
        elif SCORE_TYPE=="logLd":
            postf="logLd"
            dist_tr = kmeans.transform(X_bg)
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
            part_L_bg=np.zeros(dist_bg_val.shape)
            part_L_sg=np.zeros(dist_sg_val.shape)
            part_L_tr=np.zeros(dist_tr.shape)
            bg_L, sg_L, tr_L=0, 0, 0
            for i in range(k):
                part_L_tr[:, i]=np.exp(-dist_tr[:, i])*weights[i]
                part_L_bg[:, i]=np.exp(-dist_bg_val[:, i])*weights[i]
                part_L_sg[:, i]=np.exp(-dist_sg_val[:, i])*weights[i]
                bg_L-=part_L_bg[:, i]
                sg_L-=part_L_sg[:, i]
                tr_L-=part_L_tr[:, i]
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
            
        elif SCORE_TYPE=="Lr":
            postf="Lr"
            dist_tr = kmeans.transform(X_bg)
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
            part_L_bg=np.zeros(dist_bg_val.shape)
            part_L_sg=np.zeros(dist_sg_val.shape)
            part_L_tr=np.zeros(dist_tr.shape)
            bg_L, sg_L, tr_L=0, 0, 0
            for i in range(k):
                part_L_tr[:, i]=gaussian(dist_tr[:, i], means[i], sigmas[i], weights[i])
                part_L_bg[:, i]=gaussian(dist_bg_val[:, i], means[i], sigmas[i], weights[i])
                part_L_sg[:, i]=gaussian(dist_sg_val[:, i], means[i], sigmas[i], weights[i])
                bg_L-=part_L_bg[:, i]
                sg_L-=part_L_sg[:, i]
                tr_L-=part_L_tr[:, i]
            tr_losses = part_L_tr
            bg_losses = part_L_bg
            sg_losses = part_L_sg
            bg_scores = bg_L
            sg_scores = sg_L
            tr_scores = tr_L
            
        elif SCORE_TYPE=="logLr":
            postf="logLr"
            dist_tr = kmeans.transform(X_bg)
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
            part_L_bg=np.zeros(dist_bg_val.shape)
            part_L_sg=np.zeros(dist_sg_val.shape)
            part_L_tr=np.zeros(dist_tr.shape)
            bg_L, sg_L, tr_L=0, 0, 0
            for i in range(k):
                part_L_tr[:, i]=gaussian(dist_tr[:, i], means[i], sigmas[i], weights[i])
                part_L_bg[:, i]=gaussian(dist_bg_val[:, i], means[i], sigmas[i], weights[i])
                part_L_sg[:, i]=gaussian(dist_sg_val[:, i], means[i], sigmas[i], weights[i])
                bg_L-=part_L_bg[:, i]
                sg_L-=part_L_sg[:, i]
                tr_L-=part_L_tr[:, i]
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
            
        elif SCORE_TYPE=="Lrh":
            postf="Lrh"
            dist_tr = kmeans.transform(X_bg)
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
            part_L_bg=np.zeros(dist_bg_val.shape)
            part_L_sg=np.zeros(dist_sg_val.shape)
            part_L_tr=np.zeros(dist_tr.shape)
            bg_L, sg_L, tr_L=0, 0, 0
            for i in range(k):
                part_L_tr[:, i]=half_gaussian(dist_tr[:, i], means[i], sigmas[i], weights[i])
                part_L_bg[:, i]=half_gaussian(dist_bg_val[:, i], means[i], sigmas[i], weights[i])
                part_L_sg[:, i]=half_gaussian(dist_sg_val[:, i], means[i], sigmas[i], weights[i])
                bg_L-=part_L_bg[:, i]
                sg_L-=part_L_sg[:, i]
                tr_L-=part_L_tr[:, i]
            tr_losses = part_L_tr
            bg_losses = part_L_bg
            sg_losses = part_L_sg
            bg_scores = bg_L
            sg_scores = sg_L
            tr_scores = tr_L
            
        elif SCORE_TYPE=="logLrh":
            postf="logLrh"
            dist_tr = kmeans.transform(X_bg)
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
            part_L_bg=np.zeros(dist_bg_val.shape)
            part_L_sg=np.zeros(dist_sg_val.shape)
            part_L_tr=np.zeros(dist_tr.shape)
            bg_L, sg_L, tr_L=0, 0, 0
            for i in range(k):
                print(means[i])
                part_L_tr[:, i]=half_gaussian(dist_tr[:, i], means[i], sigmas[i], weights[i], smear=smear)
                part_L_bg[:, i]=half_gaussian(dist_bg_val[:, i], means[i], sigmas[i], weights[i], smear=smear)
                part_L_sg[:, i]=half_gaussian(dist_sg_val[:, i], means[i], sigmas[i], weights[i], smear=smear)
                bg_L-=part_L_bg[:, i]
                sg_L-=part_L_sg[:, i]
                tr_L-=part_L_tr[:, i]
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
        
        #%%
        if DO_TSNE:
            max_score=np.quantile(np.concatenate([tr_scores, sg_scores]), 0.99)#max(np.max(tr_scores), np.max(sg_scores))
            min_score=np.quantile(np.concatenate([tr_scores, sg_scores]), 0.01)#min(np.min(tr_scores), np.min(sg_scores))
            
            tr_scores_nrm=tr_scores#-min_score)/(max_score-min_score)
            sg_scores_nrm=sg_scores#-min_score)/(max_score-min_score)
            
            n_sig=1000
            n_bg=1000
            random.seed(a=10, version=2)
            IDs_TSNE=np.random.randint(0, X_bg.shape[0]-1, n_bg, )
            IDs_TSNE_sig=np.random.randint(0, X_sg_val.shape[0]-1, n_sig, )
            if DO_TSNE_CENTROIDS:
                centoids=kmeans.cluster_centers_
                labels_TSNE=np.concatenate((kmeans.labels_[IDs_TSNE], -1*np.ones(n_sig), -2*np.ones(n_clusters)))
                Y = TSNE(n_components=2, n_iter=1000, random_state=0).fit_transform(np.concatenate([X_bg[IDs_TSNE], X_sg_val[IDs_TSNE_sig], centoids]))
            else:
                labels_TSNE=np.concatenate((kmeans.labels_[IDs_TSNE], -1*np.ones(n_sig)))
                Y = TSNE(n_components=2, n_iter=1000, random_state=0).fit_transform(np.concatenate([X_bg[IDs_TSNE], X_sg_val[IDs_TSNE_sig]]))
            plt.figure(figsize=(10, 10))
            u_labels = np.unique(kmeans.labels_)
            for i in u_labels:
                if n_clusters<=10:
                    plt.scatter(Y[labels_TSNE==i, 0], Y[labels_TSNE==i, 1], label=i+1)
                else:
                    plt.scatter(Y[labels_TSNE==i, 0], Y[labels_TSNE==i, 1])
            plt.scatter(Y[labels_TSNE==-1, 0], Y[labels_TSNE==-1, 1], label="signal", marker="x")
            if DO_TSNE_CENTROIDS:
                plt.scatter(Y[labels_TSNE==-2, 0], Y[labels_TSNE==-2, 1], label="centroids", marker="X", color="black", s=70)
            plt.legend()
            
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
        
        plt.figure(figsize=(10, 10))
        #%%
        _, bins, _, = plt.hist(bg_scores, histtype='step', label='bg', bins=40, density=True)
        plt.hist(sg_scores, histtype='step', label='sig', bins=bins, density=True)
        
        labels=np.concatenate((np.zeros(len(X_bg_val)), np.ones(len(X_sg_val))))
        auc = roc_auc_score(labels, np.append(bg_scores, sg_scores))
        plt.legend(title=f'AUC: {auc:.3f}')
        fpr , tpr , thresholds = roc_curve(labels, np.append(bg_scores, sg_scores))
        plt.figure()
        plt.grid()
        plt.plot(tpr, 1/fpr)
        plt.ylim(ymin=1, ymax=1000)
        plt.plot(np.linspace(0, 1, 1000), 1/np.linspace(0, 1, 1000), color="gray")
        plt.yscale("log")
        plt.legend(title=f'AUC: {auc:.3f}')
        
        plt.figure()
        plt.grid()
        sic_max=np.max(np.nan_to_num(tpr/fpr**0.5)[fpr!=0]) #some
        plt.plot(tpr, tpr/fpr**0.5, label="max {0:.2f}".format(sic_max))
        plt.legend()
        
        counts=np.array([np.sum(kmeans.labels_==i) for i in range(n_clusters)])
        counts.sort()
        print(counts)
        
        #%% Cluster populations:
        unique, counts_train = np.unique(kmeans.labels_, return_counts=True)
        print("labels :", unique)
        print("bg trai:", counts)
        
        test_bg_labels=bg_losses.argmin(1)
        test_sg_labels=sg_losses.argmin(1)
        bg_min=bg_losses.min(1)
        sg_min=sg_losses.min(1)
        
        cluster_counts_bg=np.bincount(test_bg_labels)
        cluster_counts_sg=np.bincount(test_sg_labels)
        if len(cluster_counts_bg)<n_clusters:
            cluster_counts_bg=np.concatenate((cluster_counts_bg, np.zeros(n_clusters-len(cluster_counts_bg), dtype=int)))
        if len(cluster_counts_sg)<n_clusters:
            cluster_counts_sg=np.concatenate((cluster_counts_bg, np.zeros(n_clusters-len(cluster_counts_sg), dtype=int)))
        print("bg test:", cluster_counts_bg)
        print("sg test:", cluster_counts_sg)
        
        cols=3+full_mean_diffs+non_smeared_mean*2
        num=min(10, n_clusters)
        fig, ax = plt.subplots(num, cols, figsize=(cols*4, num*4.2), squeeze=False)
        max_dist=max(np.max(bg_losses), np.max(sg_losses))
        max_min_dist=max(np.max(bg_min), np.max(sg_min))
        min_dist=min(0, np.min(bg_losses), np.min(sg_losses))
        min_min_dist=min(0, np.min(bg_min), np.min(sg_min))
        bins=np.linspace(min_dist, max_dist, 40)
        bins2=np.linspace(min_min_dist, max_min_dist, 40)
        
        X_bg_val_no_sm=prepare_data(bg_val_data_path, preproc=preproc, SIGMA=0)
        for i in range(num):
            #mean image
            plt.sca(ax[i][2])
            plt.xticks([])
            plt.yticks([])
            plt.imshow(kmeans.cluster_centers_[i].reshape((40, 40)))
            #histogram of distances
            plt.sca(ax[i][0])
            plt.yticks([])
            plt.hist(bg_losses[:, i], bins=bins, histtype='step')
            plt.hist(sg_losses[:, i], bins=bins, histtype='step')
            plt.hist(bg_min[test_bg_labels==i], bins=bins, histtype='step')
            plt.hist(sg_min[test_sg_labels==i], bins=bins, histtype='step')
            if i<num-1:
                plt.xticks([])
            plt.sca(ax[i][1])
            plt.yticks([])
            plt.title("tr"+str(counts_train[i])+" bg"+str(cluster_counts_bg[i])+" sg"+str(cluster_counts_sg[i]))
            plt.hist(bg_min[test_bg_labels==i], bins=bins2, histtype='step')
            plt.hist(sg_min[test_sg_labels==i], bins=bins2, histtype='step')
            if i<num-1:
                plt.xticks([])
            
            curr_col=2
            if non_smeared_mean:
                curr_col+=1
                plt.sca(ax[i][curr_col])
                plt.xticks([])
                plt.yticks([])
                mat=kmeans.cluster_centers_[i].reshape((40, 40))-np.mean(X_bg_val, 0).reshape((40, 40))
                max_mat=max(np.max(mat), -np.min(mat))
                plt.imshow(mat, vmin=-max_mat, vmax=max_mat, cmap="bwr")
            
            if full_mean_diffs:
                curr_col+=1
                plt.sca(ax[i][curr_col])
                plt.xticks([])
                plt.yticks([])
                mat=np.mean(X_bg_val_no_sm[kmeans.predict(X_bg_val)==i], 0).reshape((40, 40))
                plt.imshow(mat)
                curr_col+=1
                plt.sca(ax[i][curr_col])
                plt.xticks([])
                plt.yticks([])
                mat=np.mean(X_bg_val_no_sm[kmeans.predict(X_bg_val)==i], 0).reshape((40, 40))-np.mean(X_bg_val_no_sm, 0).reshape((40, 40))
                max_mat=max(np.max(mat), -np.min(mat))
                plt.imshow(mat, vmin=-max_mat, vmax=max_mat, cmap="bwr")
                plt.colorbar()
                
        #%% Save results:
        if SAVE_CHAR:
            path="char/{:}+{:}m{:}s{:}c{:}r{:}KI{:}{:}{:}/".format(pref, pref2, k, SIGMA, crop//1000, reprocessing.reproc_names(preproc), train_mode, Id, postf)
            os.makedirs(path, exist_ok=True)
            k=0
            if DO_TSNE:
                k+=1
                plt.figure(k)
                plt.savefig(path+"TSNE.png", bbox_inches="tight")
                if TSNE_scores:
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
            plt.savefig(path+"dist.png", bbox_inches="tight")
            k+=1
            plt.figure(k)
            plt.savefig(path+"ROC.png", bbox_inches="tight")
            k+=1
            plt.figure(k)
            plt.savefig(path+"SIC.png", bbox_inches="tight")
            k+=1
            plt.figure(k)
            plt.savefig(path+"clusters.png", bbox_inches="tight")
            
            res={}
            res["fpr"]=fpr
            res["tps"]=tpr
            res["AUC"]=auc
            pickle.dump(res, open(path+"res.pickle", "wb"))
            
    return X_bg
    
if __name__ == "__main__":
    # possible_switches=[1, 2]
    switches=[13]
    
    if 0 in switches:
        k_means_process(dataset=1,
                    n_clusters=10,
                    SIGMA=3,
                    crop=10000,
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
    
    if 1 in switches: #results for Topic 13 and 13.1
        k_means_process(dataset=2, characterise=True, REVERSE=True, n_clusters=10, preproc=reprocessing.reproc_4rt, knc=1, 
                        full_mean_diffs=True,
                        non_smeared_mean=True)
        
    if 2 in switches: #Topic 15: HD dataset
        k_arr=[1, 2, 4, 8, 16, 32, 64, 128, 256]
        data=None
        for k in k_arr:
            data = k_means_process(n_clusters=k, dataset=3, train_mode="d", data=data)
            
    if 3 in switches:
        k_arr=[1, 2, 4, 8, 16, 32, 64, 128, 256]
        preproc_arr=[reprocessing.reproc_none, reprocessing.reproc_sqrt, reprocessing.reproc_4rt, reprocessing.reproc_log1000, reprocessing.reproc_heavi]
        
        for preproc in preproc_arr:
            for k in k_arr:
                k_means_process(n_clusters=k, dataset=1, train_mode="f", preproc=preproc)
    
    if 4 in switches:
        k_arr=[1, 2, 4, 8, 16, 32, 64, 128, 256]
        Ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        data=None
        for Id in Ids:
            for k in k_arr:
                data = k_means_process(n_clusters=k, dataset=1, train_mode="f", Id=Id, data=data) 
    if 5 in switches:
        k_arr=[1, 2, 4, 8, 16, 32, 64, 128, 256]
        Ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        data=None
        for Id in Ids:
            for k in k_arr:
                data = k_means_process(n_clusters=k, dataset=1, train_mode="s", Id=Id, data=data)
    if 6 in switches:
        k_arr=[1, 2, 4, 8, 16, 32, 64, 128]
        Ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        data=None
        for k in k_arr:
            data = k_means_process(n_clusters=k, dataset=1, train_mode="d", Id=0, data=data, REVERSE=True)
            for Id in Ids:
                data = k_means_process(n_clusters=k, dataset=1, train_mode="f", Id=Id, data=data, REVERSE=True)
                data = k_means_process(n_clusters=k, dataset=1, train_mode="s", Id=Id, data=data, REVERSE=True)
    if 7 in switches:
        k_arr=[1, 2, 4, 8, 16, 32, 64, 128]
        Ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        data=None
        for k in k_arr:
            data = k_means_process(n_clusters=k, dataset=2, train_mode="d", Id=0, data=data, preproc=reprocessing.reproc_4rt)
            for Id in Ids:
                data = k_means_process(n_clusters=k, dataset=2, train_mode="f", Id=Id, data=data, preproc=reprocessing.reproc_4rt)
                data = k_means_process(n_clusters=k, dataset=2, train_mode="s", Id=Id, data=data, preproc=reprocessing.reproc_4rt)

    if 8 in switches: #Topic 15.2
        SIGMAS=[0, 1, 3, 5]
        preproc_arr=[reprocessing.reproc_none, reprocessing.reproc_sqrt, reprocessing.reproc_4rt, reprocessing.reproc_log1000, reprocessing.reproc_heavi]
        
        for preproc in preproc_arr:
            for SIGMA in SIGMAS:
                k_means_process(dataset=3, n_clusters=32, preproc=preproc, SIGMA=SIGMA)
                k_means_process(dataset=4, n_clusters=32, preproc=preproc, SIGMA=SIGMA)
                

    if 9 in switches: #Topic 15.2
        k_arr=[1, 2, 4, 8, 16, 32, 64, 128, 256]
        data=None
        for k in k_arr:
            data = k_means_process(n_clusters=k, dataset=3, preproc=reprocessing.reproc_sqrt, data=data)
        data=None
        for k in k_arr:
            data = k_means_process(n_clusters=k, dataset=3, preproc=reprocessing.reproc_4rt, data=data)
    
    if 10 in switches: #To topic 15.1 min AUC anomaly
        k_means_process(dataset=2,
                        n_clusters=10,
                        SIGMA=3,
                        crop=10000,
                        preproc=None,
                        Id=0,
                        knc=1,                                 
                        characterise=True,
                        SAVE_CHAR=True,
                        REVERSE=True,
                        DO_TSNE=True,
                        DO_TSNE_CENTROIDS=False, 
                        train_mode="d", 
                        data=None)
        
    if 11 in switches:
        k_arr=[1, 2, 4, 8, 16, 32, 64, 128, 256]
        data=None
        for k in k_arr:
            data = k_means_process(n_clusters=k, dataset=1, data=data)
            
    if 12 in switches:
        k_arr=[1, 2, 4, 8, 16, 32, 64, 128, 256]
        data=None
        for k in k_arr:
            data = k_means_process(n_clusters=k, dataset=1, data=data, REVERSE=True)
            
    if 13 in switches: #To topic 17
       k_means_process(dataset=1,
                        n_clusters=10,
                        SIGMA=3,
                        crop=10000,
                        preproc=None, #reprocessing.reproc_4rt,
                        Id=0,
                        knc=1,
                        characterise=True,
                        SAVE_CHAR=True,
                        REVERSE=False,
                        DO_TSNE=True,
                        DO_TSNE_CENTROIDS=True, 
                        train_mode="d", 
                        data=None,
                        smear=1,
                        SCORE_TYPE="logLr")