from sklearn.cluster import KMeans, MiniBatchKMeans
from functools import partial
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import pickle
import reprocessing
from sklearn.metrics import roc_auc_score, roc_curve
import os
from sklearn_extra.cluster import KMedoids
from dataset_path_and_pref import dataset_path_and_pref, prepare_data
from utilities import d_ball_volume, dm1_sphere_area
from likelyhood_estimation import infinity_to_min_max, likelyhood_estimation_dim_Gauss, likelyhood_estimation_dim_Uniform, likelyhood_estimation_dim_Uniform_special, likelyhood_estimation
import set_matplotlib_default
from TSNE_routine import TSNE_kmeans
from sklearn_extra.robust import RobustWeightedKMeans
from dijet_scores_combine import dijet_scores_combine_ROC_SIC
from anomalousity_histogram import anomalousity_histogram
from hist_and_av import hist_and_av
def gaussian_mult_dim(x, a, s, weight):
    return weight/((s*(2*np.pi)**0.5)**a)*np.exp(-(x)**2/(2*s**2))

def gaussian_mult_var(x, mean, sigma, weight):
    s=sigma/(2**0.5)
    a=mean**2/s**2
    return weight/((s*(2*np.pi)**0.5)**a)*np.exp(-(x)**2/(2*s**2))

def gaussian_mult_c(x, sigma_0, weight, d=1):
    return weight/((sigma_0*(2*np.pi)**0.5)**d)*np.exp(-(x)**2/(2*sigma_0**2))

def gaussian_mult(x, mean, sigma, weight, d=1):
    s=mean
    return weight/((s*(2*np.pi)**0.5)**d)*np.exp(-(x)**2/(2*s**2))

def exponential_slope_parameterless(x, mean, sigma, weight):
    return np.exp(-x)*weight

def gaussian(x, mean, sigma, weight, d=1):
    if d==1:
        return weight/(sigma*(2*np.pi)**0.5)*np.exp(-(x-mean)**2/(2*sigma**2))
    else:
        return weight/(sigma*(2*np.pi)**0.5)**d*np.exp(-np.sum((x-mean)**2, axis=1)/(2*sigma**2))

def half_gaussian(x, mean, sigma, weight, smear=1):
    sigma_=sigma*smear
    out = 1/(sigma_*(2*np.pi)**0.5)*np.exp(-(x-mean)**2/(2*sigma_**2))
    out[x<mean]=np.max(out)
    return weight*out

def half_gaussian_max_norm(x, mean, sigma, weight, smear=1):
    sigma_=sigma*smear
    out = 1/(sigma_*(2*np.pi)**0.5)*np.exp(-(x-mean)**2/(2*sigma_**2))
    out[x<mean]=np.max(out)
    out=out/np.max(out)
    return weight*out

def half_gaussian_max_norm_no_weights(x, mean, sigma, weight, smear=1):
    sigma_=sigma*smear
    out = 1/(sigma_*(2*np.pi)**0.5)*np.exp(-(x-mean)**2/(2*sigma_**2))
    out[x<mean]=np.max(out)
    out=out/np.max(out)
    return out

def half_gaussian_norm(x, mean, sigma, weight, smear=1):
    sigma_=sigma*smear
    out = 1/(sigma_*(2*np.pi)**0.5)*np.exp(-(x-mean)**2/(2*sigma_**2))
    out[x<mean]=np.max(out)
    out/=2*(0.5+1*mean*np.max(out))
    return weight*out

def ball_norm(x, mean, sigma, weight, d=1):
    mean*=2
    N=d_ball_volume(d, mean) #O((sigma/mean)^0)
    out = np.zeros(len(x))
    out[x<mean]=1
    out/=N
    return weight*out

def half_gaussian_norm_d(x, mean, sigma, weight, d=1, dont_use_weights=False):
    # the likelyhood is comnputed to the O((sigma/mean)^1)
    N_0=d_ball_volume(d, mean) #O((sigma/mean)^0)
    N_1=dm1_sphere_area(d, mean)*sigma*np.sqrt(np.pi/2) #O((sigma/mean)^1)
    N=N_0+N_1
    out = np.exp(-(x-mean)**2/(2*sigma**2))
    out[x<mean]=1
    out/=np.max(out)
    out/=N
    print(N)
    if dont_use_weights:
        return out
    else:
        return out*weight
    
def half_gaussian_norm_d_inside(x, mean, sigma, weight, d=1, dont_use_weights=False):
    # the likelyhood is comnputed to the O((sigma/mean)^1)
    N=d_ball_volume(d, mean) #O((sigma/mean)^0)
    out = np.exp(-(x-mean)**2/(2*sigma**2))
    out[x<mean]=1
    out/=np.max(out)
    out/=N
    if dont_use_weights:
        return out
    else:
        return out*weight
    
def d_slopes_norm(x, mean, sigma, weight, d=1, dont_use_weights=False, dont_use_volume=False):
    N_in=d_ball_volume(d, mean)
    N_sl=dm1_sphere_area(d, mean)*sigma*np.sqrt(np.pi/2)
    N=N_in+N_sl
    out=np.zeros(x.shape)
    out[x>=mean] =((mean/x[x>=mean])**(d-1))* np.exp(-(x[x>=mean]-mean)**2/(2*sigma**2))
    out[x<mean]=1
    out/=np.max(out)
    if dont_use_volume==False:
        out/=N
    if dont_use_weights:
        return out
    else:
        return out*weight

def sum_n_mins(losses, knc):
    losses_cop=np.copy(losses)
    losses_cop.sort(1)
    return np.mean(losses_cop[:, :knc], 1)

def train_k_means(DI, pref, k, SIGMA, crop, cont, preproc, Id, train_mode, data=None, MINI_BATCH=False):
    standard=None
    if data is None:
        X_tr, standard=prepare_data(DI["tra_data_path"], field=DI["tra_data_field"], crop=crop, preproc=preproc, SIGMA=SIGMA)
        if cont>0:
            X_cont, _ =prepare_data(DI["con_data_path"], field=DI["con_data_field"], crop=cont, preproc=preproc, SIGMA=SIGMA)
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
    if train_mode=="dl":
        if MINI_BATCH:
            kmeans=MiniBatchKMeans(n_clusters=k, random_state=Id, max_iter=10000).fit(X_tr)
        else:
            kmeans=KMeans(n_clusters=k, random_state=Id, max_iter=10000).fit(X_tr)
    if train_mode=="med":
        kmeans=KMedoids(n_clusters=k, random_state=Id).fit(X_tr)  
    if train_mode=="rob":
        kmeans=RobustWeightedKMeans(n_clusters=k, random_state=Id).fit(X_tr)  
    model_path="models/{:}m{:}s{:}c{:}r{:}KI{:}{:}.pickle".format(pref, k, SIGMA, crop//1000, reprocessing.reproc_names(preproc), train_mode, Id)
    pickle.dump(kmeans, open(model_path, "wb"))
    if not (train_mode in ["d", "f", "s"]):
        print("invalid train mode!")
    return X_tr, kmeans, standard
    
def train_or_load_k_means(DI, pref, k, SIGMA, crop, cont, preproc, Id, train_mode, data=None, return_data=False, MINI_BATCH=False):
    model_path="models/{:}m{:}s{:}c{:}r{:}KI{:}{:}.pickle".format(pref, k, SIGMA, crop//1000, reprocessing.reproc_names(preproc), train_mode, Id)
    if os.path.isfile(model_path):
        print("##loading trained model", model_path)
        kmeans=pickle.load(open(model_path, "rb"))
        if data is None:
            X_tr, standard=prepare_data(DI["tra_data_path"], field=DI["tra_data_field"], crop=crop, preproc=preproc, SIGMA=SIGMA)
            if cont>0:
                X_cont, _ =prepare_data(DI["con_data_path"], field=DI["con_data_field"], crop=cont, preproc=preproc, SIGMA=SIGMA)
                X_tr=np.concatenate((X_tr, X_cont))
            return X_tr, kmeans, standard
        else:
            return data, kmeans, None
    else:
        print("##training a new model", model_path)
        X_tr, kmeans, standard = train_k_means(DI, pref, k, SIGMA, crop, cont, preproc, Id, train_mode=train_mode, data=data, MINI_BATCH=MINI_BATCH)
        return X_tr, kmeans, standard

def density_slopex1(x):
            d=np.zeros(shape=x.shape)
            d=(1-x[:, 0])*2
            for i in range(len(x[0])):
                d[x[:, i]<0]=0
                d[x[:, i]>1]=0
            return d

def k_means_process(dataset=1, #dataset to work on
                    
                    ## k-means parameters ##
                    n_clusters=10,  #k in k-means
                    train_mode="d", #explanation below
                    MINI_BATCH=False, #Mini-batch k-means is faster
                    Id=0,           #random number generator initialiser (run with same/different Id -> same/different result)
                    
                    ## Data preprocessing ##
                    preproc=None,   #reweighting function
                    SIGMA=3,        #std of gaussian smearing kernel
                    crop=100000,    #crop training data for faster training 
                    cont=0,         #add contamination
                    
                    ## Score evaluation parameters ##
                    SCORE_TYPE="KNC",
                    knc=1,  
                    
                    ## characttereisation flugs and stuff ##
                    subfolder="", 
                    do_char=False,
                    do_plots=True,
                    SAVE_CHAR=True,
                    REVERSE=False,
                    DO_TSNE=True,
                    DO_TSNE_CENTROIDS=False, 
                    data=None,#training data that was alredy loaded and preprocessed by the prevous iteration of the algorythm and so can be reused
                    full_mean_diffs=False,
                    non_smeared_mean=False,
                    TSNE_scores=True,
                    return_scores=False,
                    return_data=True, 
                    images=True,
                    density="",
                    plot_dim=False,
                    return_k_means=False,
                    save_plots=True,
                    dijet_score_comb=False):
# To train_mode
# we are not so interested in the clustering itself at this point thus we dont
# really require the best and fully convergent clustering, repeating clustering
# with 10 initialisations is thus a waist of resources (it may be better to
# build an ensemble out of such 10 instead of picking one of 10 with best clustering)
# "d" - default as it is default in scikit
# "f" - fast (train only with one initialisation)
# "s" - stochastic train with only 1 initialisation for only 10 steps and random initialisation
    res={}    
    plt.close("all")
    k=n_clusters
    random.seed(a=10, version=2)
    
    DI = dataset_path_and_pref(dataset, REVERSE)
    
    if MINI_BATCH:
        DI["pref"]=subfolder+"MB"+DI["pref"]
        
    if cont>0:
        DI["pref"]=subfolder+DI["pref"]+"+"+DI["pref2"]+"{:}".format(cont)+"_"
    
    hyp=(DI["pref"], k, SIGMA, crop, cont, preproc, Id)
    
    X_tr, kmeans, standard = train_or_load_k_means(DI, *hyp, train_mode=train_mode, data=data, MINI_BATCH=MINI_BATCH)
    print("kmeans converged in", kmeans.n_iter_, "iterations")
    
    if return_scores or do_char:
        X_bg_val, _=prepare_data(DI["bg_val_data_path"], field=DI["bg_val_data_field"], preproc=preproc, SIGMA=SIGMA, standard=standard)
        X_sg_val, _=prepare_data(DI["sg_val_data_path"], field=DI["sg_val_data_field"], preproc=preproc, SIGMA=SIGMA, standard=standard)
        
        means=[]    #list of rho_i of the clusters
        dist_tr = kmeans.transform(X_tr)
        for i in range(k):
            dist=dist_tr[kmeans.labels_==i, i] #find distances to cluster i of points assigned to cluster i
            means.append(np.mean(dist))  #calculate rho_i
        res["means"]=means
        
        #scores use d in publication:
        if SCORE_TYPE[:4]=="logL":
            LOG_LIKELIHOOD=True 
            SCORE_TYPE_1=SCORE_TYPE[3:]
        else:
            LOG_LIKELIHOOD=False
            SCORE_TYPE_1=SCORE_TYPE
        
        if SCORE_TYPE_1=="MinD": #minimal distance
            postf="MinD"
            tr_losses = kmeans.transform(X_tr)
            bg_losses = kmeans.transform(X_bg_val)
            sg_losses = kmeans.transform(X_sg_val)
            bg_scores = np.min(bg_losses, 1)
            sg_scores = np.min(sg_losses, 1)
            tr_scores = np.min(tr_losses, 1)
    
        elif SCORE_TYPE_1=="KNC": #k-nearest clusters 
            postf="KNC"+str(knc)
            tr_losses = kmeans.transform(X_tr)
            bg_losses = kmeans.transform(X_bg_val)
            sg_losses = kmeans.transform(X_sg_val)
            bg_scores = sum_n_mins(bg_losses, knc)
            sg_scores = sum_n_mins(sg_losses, knc)
            tr_scores = sum_n_mins(tr_losses, knc)
            
        elif SCORE_TYPE_1=="Lds": #[d]=med(d), poly-gauss slopes, full volume
            postf=SCORE_TYPE
            tr_scores, bg_scores, sg_scores, tr_losses, bg_losses, sg_losses = likelyhood_estimation_dim_Uniform(kmeans, crop, k, X_tr, X_bg_val, X_sg_val, d_slopes_norm, LOG_LIKELIHOOD=LOG_LIKELIHOOD, res=res, PLOT_DIM=plot_dim, d="med")

        elif SCORE_TYPE_1=="Lrh0":
            postf=SCORE_TYPE
            new_dist=partial(d_slopes_norm, dont_use_volume=True)
            tr_scores, bg_scores, sg_scores, tr_losses, bg_losses, sg_losses = likelyhood_estimation_dim_Uniform(kmeans, crop, k, X_tr, X_bg_val, X_sg_val, new_dist, LOG_LIKELIHOOD=LOG_LIKELIHOOD, res=res, d=1)

        elif SCORE_TYPE_1=="Lrh0sp":
            postf=SCORE_TYPE
            new_dist=partial(d_slopes_norm, dont_use_volume=True)
            tr_scores, bg_scores, sg_scores, tr_losses, bg_losses, sg_losses = likelyhood_estimation_dim_Uniform_special(kmeans, crop, k, X_tr, X_bg_val, X_sg_val, new_dist, LOG_LIKELIHOOD=LOG_LIKELIHOOD, res=res, d=1)


        elif SCORE_TYPE_1=="Lrhn": #[d]=1, full volume
            postf=SCORE_TYPE
            tr_scores, bg_scores, sg_scores, tr_losses, bg_losses, sg_losses = likelyhood_estimation_dim_Uniform(kmeans, crop, k, X_tr, X_bg_val, X_sg_val, d_slopes_norm, LOG_LIKELIHOOD=LOG_LIKELIHOOD, res=res, d=1)
        
        elif SCORE_TYPE_1=="Lrh0nw": 
            postf=SCORE_TYPE
            new_dist=partial(d_slopes_norm, dont_use_volume=True, dont_use_weights=True)
            tr_scores, bg_scores, sg_scores, tr_losses, bg_losses, sg_losses = likelyhood_estimation_dim_Uniform(kmeans, crop, k, X_tr, X_bg_val, X_sg_val, new_dist, LOG_LIKELIHOOD=LOG_LIKELIHOOD, res=res, d=1)
       
        elif SCORE_TYPE_1=="Ld1600": #[d]=1, full volume
            postf=SCORE_TYPE
            tr_scores, bg_scores, sg_scores, tr_losses, bg_losses, sg_losses = likelyhood_estimation_dim_Uniform(kmeans, crop, k, X_tr, X_bg_val, X_sg_val, d_slopes_norm, LOG_LIKELIHOOD=LOG_LIKELIHOOD, res=res, d=1600)
                
        #experimental scores
        elif SCORE_TYPE_1=="Ldis": #[d]=d_i, poly-gauss slopes, full volume
            postf=SCORE_TYPE
            tr_scores, bg_scores, sg_scores, tr_losses, bg_losses, sg_losses = likelyhood_estimation_dim_Uniform(kmeans, crop, k, X_tr, X_bg_val, X_sg_val, d_slopes_norm, LOG_LIKELIHOOD=LOG_LIKELIHOOD, res=res, PLOT_DIM=plot_dim, d="ind")            
        
        elif SCORE_TYPE_1=="Lmdui": #[d]=med(d), gauss slopes, inside volume
            postf=SCORE_TYPE
            tr_scores, bg_scores, sg_scores, tr_losses, bg_losses, sg_losses = likelyhood_estimation_dim_Uniform(kmeans, crop, k, X_tr, X_bg_val, X_sg_val, half_gaussian_norm_d_inside, LOG_LIKELIHOOD=LOG_LIKELIHOOD, res=res, PLOT_DIM=plot_dim, d="med")

        elif SCORE_TYPE_1=="Lmrh1": #[d]=1, full volume
            postf=SCORE_TYPE
            tr_scores, bg_scores, sg_scores, tr_losses, bg_losses, sg_losses = likelyhood_estimation(kmeans, crop, k, X_tr, X_bg_val, X_sg_val, half_gaussian_norm_d, LOG_LIKELIHOOD=LOG_LIKELIHOOD)
            
        elif SCORE_TYPE_1=="Lrh1nw": 
            postf=SCORE_TYPE
            new_dist=partial(half_gaussian_norm_d, d=1, dont_use_weights=True)
            tr_scores, bg_scores, sg_scores, tr_losses, bg_losses, sg_losses = likelyhood_estimation(kmeans, crop, k, X_tr, X_bg_val, X_sg_val, new_dist, LOG_LIKELIHOOD=LOG_LIKELIHOOD)

        elif SCORE_TYPE_1=="Lmdu1":
            postf=SCORE_TYPE
            tr_scores, bg_scores, sg_scores, tr_losses, bg_losses, sg_losses = likelyhood_estimation(kmeans, crop, k, X_tr, X_bg_val, X_sg_val, half_gaussian_norm_d, LOG_LIKELIHOOD=LOG_LIKELIHOOD)
        
        elif SCORE_TYPE_1=="Lmdg":
            postf=SCORE_TYPE
            tr_scores, bg_scores, sg_scores, tr_losses, bg_losses, sg_losses = likelyhood_estimation_dim_Uniform(kmeans, crop, k, X_tr, X_bg_val, X_sg_val, gaussian_mult, LOG_LIKELIHOOD=LOG_LIKELIHOOD, d="med")

        elif SCORE_TYPE_1=="Lmdb":
            postf=SCORE_TYPE
            tr_scores, bg_scores, sg_scores, tr_losses, bg_losses, sg_losses = likelyhood_estimation_dim_Uniform(kmeans, crop, k, X_tr, X_bg_val, X_sg_val, ball_norm, LOG_LIKELIHOOD=LOG_LIKELIHOOD, d="med")
        
        elif SCORE_TYPE_1=="Lmdu":
            postf=SCORE_TYPE
            tr_scores, bg_scores, sg_scores, tr_losses, bg_losses, sg_losses = likelyhood_estimation_dim_Uniform(kmeans, crop, k, X_tr, X_bg_val, X_sg_val, half_gaussian_norm_d, LOG_LIKELIHOOD=LOG_LIKELIHOOD, d="med")
            
        elif SCORE_TYPE_1=="Lmd":
            postf=SCORE_TYPE
            tr_scores, bg_scores, sg_scores, tr_losses, bg_losses, sg_losses = likelyhood_estimation_dim_Uniform(kmeans, crop, k, X_tr, X_bg_val, X_sg_val, gaussian_mult_dim, LOG_LIKELIHOOD=LOG_LIKELIHOOD, d="med")
    
        elif SCORE_TYPE_1=="Lmv":
            postf=SCORE_TYPE
            tr_scores, bg_scores, sg_scores, tr_losses, bg_losses, sg_losses = likelyhood_estimation(kmeans, crop, k, X_tr, X_bg_val, X_sg_val, gaussian_mult_var, LOG_LIKELIHOOD=LOG_LIKELIHOOD)

        elif SCORE_TYPE_1=="Lrm":
            postf=SCORE_TYPE
            tr_scores, bg_scores, sg_scores, tr_losses, bg_losses, sg_losses = likelyhood_estimation(kmeans, crop, k, X_tr, X_bg_val, X_sg_val, gaussian_mult, LOG_LIKELIHOOD=LOG_LIKELIHOOD)
        
        elif SCORE_TYPE_1=="Ld":
            postf=SCORE_TYPE
            tr_scores, bg_scores, sg_scores, tr_losses, bg_losses, sg_losses = likelyhood_estimation(kmeans, crop, k, X_tr, X_bg_val, X_sg_val, exponential_slope_parameterless, LOG_LIKELIHOOD=LOG_LIKELIHOOD)
            
        elif SCORE_TYPE_1=="Lr":
            postf=SCORE_TYPE
            tr_scores, bg_scores, sg_scores, tr_losses, bg_losses, sg_losses = likelyhood_estimation(kmeans, crop, k, X_tr, X_bg_val, X_sg_val, gaussian, LOG_LIKELIHOOD=LOG_LIKELIHOOD)
            
        elif SCORE_TYPE_1=="Lrhnmax":
            postf=SCORE_TYPE
            tr_scores, bg_scores, sg_scores, tr_losses, bg_losses, sg_losses = likelyhood_estimation(kmeans, crop, k, X_tr, X_bg_val, X_sg_val, half_gaussian_max_norm, LOG_LIKELIHOOD=LOG_LIKELIHOOD)
            
        elif SCORE_TYPE_1=="Lrh":
            postf="Lrh"
            tr_scores, bg_scores, sg_scores, tr_losses, bg_losses, sg_losses = likelyhood_estimation(kmeans, crop, k, X_tr, X_bg_val, X_sg_val, half_gaussian, LOG_LIKELIHOOD=LOG_LIKELIHOOD)
            
        elif SCORE_TYPE=="logLmc": #???
            postf="logLmc"
            dist_tr = kmeans.transform(X_tr)
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
            infinity_to_min_max(bg_scores, sg_scores, tr_scores)
            infinity_to_min_max(bg_losses, sg_losses, tr_losses)    

    if do_char:
        
        if SCORE_TYPE in ["logLds", "logLdis",  "logLmdui", "logLrhn", "logLrh0"] and do_plots:
            anomalousity_histogram(X_bg_val, bg_scores, res, "_bg")
            anomalousity_histogram(X_sg_val, sg_scores, res, "_sg")
        
        if density!="" and do_plots:
            plt.figure("density", figsize=(5, 5))
            if density=="slopex1":
                d_tr=density_slopex1(X_tr)
                plt.ylabel("p(x)")

            if density=="slopex1log":
                d_tr=np.log(density_slopex1(X_tr))
                plt.ylabel("log(p(x))")
            
            if density=="Gaussian":
                d_tr=gaussian(X_tr, 0, 1, 1, 2)
                plt.ylabel("p(x)")
                
            if density=="Gaussianlog":
                d_tr=np.log(gaussian(X_tr, 0, 1, 1, 2))
                plt.ylabel("log(p(x))")
                
            
            corr=np.corrcoef(tr_scores, d_tr)
            #plt.hist2d(tr_scores, d_tr, bins=50)
            plt.scatter(tr_scores, d_tr, s=0.5, label="Pearson correlation {:.3f}".format(corr[0, 1]))#color=np.array([0, 0, 0.5, 0.1])
            #plt.axline((0, 0), slope=-1, color='k')

            #plt.hist2d(tr_scores, d_tr, bins=40, label="corellation {:.4f}".format(corr[0, 1]))
            plt.xlabel(SCORE_TYPE)
            ax=plt.gca()
            leg = ax.legend(handlelength=0, handletextpad=0, fancybox=True)
            for item in leg.legendHandles:
                item.set_visible(False)
            #ax.annotate("Pearson corellation {:.4f}".format(corr[0, 1]), xy=(-1, -1), xycoords='axes points',
            #            size=14, ha='right', va='top',
            #            bbox=dict(boxstyle='round', fc='w'))
            #plt.show()
            
        if DO_TSNE and do_plots:
            TSNE_kmeans(X_tr, X_sg_val, tr_scores, sg_scores, tr_losses, sg_losses, kmeans, n_clusters, DO_TSNE_CENTROIDS, TSNE_scores, rhos=res["means"])

        counts=np.array([np.sum(kmeans.labels_==i) for i in range(n_clusters)])
        counts.sort()
        #print(counts)
        
        # Cluster populations:
        unique, counts_train = np.unique(kmeans.labels_, return_counts=True)
        print("labels :", unique)
        print("bg train counts:", counts)
        res["cluster_counts_tr"]=counts
        
        test_bg_labels=bg_losses.argmin(1)
        test_sg_labels=sg_losses.argmin(1)
        bg_min=bg_losses.min(1)
        sg_min=sg_losses.min(1)
        
        cluster_counts_bg=np.bincount(test_bg_labels)
        cluster_counts_sg=np.bincount(test_sg_labels)
        if len(cluster_counts_bg)<n_clusters:
            cluster_counts_bg=np.concatenate((cluster_counts_bg, np.zeros(n_clusters-len(cluster_counts_bg), dtype=int)))
        if len(cluster_counts_sg)<n_clusters:
            cluster_counts_sg=np.concatenate((cluster_counts_sg, np.zeros(n_clusters-len(cluster_counts_sg), dtype=int)))
        res["cluster_counts_bg"]=cluster_counts_bg
        res["cluster_counts_sg"]=cluster_counts_sg
        
        labels=np.concatenate((np.zeros(len(X_bg_val)), np.ones(len(X_sg_val))))
        auc = roc_auc_score(labels, np.append(bg_scores, sg_scores))
        
        fpr , tpr , thresholds = roc_curve(labels, np.append(bg_scores, sg_scores))
        plt.figure("ROC")
        plt.grid()
        plt.plot(tpr, 1/fpr)
        plt.ylim(ymin=1, ymax=1000)
        plt.plot(np.linspace(0, 1, 1000), 1/np.linspace(0, 1, 1000), color="gray")
        plt.yscale("log")

        plt.figure("dist", figsize=(10, 10))
        _, bins, _, = plt.hist(bg_scores, histtype='step', label='bg', bins=40, density=True)
        plt.hist(sg_scores, histtype='step', label='sig', bins=bins, density=True)
        plt.legend(title=f'AUC: {auc:.3f}')
        
        plt.figure("SIC")
        plt.grid()
        sic_max=np.max(np.nan_to_num(tpr/fpr**0.5)[fpr!=0]) #some
        plt.plot(tpr, tpr/fpr**0.5, label="max {0:.2f}".format(sic_max))
        
        if dijet_score_comb:
            dijet_scores_combine_ROC_SIC(bg_scores, sg_scores, comb_method=dijet_score_comb, res=res)
            
        plt.figure("ROC")
        plt.legend(title=f'AUC: {auc:.3f}')
        plt.figure("SIC")
        plt.legend()
    
        
        #print("bg test:", cluster_counts_bg)
        #print("sg test:", cluster_counts_sg)
        if do_plots:
            hist_and_av(full_mean_diffs, 
                            non_smeared_mean, 
                            n_clusters, 
                            bg_losses, 
                            sg_losses, 
                            bg_min, 
                            sg_min, 
                            DI, 
                            preproc, 
                            standard, 
                            images, 
                            test_bg_labels,
                            test_sg_labels,
                            counts_train,
                            cluster_counts_bg,
                            cluster_counts_sg,
                            kmeans,
                            X_bg_val)

        """
        if do_plots:
            cols=3+full_mean_diffs+non_smeared_mean*2
            num=min(10, n_clusters)
            fig, ax = plt.subplots(num, cols, figsize=(cols*4, num*4.2), squeeze=False)
            fig.set_label("clusters")
            max_dist=max(np.max(bg_losses), np.max(sg_losses))
            max_min_dist=max(np.max(bg_min), np.max(sg_min))
            min_dist=min(0, np.min(bg_losses), np.min(sg_losses))
            min_min_dist=min(0, np.min(bg_min), np.min(sg_min))
            bins=np.linspace(min_dist, max_dist, 40)
            bins2=np.linspace(min_min_dist, max_min_dist, 40)
            
            X_bg_val_no_sm=prepare_data(DI["bg_val_data_path"], field=DI["bg_val_data_field"], preproc=preproc, SIGMA=0, standard=standard)
            for i in range(num):
                #mean image
                plt.sca(ax[i][2])
                plt.xticks([])
                plt.yticks([])
                if images:
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
                if images:
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
        """
        # Save results:
        if SAVE_CHAR:
            path="char/{:}+{:}m{:}s{:}c{:}r{:}KI{:}{:}{:}/".format(DI["pref"], DI["pref2"], k, SIGMA, crop//1000, reprocessing.reproc_names(preproc), train_mode, Id, postf)
            os.makedirs(path, exist_ok=True)
            print(plt.get_figlabels())
            if save_plots:
                for fig_label in plt.get_figlabels():
                    plt.figure(num=fig_label)
                    plt.savefig(path+'{:}.png'.format(fig_label), bbox_inches="tight")
            res["fpr"]=fpr
            res["tpr"]=tpr
            res["AUC"]=auc
            pickle.dump(res, open(path+"res.pickle", "wb"))
            print(path+"res.pickle")
    if return_scores:
        if return_data:
            if return_k_means:
                return X_tr, tr_scores, bg_scores, sg_scores, kmeans
            else:
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
        k_means_process(dataset="1h5",
                    n_clusters=100,
                    SIGMA=3,
                    crop=100000,
                    preproc=None, #reprocessing.reproc_4rt,
                    Id=0,
                    knc=5,                        
                    do_char=True,
                    SAVE_CHAR=True,
                    REVERSE=False,
                    DO_TSNE=True,
                    DO_TSNE_CENTROIDS=True, 
                    train_mode="d", 
                    SCORE_TYPE="MinD",
                    data=None,
                    images=False,
                    density="",
                    plot_dim=True)
    
    if 0.1 in switches:
        for k in [1, 4, 10, 40]:
            k_means_process(dataset="moon_demo",
                        n_clusters=k,
                        SIGMA=0,
                        crop=100000,
                        preproc=None,
                        Id=0,
                        knc=5,                                 
                        do_char=True,
                        SAVE_CHAR=True,
                        REVERSE=False,
                        DO_TSNE=True,
                        DO_TSNE_CENTROIDS=True, 
                        train_mode="d", 
                        SCORE_TYPE="MinD",
                        data=None,
                        images=False,
                        density="")
    
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
        k_arr=[1, 2, 4, 8, 16, 32, 64, 100]
        data=None
        for k in k_arr:
            data = k_means_process(n_clusters=k, dataset=1, data=data)
            
    if 12 in switches:
        k_arr=[1, 2, 4, 8, 16, 32, 64, 128, 256]
        data=None
        for k in k_arr:
            data = k_means_process(n_clusters=k, dataset=5.5, data=data, SCORE_TYPE="MinD",
                                   characterise=True,
                                   SAVE_CHAR=True,
                                   REVERSE=False,
                                   DO_TSNE=True,
                                   DO_TSNE_CENTROIDS=True,
                                   knc=5,
                                   crop=100000,
                                   SIGMA=0,
                                   cont=0,
                                   images=False)
    if 12.1 in switches:
        data=None
        data = k_means_process(n_clusters=100, dataset=3, data=data, SCORE_TYPE="logLrhn",
                               characterise=True,
                               SAVE_CHAR=True,
                               REVERSE=False,
                               DO_TSNE=True,
                               knc=5,
                               crop=100000,
                               cont=0,
                               SIGMA=3,
                               images=False)

    if 13 in switches: #To topic 17
        k_means_process(dataset=1,
                        n_clusters=100,
                        SIGMA=3,
                        crop=100000,
                        cont=0,
                        preproc=None, #reprocessing.reproc_4rt,
                        Id=0,
                        knc=5,
                        characterise=True,
                        SAVE_CHAR=True,
                        REVERSE=True,
                        DO_TSNE=True,
                        DO_TSNE_CENTROIDS=True, 
                        train_mode="d", 
                        data=None,
                        SCORE_TYPE="logLr",
                        MINI_BATCH=False)
       
    if 14 in switches: #train on contaminated datasets
       cont_list=[100, 500]
       SCORE_TYPE="MinD"
       train_mode="d"
       for cont in cont_list:

           k_means_process(dataset=1,
                            n_clusters=100,
                            SIGMA=3,
                            crop=100000,
                            cont=cont,
                            preproc=None, #reprocessing.reproc_4rt,
                            Id=0,
                            knc=1,
                            characterise=True,
                            SAVE_CHAR=True,
                            REVERSE=False,
                            DO_TSNE=True,
                            DO_TSNE_CENTROIDS=True, 
                            train_mode=train_mode, 
                            data=None,
                            SCORE_TYPE=SCORE_TYPE,
                            MINI_BATCH=False)
           
           k_means_process(dataset=1,
                            n_clusters=100,
                            SIGMA=3,
                            crop=100000,
                            cont=cont,
                            preproc=None, #reprocessing.reproc_4rt,
                            Id=0,
                            knc=1,
                            characterise=True,
                            SAVE_CHAR=True,
                            REVERSE=True,
                            DO_TSNE=True,
                            DO_TSNE_CENTROIDS=True, 
                            train_mode=train_mode, 
                            data=None,
                            SCORE_TYPE=SCORE_TYPE,
                            MINI_BATCH=False)
           
           k_means_process(dataset=2,
                            n_clusters=100,
                            SIGMA=3,
                            crop=100000,
                            cont=cont,
                            preproc=reprocessing.reproc_4rt,
                            Id=0,
                            knc=1,
                            characterise=True,
                            SAVE_CHAR=True,
                            REVERSE=False,
                            DO_TSNE=True,
                            DO_TSNE_CENTROIDS=True, 
                            train_mode=train_mode, 
                            data=None,
                            SCORE_TYPE=SCORE_TYPE,
                            MINI_BATCH=False)

           k_means_process(dataset=3,
                            n_clusters=100,
                            SIGMA=3,
                            crop=100000,
                            cont=cont,
                            preproc=None, #reprocessing.reproc_4rt,
                            Id=0,
                            knc=1,
                            characterise=True,
                            SAVE_CHAR=True,
                            REVERSE=False,
                            DO_TSNE=True,
                            DO_TSNE_CENTROIDS=True, 
                            train_mode=train_mode, 
                            data=None,
                            SCORE_TYPE=SCORE_TYPE,
                            MINI_BATCH=False)
           
           k_means_process(dataset=4,
                            n_clusters=100,
                            SIGMA=3,
                            crop=100000,
                            cont=cont,
                            preproc=None, #reprocessing.reproc_4rt,
                            Id=0,
                            knc=1,
                            characterise=True,
                            SAVE_CHAR=True,
                            REVERSE=False,
                            DO_TSNE=True,
                            DO_TSNE_CENTROIDS=True, 
                            train_mode=train_mode, 
                            data=None,
                            SCORE_TYPE=SCORE_TYPE,
                            MINI_BATCH=False)
           
           
    if 15 in switches: #production of all main uncontaminated results
       SCORE_TYPE_arr=["logLmdu"]#"MinD",  "KNC", "logLrhn", "logLmc", "logLr", "logLrh" 
       knc=5
       cont=0
       train_mode="d"
       for SCORE_TYPE in SCORE_TYPE_arr:
           k_means_process(dataset=1,
                            n_clusters=100,
                            SIGMA=3,
                            crop=100000,
                            cont=cont,
                            preproc=reprocessing.reproc_4rt,
                            Id=0,
                            knc=knc,
                            characterise=True,
                            SAVE_CHAR=True,
                            REVERSE=False,
                            DO_TSNE=True,
                            DO_TSNE_CENTROIDS=True, 
                            train_mode=train_mode, 
                            data=None,
                            SCORE_TYPE=SCORE_TYPE,
                            MINI_BATCH=False)

           k_means_process(dataset=1,
                            n_clusters=100,
                            SIGMA=3,
                            crop=100000,
                            cont=cont,
                            preproc=None, #reprocessing.reproc_4rt,
                            Id=0,
                            knc=knc,
                            characterise=True,
                            SAVE_CHAR=True,
                            REVERSE=True,
                            DO_TSNE=True,
                            DO_TSNE_CENTROIDS=True, 
                            train_mode=train_mode, 
                            data=None,
                            SCORE_TYPE=SCORE_TYPE,
                            MINI_BATCH=False)
           
           k_means_process(dataset=2,
                            n_clusters=100,
                            SIGMA=3,
                            crop=100000,
                            cont=cont,
                            preproc=reprocessing.reproc_4rt,
                            Id=0,
                            knc=knc,
                            characterise=True,
                            SAVE_CHAR=True,
                            REVERSE=False,
                            DO_TSNE=True,
                            DO_TSNE_CENTROIDS=True, 
                            train_mode=train_mode, 
                            data=None,
                            SCORE_TYPE=SCORE_TYPE,
                            MINI_BATCH=False)

           k_means_process(dataset=3,
                            n_clusters=100,
                            SIGMA=3,
                            crop=100000,
                            cont=cont,
                            preproc=None, #reprocessing.reproc_4rt,
                            Id=0,
                            knc=knc,
                            characterise=True,
                            SAVE_CHAR=True,
                            REVERSE=False,
                            DO_TSNE=True,
                            DO_TSNE_CENTROIDS=True, 
                            train_mode=train_mode, 
                            data=None,
                            SCORE_TYPE=SCORE_TYPE,
                            MINI_BATCH=False)
           
           k_means_process(dataset=4,
                            n_clusters=100,
                            SIGMA=3,
                            crop=100000,
                            cont=cont,
                            preproc=None, #reprocessing.reproc_4rt,
                            Id=0,
                            knc=knc,
                            characterise=True,
                            SAVE_CHAR=True,
                            REVERSE=False,
                            DO_TSNE=True,
                            DO_TSNE_CENTROIDS=True, 
                            train_mode=train_mode, 
                            data=None,
                            SCORE_TYPE=SCORE_TYPE,
                            MINI_BATCH=False)

