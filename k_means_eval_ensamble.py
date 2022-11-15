import numpy as np
import matplotlib.pyplot as plt
import pickle
import reprocessing
from sklearn.metrics import roc_auc_score, roc_curve
import os
from k_means_train_experiments import k_means_process

def standatisation_params(training_set):
    mean=np.mean(training_set)
    std=np.std(training_set)
    return std, mean

def standatisation(dataset, std, mean):
    return (dataset-mean)/std

def score_sum(scores_list):
    fin_score=np.sum(np.vstack(scores_list), axis=0)
    return fin_score
    
def score_mean(scores_list):
    fin_score=np.mean(np.vstack(scores_list), axis=0)
    return fin_score

def score_max(scores_list):
    fin_score=np.max(np.vstack(scores_list), axis=0)
    return fin_score

def score_min(scores_list):
    fin_score=np.min(np.vstack(scores_list), axis=0)
    return fin_score

def rank_max(scores_list):
    ranks_list=[]
    for scores in scores_list:
        ranks_list.append(np.argsort(scores))
    return score_max(ranks_list)

def rank_mean(scores_list):
    ranks_list=[]
    for scores in scores_list:
        ranks_list.append(np.argsort(scores))
    return score_mean(ranks_list)

def rank_min(scores_list):
    ranks_list=[]
    for scores in scores_list:
        ranks_list.append(np.argsort(scores))
    return score_mean(ranks_list)
    
    
def characterise(tr_scores, bg_scores, sg_scores):
    
    labels=np.concatenate((np.zeros(len(bg_scores)), np.ones(len(sg_scores))))
    auc = roc_auc_score(labels, np.append(bg_scores, sg_scores))
    
    fpr , tpr , thresholds = roc_curve(labels, np.append(bg_scores, sg_scores))
    
    plt.figure(2)
    plt.grid()
    plt.plot(tpr, 1/fpr)
    plt.ylim(ymin=1, ymax=1000)
    plt.plot(np.linspace(0, 1, 1000), 1/np.linspace(0, 1, 1000), color="gray")
    plt.yscale("log")
    plt.legend(title=f'AUC: {auc:.3f}')    
    
    plt.figure(3)
    plt.grid()
    sic_max=np.max(np.nan_to_num(tpr/fpr**0.5)[fpr!=0]) #some
    plt.plot(tpr, tpr/fpr**0.5, label="max {0:.2f}".format(sic_max))
    plt.legend()

    plt.figure()
    _, bins, _, = plt.hist(bg_scores, histtype='step', label='bg', bins=40, density=True)
    plt.hist(sg_scores, histtype='step', label='sig', bins=bins, density=True)
    plt.legend(title=f'AUC: {auc:.3f}')

tr_scores_arr=[]
bg_scores_arr=[]
sg_scores_arr=[]

tr_scores, bg_scores, sg_scores = k_means_process(dataset=1,
                                                  n_clusters=100,
                                                  SIGMA=3,
                                                  crop=100000,
                                                  preproc=None,
                                                  Id=0,
                                                  knc=1,
                                                  characterise=False,
                                                  SAVE_CHAR=True,
                                                  REVERSE=False,
                                                  DO_TSNE=True,
                                                  DO_TSNE_CENTROIDS=True, 
                                                  train_mode="d", 
                                                  data=None,
                                                  SCORE_TYPE="logLrhn",
                                                  MINI_BATCH=False,
                                                  return_data=False,
                                                  return_scores=True)

std, mean = standatisation_params(tr_scores)
tr_scores_arr.append(standatisation(tr_scores, std, mean))
bg_scores_arr.append(standatisation(bg_scores, std, mean))
sg_scores_arr.append(standatisation(sg_scores, std, mean))

tr_scores, bg_scores, sg_scores = k_means_process(dataset=1,
                                                  n_clusters=100,
                                                  SIGMA=3,
                                                  crop=100000,
                                                  preproc=reprocessing.reproc_sqrt,
                                                  Id=0,
                                                  knc=1,
                                                  characterise=False,
                                                  SAVE_CHAR=True,
                                                  REVERSE=False,
                                                  DO_TSNE=True,
                                                  DO_TSNE_CENTROIDS=True, 
                                                  train_mode="d", 
                                                  data=None,
                                                  smear=1,
                                                  SCORE_TYPE="logLrhn",
                                                  MINI_BATCH=False,
                                                  return_data=False,
                                                  return_scores=True)
std, mean = standatisation_params(tr_scores)
tr_scores_arr.append(standatisation(tr_scores, std, mean))
bg_scores_arr.append(standatisation(bg_scores, std, mean))
sg_scores_arr.append(standatisation(sg_scores, std, mean))

tr_scores, bg_scores, sg_scores = k_means_process(dataset=1,
                                                  n_clusters=100,
                                                  SIGMA=3,
                                                  crop=100000,
                                                  preproc=reprocessing.reproc_4rt,
                                                  Id=0,
                                                  knc=1,
                                                  characterise=False,
                                                  SAVE_CHAR=True,
                                                  REVERSE=False,
                                                  DO_TSNE=True,
                                                  DO_TSNE_CENTROIDS=True, 
                                                  train_mode="d", 
                                                  data=None,
                                                  smear=1,
                                                  SCORE_TYPE="logLrhn",
                                                  MINI_BATCH=False,
                                                  return_data=False,
                                                  return_scores=True)


std, mean = standatisation_params(tr_scores)
tr_scores_arr.append(standatisation(tr_scores, std, mean))
bg_scores_arr.append(standatisation(bg_scores, std, mean))
sg_scores_arr.append(standatisation(sg_scores, std, mean))

tr_scores, bg_scores, sg_scores = k_means_process(dataset=1,
                                                  n_clusters=100,
                                                  SIGMA=1,
                                                  crop=100000,
                                                  preproc=None,
                                                  Id=0,
                                                  knc=1,
                                                  characterise=False,
                                                  SAVE_CHAR=True,
                                                  REVERSE=False,
                                                  DO_TSNE=True,
                                                  DO_TSNE_CENTROIDS=True, 
                                                  train_mode="d", 
                                                  data=None,
                                                  SCORE_TYPE="logLrhn",
                                                  MINI_BATCH=False,
                                                  return_data=False,
                                                  return_scores=True)

std, mean = standatisation_params(tr_scores)
tr_scores_arr.append(standatisation(tr_scores, std, mean))
bg_scores_arr.append(standatisation(bg_scores, std, mean))
sg_scores_arr.append(standatisation(sg_scores, std, mean))

tr_scores, bg_scores, sg_scores = k_means_process(dataset=1,
                                                  n_clusters=100,
                                                  SIGMA=1,
                                                  crop=100000,
                                                  preproc=reprocessing.reproc_sqrt,
                                                  Id=0,
                                                  knc=1,
                                                  characterise=False,
                                                  SAVE_CHAR=True,
                                                  REVERSE=False,
                                                  DO_TSNE=True,
                                                  DO_TSNE_CENTROIDS=True, 
                                                  train_mode="d", 
                                                  data=None,
                                                  smear=1,
                                                  SCORE_TYPE="logLrhn",
                                                  MINI_BATCH=False,
                                                  return_data=False,
                                                  return_scores=True)

std, mean = standatisation_params(tr_scores)
tr_scores_arr.append(standatisation(tr_scores, std, mean))
bg_scores_arr.append(standatisation(bg_scores, std, mean))
sg_scores_arr.append(standatisation(sg_scores, std, mean))

tr_scores, bg_scores, sg_scores = k_means_process(dataset=1,
                                                  n_clusters=100,
                                                  SIGMA=1,
                                                  crop=100000,
                                                  preproc=reprocessing.reproc_4rt,
                                                  Id=0,
                                                  knc=1,
                                                  characterise=False,
                                                  SAVE_CHAR=True,
                                                  REVERSE=False,
                                                  DO_TSNE=True,
                                                  DO_TSNE_CENTROIDS=True, 
                                                  train_mode="d", 
                                                  data=None,
                                                  smear=1,
                                                  SCORE_TYPE="logLrhn",
                                                  MINI_BATCH=False,
                                                  return_data=False,
                                                  return_scores=True)


std, mean = standatisation_params(tr_scores)
tr_scores_arr.append(standatisation(tr_scores, std, mean))
bg_scores_arr.append(standatisation(bg_scores, std, mean))
sg_scores_arr.append(standatisation(sg_scores, std, mean))

#%%
plt.figure()
for tr_scores in tr_scores_arr:
    plt.hist(tr_scores, histtype='step', bins=50)
#%%
func_arr=[score_max, score_min, score_mean]#, rank_max, rank_min, rank_mean]
names_arr=[]

for i in range(len(func_arr)):
    tr_scores=func_arr[i](tr_scores_arr) 
    bg_scores=func_arr[i](bg_scores_arr) 
    sg_scores=func_arr[i](sg_scores_arr) 
    
    characterise(tr_scores, bg_scores, sg_scores)


