from sklearn.cluster import KMeans, MiniBatchKMeans
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
from k_means_train_experiments import k_means_process
from train_MoG_sphere import MoG_sphere_process

cont=0
train_mode="bl"
reg_covar=-11

MoG_sphere_process(dataset="4f",
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
            
MoG_sphere_process(dataset="4f",
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