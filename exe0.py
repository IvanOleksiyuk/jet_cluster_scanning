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


cont=0
k_means_process(dataset="1h5",
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
                train_mode="d", 
                data=None,
                smear=1,
                SCORE_TYPE="MinD",
                MINI_BATCH=False)

k_means_process(dataset="1h5",
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
                train_mode="d", 
                data=None,
                smear=1,
                SCORE_TYPE="MinD",
                MINI_BATCH=False)


