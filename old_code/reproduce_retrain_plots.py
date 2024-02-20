import numpy as np
import random
import h5py
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN, Birch
from sklearn.manifold import TSNE
import time
import os
import preproc.reprocessing as reprocessing
import pickle
from scipy.ndimage import gaussian_filter
import copy
from utils.lowpasfilter import butter_lowpass_filter
import datetime
import scipy.signal
import utils.set_matplotlib_default as smd


def sliding_cluster_performance_evaluation(
    counts_windows=None,
    plotting=True,
    filterr="med",
    labeling=">5sigma",
    W=250,
    lb=2600,
    rb=6000,
    save_path="char/0kmeans_scan/26w60wk50ret2con0.1W250ste50rewsqrtsme1ID1/",
    save_load=True,
    verbous=True,
):
    # /home/ivan/mnt/cluster/k_means_anomaly_jet/char/0kmeans_scan/26w60wk50ret2con0.1W250ste50rewsqrtsme1ID1
    res = pickle.load(open(save_path + "res.pickle", "rb"))
    kmeans_all = res["kmeans_all"]
    counts_windows = res["counts_windows"]
    steps = counts_windows.shape[0]

    Mjjmin_arr = np.linspace(lb, rb - W, steps)
    Mjjmax_arr = Mjjmin_arr + W
    window_centers = (Mjjmin_arr + Mjjmax_arr) / 2
    delta = (window_centers[1] - window_centers[0]) / 2

    k = counts_windows.shape[1]

    res = {}
    # res["counts_windows"]=counts_windows
    # res["partials_windows"]=partials_windows
    # res["HP"]=HP

    diffs = []
    for i in range(steps - 1):
        diffs.append(
            np.sum(
                (kmeans_all[i + 1].cluster_centers_ - kmeans_all[0].cluster_centers_)
                ** 2,
                axis=1,
            )
            ** 0.5
        )
    diffs = np.array(diffs)
    plt.figure(figsize=(4, 3))
    plt.grid()
    mjj_arr = (Mjjmin_arr[:-1] + Mjjmax_arr[:-1]) / 4 + (
        Mjjmin_arr[1:] + Mjjmax_arr[1:]
    ) / 4
    for j in range(k):
        plt.plot(mjj_arr, diffs[:, j])
    plt.xlabel(r"Bin centre $m_{jj}$")
    plt.ylabel(r"$|\mu-\mu_0|$")
    plt.savefig(save_path + "kmeans_cluster_abs_init_change.png", bbox_inches="tight")

    diffs = []
    for i in range(steps - 1):
        diffs.append(
            np.sum(
                (kmeans_all[i + 1].cluster_centers_ - kmeans_all[i].cluster_centers_)
                ** 2,
                axis=1,
            )
            ** 0.5
        )
    diffs = np.array(diffs)
    plt.figure(figsize=(4, 3))
    plt.grid()
    mjj_arr = (Mjjmin_arr[:-1] + Mjjmax_arr[:-1]) / 4 + (
        Mjjmin_arr[1:] + Mjjmax_arr[1:]
    ) / 4
    for j in range(k):
        plt.plot(mjj_arr, diffs[:, j])
    plt.xlabel(r"$m_{jj}$")
    plt.ylabel(r"$|d \mu/d m_{jj}|$")
    plt.savefig(save_path + "kmeans_cluster_abs_deriv.png", bbox_inches="tight")

    diffs = []
    for i in range(steps - 2):
        diffs.append(
            np.sum(
                (
                    -2 * kmeans_all[i + 1].cluster_centers_
                    + kmeans_all[i].cluster_centers_
                    + kmeans_all[i + 2].cluster_centers_
                )
                ** 2,
                axis=1,
            )
            ** 0.5
        )
    diffs = np.array(diffs)
    plt.figure(figsize=(4, 3))
    plt.grid()
    mjj_arr = (Mjjmin_arr[1:-1] + Mjjmax_arr[1:-1]) / 2
    for j in range(k):
        plt.plot(mjj_arr, diffs[:, j])
    plt.xlabel(r"$m_{jj}$")
    plt.ylabel(r"$|d^2 \mu/d m_{jj}^2|$")
    plt.savefig(save_path + "kmeans_cluster_abs_2deriv.png", bbox_inches="tight")


sliding_cluster_performance_evaluation()
