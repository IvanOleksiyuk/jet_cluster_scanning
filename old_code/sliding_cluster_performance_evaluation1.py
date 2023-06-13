import numpy as np
import random
import h5py
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN, Birch
from sklearn.manifold import TSNE
import time
import os
import reprocessing
import pickle
from scipy.ndimage import gaussian_filter
import copy
from utils.lowpasfilter import butter_lowpass_filter
import datetime

plt.close("all")


def num_der(x):
    return (x[1:] - x[:-1]) / 2


def middles(x):
    return (x[1:] + x[:-1]) / 2


def max_norm(x):
    return x / np.max(x)


cluster_with_derivatives = True
lowpass = True
W = 250
steps = 201
Mjjmin_arr = np.linspace(2000, 6000 - W, steps)
Mjjmax_arr = Mjjmin_arr + W
window_centers = (Mjjmin_arr + Mjjmax_arr) / 2
delta = (window_centers[1] - window_centers[0]) / 2
save_path = "char/0kmeans_scan/k20ret1con0.1W250ste201rewnonesme0ID0/"
os.makedirs(save_path + "eval/", exist_ok=True)

res = pickle.load(open(save_path + "res.pickle", "rb"))
counts_windows = res["counts_windows"]
k = counts_windows.shape[1]

countmax_windows = np.zeros(counts_windows.shape)
for i in range(k):
    countmax_windows[:, i] = max_norm(counts_windows[:, i])

count_sum = np.sum(counts_windows, axis=1)
count_sum = count_sum.reshape((len(count_sum), 1))
countmax_sum = count_sum / np.max(count_sum)

plt.grid()
plt.hist(np.sum((countmax_windows - countmax_sum) ** 2, axis=0) ** 0.5, bins=20)
plt.xlabel("distance to bg")
plt.ylabel("cluster n")
plt.savefig(save_path + "eval/curve_distance.png")

plt.figure()
plt.grid()
plt.hist(np.sum(countmax_windows, axis=0), bins=20)
plt.xlabel("integral under the curve")
plt.ylabel("cluster n")
plt.savefig(save_path + "eval/curves_integrals.png")


num_der_counts_windows = num_der(countmax_windows)
if lowpass:
    order = 6
    fs = 1  # sample rate, Hz
    cutoff = 0.05  # desired cutoff frequency of the filter, Hz
    for i in range(k):
        num_der_counts_windows[:, i] = butter_lowpass_filter(
            num_der_counts_windows[:, i], cutoff, fs, order
        )


kmeans = KMeans(2)

if cluster_with_derivatives:
    kmeans.fit(num_der_counts_windows.T)
    as_vectors = "derivatives"
else:
    kmeans.fit(countmax_windows.T)
    as_vectors = "curves"


# all center curves with found labels
plt.figure()
plt.grid()
for j in range(k):
    if kmeans.labels_[j]:
        plt.plot(window_centers, countmax_windows[:, j], color="red")
    else:
        plt.plot(window_centers, countmax_windows[:, j], color="blue")
plt.xlabel("mjj window centre")
plt.ylabel("n_clusater/max(n_cluster)")
plt.title(f"curves clustered with kmeans with {as_vectors} as vectors")
plt.savefig(save_path + "eval/countmax_windows.png")

# all center curve derivatives with found labels
plt.figure()
plt.grid()
for j in range(k):
    if kmeans.labels_[j]:
        plt.plot(middles(window_centers), num_der_counts_windows[:, j], color="red")
    else:
        plt.plot(middles(window_centers), num_der_counts_windows[:, j], color="blue")
plt.xlabel("mjj")
plt.ylabel("delta(n_clusater/max(n_cluster))")
plt.title(f"derivatives clustered with kmeans with {as_vectors} as vectors")
plt.savefig(save_path + "eval/num_der_counts_windows.png")

# TSNE
X_embedded = TSNE().fit_transform(num_der_counts_windows.T)
plt.figure()
plt.grid()
plt.plot(
    X_embedded[:, 0][kmeans.labels_ == 1],
    X_embedded[:, 1][kmeans.labels_ == 1],
    ".",
    color="red",
)
plt.plot(
    X_embedded[:, 0][kmeans.labels_ == 0],
    X_embedded[:, 1][kmeans.labels_ == 0],
    ".",
    color="blue",
)
plt.xlabel("embedding dim 0")
plt.ylabel("embedding dim 1")
plt.savefig(save_path + "eval/TSNE.png")

# combinations
plt.figure()
plt.grid()
anomaly_poor = np.sum(counts_windows[:, kmeans.labels_ == 1], axis=1)
anomaly_rich = np.sum(counts_windows[:, kmeans.labels_ == 0], axis=1)
plt.plot(window_centers, max_norm(anomaly_poor), label="sum of cluster 0 curves")
plt.plot(window_centers, max_norm(anomaly_rich), label="sum of cluster 1 curves")
plt.plot(window_centers, max_norm(count_sum), "--", label="all")
plt.xlabel("mjj window centre")
plt.legend()
plt.savefig(save_path + "eval/comb.png")

# bump-hunting!
import pyBumpHunter as BH

hunter = BH.BumpHunter1D(
    rang=[2125 - delta, 6000 - 125 + delta],
    bins=201,
    width_min=5,
    width_max=19,
    width_step=2,
    scan_step=1,
    npe=100,
    nworker=7,
    seed=666,
)


def dehist(hist):
    data = np.array([])
    for k, n in enumerate(anomaly_rich):
        data = np.append(data, window_centers[k] + (np.random.rand(n) - 0.5) * delta)
    return data


"""
data=dehist(anomaly_rich)

print('####bump_scan call####')
hunter.bump_scan(data, dehist(anomaly_poor))
print('')

print(hunter.bump_info(data))
#print(f'   mean (true) = {Lth}')
"""
# hunter.plot_tomography(data)
