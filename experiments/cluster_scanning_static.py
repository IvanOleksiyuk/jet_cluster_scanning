# imports
import copy
import os
import pickle
import random
import time
import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans, MiniBatchKMeans
import preproc.reprocessing as reprocessing
from utils.config_utils import Config

# Loading configuration
config_file = "config/test.yml"
config = Config(config_file)
cfg = config.get_dotmap()

# parsing connected to preprocessing
reproc = reprocessing.Reprocessing(cfg.reproc_arg_string)
cfg.reproc_name = reproc.name

# Creating a path to save results
save_path = "char/0kmeans_scan/k{:}{:}ret{:}con{:}" "W{:}ste{:}_{:}_ID{:}/".format(
    cfg.k,
    cfg.MiniBatch,
    cfg.retrain,
    cfg.signal_fraction,
    cfg.W,
    cfg.steps,
    cfg.reproc_name,
    cfg.ID,
)
os.makedirs(save_path, exist_ok=True)


# Transform some config arguments into useful stuff
Mjjmin_arr = np.linspace(cfg.eval_interval[0], cfg.eval_interval[1] - cfg.W, cfg.steps)
Mjjmax_arr = Mjjmin_arr + cfg.W
HP = config.get_dict()
random.seed(a=cfg.ID, version=2)  # set a seed corresponding to the ID

# Loading data
mjj_bg = np.load(cfg.data_path + "mjj_bkg_sort.npy")
mjj_sg = np.load(cfg.data_path + "mjj_sig_sort.npy")
im_bg_file = h5py.File(cfg.data_path + "v2JetImSort_bkg.h5", "r")
im_sg_file = h5py.File(cfg.data_path + "v2JetImSort_sig.h5", "r")
im_bg = im_bg_file["data"]
im_sg = im_sg_file["data"]

plt.imshow(reproc(im_bg[1:3].reshape((-1, 40, 40)))[0])
plt.show()

# Create flags for the signal data taken in this run
num_true = int(np.rint(cfg.signal_fraction * len(im_sg)))
print(num_true)
allowed = np.concatenate(
    (
        np.zeros(len(im_sg) - num_true, dtype=bool),
        np.ones(num_true, dtype=bool),
    )
)
np.random.shuffle(allowed)


# Smart slising
def Mjj_slise(Mjjmin, Mjjmax, allowed=None, bootstrap_bg=None):
    """Returns the background an signal jets in a given Mjj window

    Args:
        Mjjmin (float): lower Mjj interval limit
        Mjjmax (float): upper Mjj interval limit
        allowed (list/array of integers or bool of size len(im_sg)):
            Indicates which and how any times each signal image is chosen for the dataset
        bootstrap_bg (list/array of integers of size len(im_bg)):
            Indicates which and how any times each background image is chosen for the dataset

    Returns:
        _type_: _description_
    """
    print("loading window", Mjjmin, Mjjmax)
    indexing_bg = np.logical_and(mjj_bg >= Mjjmin, mjj_bg <= Mjjmax)
    indexing_bg = np.where(indexing_bg)[0]

    indexing_sg = np.logical_and(mjj_sg >= Mjjmin, mjj_sg <= Mjjmax)
    indexing_sg = np.where(indexing_sg)[0]

    print(len(indexing_bg), "bg events found in interval")
    print(len(indexing_sg), "sg events found in interval")

    start_time = time.time()
    print("start data extraction")
    if bootstrap_bg is None:
        bg = im_bg[indexing_bg[0] : indexing_bg[-1]]
    else:
        print(len(im_bg[indexing_bg[0] : indexing_bg[-1]]))
        print(len(bootstrap_bg[indexing_bg[0] : indexing_bg[-1]]))
        bg = np.repeat(
            im_bg[indexing_bg[0] : indexing_bg[-1]],
            bootstrap_bg[indexing_bg[0] : indexing_bg[-1]],
            axis=0,
        )

    if allowed is not None:
        sg = np.repeat(
            im_sg[indexing_sg[0] : indexing_sg[-1]],
            allowed[indexing_sg[0] : indexing_sg[-1]],
            axis=0,
        )
        print("only", len(sg), "sg events taken")
    # test if chnaging to this spares some time
    # sg=im_sg[indexing_sg[0]:indexing_sg[-1]]
    # sg=sg[allowed[indexing_sg[0]:indexing_sg[-1]]]
    print("only", len(bg), "bg events taken")
    print("load --- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()

    if allowed is not None:
        data = np.concatenate((bg, sg))
    else:
        data = bg
    data = data.reshape((len(data) * cfg.jet_per_event, 40, 40))
    print("concat --- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    data = reproc(data)
    data = data.reshape((len(data), 40 * 40))
    print("reproc --- %s seconds ---" % (time.time() - start_time))
    return data


# Initiate k-means or MB k-means
if cfg.MiniBatch:
    kmeans = MiniBatchKMeans(cfg.k)
else:
    kmeans = KMeans(cfg.k)

# Train k-means in the training window
start_time = time.time()
data = Mjj_slise(cfg.train_interval[0], cfg.train_interval[1])
kmeans.fit(data)
print("trained --- %s seconds ---" % (time.time() - start_time))

print(kmeans.cluster_centers_)

# Evaluate lables for the whole dataset
bg_lab = kmeans.predict(
    reproc(im_bg[:].reshape((-1, 40, 40))).reshape((-1, 1600))
).reshape((-1, cfg.jet_per_event))
sg_lab = kmeans.predict(
    reproc(im_sg[:].reshape((-1, 40, 40))).reshape((-1, 1600))
).reshape((-1, cfg.jet_per_event))

print(bg_lab)
print(sg_lab)


# Function to count entries in one bin
def count_bin(mjjmin, mjjmax, allowed, bootstrap_bg=None):
    """
    Counts a number of events for all classes in a given Mjj window

    Args:
        mjjmin (float): lower Mjj interval limit
        mjjmax (float): upper Mjj interval limit
        allowed (list/array of integers or bool of size len(im_sg)):
            Indicates which and how any times each signal image is chosen for the dataset
        bootstrap_bg (list/array of integers of size len(im_bg)):
            Indicates which and how any times each background image is chosen for the dataset

    Returns:
        _type_: _description_
    """
    indexing_bg = np.logical_and(mjj_bg >= mjjmin, mjj_bg <= mjjmax)
    indexing_bg = np.where(indexing_bg)[0]
    indexing_sg = np.logical_and(mjj_sg >= mjjmin, mjj_sg <= mjjmax)
    indexing_sg = np.where(indexing_sg)[0]

    print(len(indexing_bg), "bg events found in interval")
    print(len(indexing_sg), "sg events found in interval")

    if bootstrap_bg is None:
        bg = bg_lab[indexing_bg[0] : indexing_bg[-1]]
    else:
        print(len(bg_lab[indexing_bg[0] : indexing_bg[-1]]))
        print(len(bootstrap_bg[indexing_bg[0] : indexing_bg[-1]]))
        bg = np.repeat(
            bg_lab[indexing_bg[0] : indexing_bg[-1]],
            bootstrap_bg[indexing_bg[0] : indexing_bg[-1]],
            axis=0,
        )

    if allowed is not None:
        sg = np.repeat(
            sg_lab[indexing_sg[0] : indexing_sg[-1]],
            allowed[indexing_sg[0] : indexing_sg[-1]],
            axis=0,
        )
        all_lab = np.concatenate((bg, sg))
    else:
        all_lab = bg
    return np.array([np.sum(all_lab == j) for j in range(cfg.k)])


# Function to count the whole histogram
counts_windows = []
for i in range(cfg.steps):
    counts_windows.append(count_bin(Mjjmin_arr[i], Mjjmax_arr[i], allowed))

# print(len(counts_windows))
counts_windows = np.stack(counts_windows)


# Some plotting
window_centers = (Mjjmin_arr + Mjjmax_arr) / 2
min_allowed_count = 100
min_min_allowed_count = 10
plt.figure()
plt.grid()
for j in range(cfg.k):
    plt.plot(window_centers, counts_windows[:, j])
plt.xlabel("m_jj")
plt.ylabel("n points from window")
plt.savefig(save_path + "kmeans_ni_mjj_total.png")
smallest_cluster_count_window = np.min(counts_windows, axis=1)
for i in range(len(window_centers)):
    if smallest_cluster_count_window[i] < min_allowed_count:
        if smallest_cluster_count_window[i] < min_min_allowed_count:
            plt.axvline(window_centers[i], color="black", alpha=0.6)
        else:
            plt.axvline(window_centers[i], color="black", alpha=0.3)

plt.savefig(save_path + "kmeans_ni_mjj_total_statAllowed.png")

partials_windows = np.zeros(counts_windows.shape)
for i in range(len(Mjjmin_arr)):
    partials_windows[i, :] = counts_windows[i, :] / np.sum(counts_windows[i, :])

plt.figure()
plt.grid()
for j in range(cfg.k):
    plt.plot(window_centers, partials_windows[:, j])
plt.xlabel("m_jj")
plt.ylabel("fraction of points in window")
plt.savefig(save_path + "kmeans_xi_mjj_total.png")

countmax_windows = np.zeros(counts_windows.shape)
for i in range(cfg.k):
    countmax_windows[:, i] = counts_windows[:, i] / np.max(counts_windows[:, i])

plt.figure()
plt.grid()
for j in range(cfg.k):
    plt.plot(window_centers, countmax_windows[:, j])

conts_bg = []
conts_sg = []
for Mjjmin, Mjjmax in zip(Mjjmin_arr, Mjjmax_arr):
    conts_bg.append(np.sum(np.logical_and(mjj_bg >= Mjjmin, mjj_bg <= Mjjmax)))
    conts_sg.append(
        np.sum(
            np.logical_and(np.logical_and(mjj_sg >= Mjjmin, mjj_sg <= Mjjmax), allowed)
        )
    )
conts_bg = np.array(conts_bg)
conts_sg = np.array(conts_sg)
conts = conts_bg + conts_sg
plt.plot(window_centers, conts / np.max(conts), "--")
plt.xlabel("m_jj")
plt.ylabel("n points from window/max(...)")
plt.savefig(save_path + "kmeans_xi_mjj_maxn.png")

for i in range(len(window_centers)):
    if smallest_cluster_count_window[i] < min_allowed_count:
        if smallest_cluster_count_window[i] < min_min_allowed_count:
            plt.axvline(window_centers[i], color="black", alpha=0.6)
        else:
            plt.axvline(window_centers[i], color="black", alpha=0.3)

plt.savefig(save_path + "kmeans_xi_mjj_maxn_statAllowed.png")

# Save results
# Save all labels


# close files
im_bg_file.close()
im_sg_file.close()
