from sklearn.cluster import KMeans
import numpy as np
import random 
import matplotlib.pyplot as plt
import pickle
import reprocessing
from sklearn.metrics import roc_auc_score, roc_curve
from dataset_path_and_pref import dataset_path_and_pref

plt.rcParams.update({'font.size': 18})
def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

random.seed(a=0, version=2)
 
plt.close("all")

REVERSE=False
SIGMA=1
CROP=10000
k=16
DATASET=3
fast_train="d"
Id=0
metric_name="AUC"

pref, pref2, tra_data_path, con_data_path, bg_val_data_path, sg_val_data_path = dataset_path_and_pref(DATASET, REVERSE)

def comb_loss(losses, n=1):
    return np.mean(losses[:, :n], 1)

def metric_auc(labels, scores):
    return roc_auc_score(labels, scores)

def metric_eps_B_inv(labels, scores, eps_s=0.1):
    fpr , tpr , thresholds = roc_curve(labels, scores)
    return 1/fpr[find_nearest_idx(tpr, eps_s)]

def eval_metric(bg_losses, sg_losses, n, metric):
    bg_scores=comb_loss(bg_losses, n)
    sg_scores=comb_loss(sg_losses, n)
    if REVERSE:
        labels=np.concatenate((np.ones(len(bg_scores)), np.zeros(len(sg_scores))))
    else:
        labels=np.concatenate((np.zeros(len(bg_scores)), np.ones(len(sg_scores))))
    return metric(labels, np.append(bg_scores, sg_scores))
    
    
reproc_arr=[None, reprocessing.reproc_sqrt, reprocessing.reproc_4rt, reprocessing.reproc_log1000, reprocessing.reproc_heavi]
reproc_names_arr=[reprocessing.reproc_names(preproc) for preproc in reproc_arr]
sigma_arr=[0, 1, 3, 5]

metric=[[] for SIGMA in sigma_arr]

for i, SIGMA in enumerate(sigma_arr):
    for preproc in reproc_arr:
        X_bg_val=prepare_data(bg_val_data_path, preproc=None, SIGMA=SIGMA)
        X_sg_val=prepare_data(sg_val_data_path, preproc=None, SIGMA=SIGMA)
        MODEl_path="models/{:}m{:}s{:}c{:}r{:}KI{:}{:}.pickle".format(pref, k, SIGMA, CROP//1000, reprocessing.reproc_names(preproc), fast_train, Id)
        kmeans=pickle.load(open(MODEl_path, "rb"))
        bg_losses = kmeans.transform(X_bg_val)
        sg_losses = kmeans.transform(X_sg_val)
        bg_losses.sort()
        sg_losses.sort()
        metric[i].append(eval_metric(bg_losses, sg_losses, 1, metric_auc))
        print("done", SIGMA, reprocessing.reproc_names(preproc))
            
metric=np.array(metric)

#%% plot

def plot_2d_array(data):
    w=data.shape[1]
    h=data.shape[0]
    
    # Limits for the extent
    x_start = 0
    x_end = w
    y_start = 0
    y_end = h
    
    extent = [x_start, x_end, y_start, y_end]
    
    # The normal figure
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111)
    im = ax.imshow(data, extent=extent, origin='lower', interpolation='None', cmap='rainbow')
    
    # Add the text
    jump_x = (x_end - x_start) / (2.0 * w)
    jump_y = (y_end - y_start) / (2.0 * h)
    x_positions = np.linspace(start=x_start, stop=x_end, num=w, endpoint=False)
    y_positions = np.linspace(start=y_start, stop=y_end, num=h, endpoint=False)
    
    for y_index, y in enumerate(y_positions):
        for x_index, x in enumerate(x_positions):
            label = data[y_index, x_index]
            text_x = x + jump_x
            text_y = y + jump_y
            ax.text(text_x, text_y, "{:.3g}".format(label), color='black', ha='center', va='center')
    
    fig.colorbar(im)
    plt.show()

plot_2d_array(metric)
plt.title("{:}+{:}m{:}n{:}c{:}KI{:}{:}".format(pref, pref2, k, 1, CROP//1000, fast_train, Id)+metric_name)
plt.yticks(np.arange(len(sigma_arr))+0.5, sigma_arr)
plt.xticks(np.arange(len(reproc_names_arr))+0.5, reproc_names_arr)
plt.ylabel("sigma")
plt.xlabel("remapping")
plt.savefig("plots/{:}+{:}m{:}n{:}c{:}KI{:}{:}".format(pref, pref2, k, 1, CROP//1000, fast_train, Id)+metric_name+".png", bbox_inches='tight')