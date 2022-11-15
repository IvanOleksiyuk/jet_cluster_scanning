from sklearn.cluster import KMeans
import numpy as np
import random 
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.ndimage import gaussian_filter
import pickle
import reprocessing
from sklearn.metrics import roc_auc_score, roc_curve

random.seed(a=0, version=2)
 
plt.close("all")

REVERSE=False


if REVERSE:
    bg_data_path="C:/bachelor work/Spyder/image_data_sets/Xtra-100Ktop-pre3-2.pickle"
else:
    bg_data_path="C:/bachelor work/Spyder/image_data_sets/Xtra-100KQCD-pre3-2.pickle"
bg_val_data_path="C:/bachelor work/Spyder/image_data_sets/Xval-40KQCD-pre3-2.pickle"
sg_val_data_path="C:/bachelor work/Spyder/image_data_sets/Xval-40Ktop-pre3-2.pickle"
"""

if REVERSE:
    bg_data_path="C:/bachelor work/Spyder/image_data_sets/Xtra-100KDMl.pickle"
else:
    bg_data_path="C:/bachelor work/Spyder/image_data_sets/Xtra-100KQCDl.pickle"
bg_val_data_path="C:/bachelor work/Spyder/image_data_sets/Xval-40KQCDl.pickle"
sg_val_data_path="C:/bachelor work/Spyder/image_data_sets/Xval-40KDMl.pickle"
"""
n_clasters=2
SIGMA=1
DO_TSNE=True
preproc=None#reprocessing.reproc_sqrt
preproc_name=reprocessing.reproc_names(preproc)
crop=10000

def prepare_data(path, crop=-1, preproc=None):    
    X = pickle.load(open(path, "rb"))
    X = X[:crop]
    if preproc is not None:
        print("preprocessing active")
        X = preproc(X)
    X = gaussian_filter(X, sigma=[0, SIGMA, SIGMA, 0]) 
    X = X.reshape((X.shape[0], 1600))
    return X

X_bg=prepare_data(bg_data_path, crop=crop, preproc=preproc)
X_bg_val=prepare_data(bg_val_data_path, preproc=preproc)
X_sg_val=prepare_data(sg_val_data_path, preproc=preproc)

print("done loading data")
kmeans_arr=[]
for i in range(1):
    kmeans_arr.append(KMeans(n_clusters=n_clasters, random_state=0).fit(X_bg))
print("done clustering")
#plot centroids:
fig, axs = plt.subplots(nrows=1, ncols=n_clasters)
kmeans=kmeans_arr[0]
for i in range(n_clasters):
    axs[i].imshow(kmeans.cluster_centers_[i].reshape((40, 40)))

if DO_TSNE:
    IDs_TSNE=np.random.randint(0, X_bg.shape[0]-1, 1000, )
    labels_TSNE=kmeans.labels_[IDs_TSNE]
    
    Y = TSNE(n_components=2, n_iter=1000, random_state=10).fit_transform(X_bg[IDs_TSNE])
    plt.figure(figsize=(10, 10))
    u_labels = np.unique(kmeans.labels_)
    for i in u_labels:
        plt.scatter(Y[labels_TSNE==i, 0], Y[labels_TSNE==i, 1], label=i)
    print("done TSNE")

bg_losses=0
sg_losses=0
for kmeans in kmeans_arr:
    bg_losses += kmeans.transform(X_bg_val)
    sg_losses += kmeans.transform(X_sg_val)

plt.figure(figsize=(10, 10))

def comb_loss(losses, n=1):
    losses.sort(1)
    return np.mean(losses[:, :n], 1)

bg_scores=comb_loss(bg_losses)
sg_scores=comb_loss(sg_losses)


#%%
plt.hist(bg_scores, histtype='step', label='bg', bins=40, density=True)
plt.hist(sg_scores, histtype='step', label='sig', bins=40, density=True)
labels=np.concatenate((np.zeros(len(X_bg_val)), np.ones(len(X_sg_val))))
auc = roc_auc_score(labels, np.append(bg_scores, sg_scores))
plt.legend(title=f'AUC: {auc:.2f}')
plt.show()
fpr , tpr , thresholds = roc_curve(labels, np.append(bg_scores, sg_scores))
plt.figure()
plt.plot(tpr, 1/fpr)
plt.yscale("log")
plt.ylim(ymin=1, ymax=10000)


counts=np.array([np.sum(kmeans.labels_==i) for i in range(n_clasters)])
counts.sort()
print(counts)

