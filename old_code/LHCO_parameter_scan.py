from k_means_train_experiments import k_means_process
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import preproc.reprocessing as reprocessing
import pickle
import preproc.reprocessing as reprocessing
import time
start_time = time.time()

#method=["default", "minibatch"]
list_k_clusters=[1, 2, 5, 10, 20, 50, 100, 200]
list_sigma=[0, 0.5, 1, 2]
list_prepr=[None, reprocessing.reproc_sqrt, reprocessing.reproc_4rt]
list_combination=["mean", "max", "min"]
list_score=["logLrhn", "logLds", "MinD", "KNC"]
for prep in list_prepr:
    for sigma in list_sigma:
        data=None
        for k in list_k_clusters:
            for score in list_score:
                print("########################")
                print(prep, sigma, k, score)
                print("########################")
                print("--- %s seconds ---" % (time.time() - start_time))
                data=k_means_process(dataset="1h5",
                                n_clusters=k,
                                SIGMA=sigma,
                                crop=100000,
                                cont=2000,
                                preproc=prep,
                                Id=0,
                                knc=5,   
                                data=data,
                                do_char=True,
                                SAVE_CHAR=True,
                                do_plots=False,
                                REVERSE=False,
                                DO_TSNE=True,
                                DO_TSNE_CENTROIDS=True, 
                                train_mode="d",
                                full_mean_diffs=False,
                                non_smeared_mean=False,
                                TSNE_scores=True,
                                SCORE_TYPE=score,
                                MINI_BATCH=True,
                                return_scores=False,
                                return_data=True, 
                                images=True,
                                density="",
                                plot_dim=False,
                                return_k_means=False,
                                save_plots=False,
                                dijet_score_comb=list_combination,
                                postpr=None )


"""
images_sg=reprocessing.reproc_4rt(pickle.load(open("C://datasets/2K_SG.pickle", "rb")))
threshold=-16.5
plt.figure()
plt.imshow(np.mean(images_sg[sg_scores>threshold], axis=0))
plt.figure()
plt.imshow(np.mean(images_sg[sg_scores<threshold], axis=0))
"""
# 1 Train a model

# 2 evaluate scores
# 3 perform analysis/cut/bump hunt after cut etc.
"""
labels=np.zeros(51000)
labels[50000:]=1

#def comb_jets_scr(scores):
#    return scores[0::2]+scores[1::2]

def comb_jets_scr(scores):
    return scores[0::2]+scores[1::2]#np.maximum(scores[0::2], scores[1::2])


scores = comb_jets_scr(np.concatenate((bg_scores, sg_scores)))
auc = roc_auc_score(labels, scores)

fpr , tpr , thresholds = roc_curve(labels, scores)
plt.figure("ROC")
plt.plot(tpr, 1/fpr)
plt.ylim(ymin=1, ymax=1000)
plt.plot(np.linspace(0, 1, 1000), 1/np.linspace(0, 1, 1000), color="gray")
plt.yscale("log")
plt.legend(title=f'AUC: {auc:.3f}')

plt.figure("dist", figsize=(10, 10))
_, bins, _, = plt.hist(comb_jets_scr(bg_scores), histtype='step', label='bg', bins=40, density=True)
plt.hist(comb_jets_scr(sg_scores), histtype='step', label='sig', bins=bins, density=True)
plt.legend(title=f'AUC: {auc:.3f}')

plt.figure("SIC")
plt.grid()
sic_max=np.max(np.nan_to_num(tpr/fpr**0.5)[fpr!=0]) #some
plt.plot(tpr, tpr/fpr**0.5, label="max {0:.2f}".format(sic_max))
plt.legend()
"""