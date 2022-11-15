from k_means_train_experiments import k_means_process
from train_MoG_sphere import MoG_sphere_process
from properties_table import create_table_of_kmeans_cluster_properties
from ROCcompare_1 import contamination_ROC_compare
import reprocessing
import shutil

#%% Datasets

dataset_list=[1, 1, 2, 2, 14, 14]
REVERSE_list=[False, True, False, False, False, False]
preproc_list=[None, None, reprocessing.reproc_4rt, None, reprocessing.reproc_4rt, None]
dataset_label_list=["QCD+top", "top+QCD", "QCDl+DMl", "QCDl+DMl", "QCDl+lptQ50M10", "QCDl+lptQ50M10"]
reproc_label_list=["none", "none", "4rt", "none", "4rt", "none"]

#%% run all k-means based evaluations
SCORE_TYPE_arr=["MinD",  "KNC", "logLds", "logLrhn", "logLrh0"]
SCORE_TYPE_true_names_list=["MinD",  "KNC5", "logLds", "logLrhn", "logLrh0"]
knc=5
crop=100000
n_clusters=100
cont=0
SIGMA=3
train_mode="d"
"""
for SCORE_TYPE in SCORE_TYPE_arr:
    for i in range(len(dataset_list)):
        plot_dim=False
        if SCORE_TYPE=="logLds":
            plot_dim=True
        k_means_process(dataset=dataset_list[i],
                         n_clusters=n_clusters,
                         SIGMA=SIGMA,
                         crop=crop,
                         cont=cont,
                         preproc=preproc_list[i],
                         Id=0,
                         knc=knc,
                         characterise=True,
                         SAVE_CHAR=True,
                         REVERSE=REVERSE_list[i],
                         DO_TSNE=True,
                         DO_TSNE_CENTROIDS=True, 
                         train_mode=train_mode, 
                         data=None,
                         SCORE_TYPE=SCORE_TYPE,
                         MINI_BATCH=False,
                         plot_dim=plot_dim)

#%% run all GMM evaluations
train_mode="bl"
reg_covar=-11
for i in range(len(dataset_list)):
        MoG_sphere_process(dataset=dataset_list[i],
                            n_clusters=n_clusters,
                            SIGMA=SIGMA,
                            crop=crop,
                            cont=cont,
                            preproc=preproc_list[i], 
                            Id=0,
                            reg_covar=reg_covar,
                            REVERSE=REVERSE_list[i],                                 
                            characterise=True,
                            SAVE_CHAR=True,
                            DO_TSNE=True,
                            DO_TSNE_CENTROIDS=True, 
                            train_mode=train_mode, 
                            data=None)
"""
#%% produce all ROC plots 
for dataset_label, reproc_label in zip(dataset_label_list, reproc_label_list):
    names=[]
    for SCORE_TYPE in SCORE_TYPE_true_names_list:
        names.append("char/{:}m100s3c100r{:}KId0{:}".format(dataset_label, reproc_label, SCORE_TYPE))
    names.append("char/MoG{:}m100s3c100r{:}KIbl0".format(dataset_label, reproc_label))
    contamination_ROC_compare(names,
                              methods=["MinD", "KNC5", "MLLED", "MLL1D", "MLLN", "GMMLL"], 
                              plot_name=dataset_label+reproc_label,
                              output_folder="pub_results/")
#%% store all ROC pickles

for SCORE_TYPE in SCORE_TYPE_true_names_list:
    for i in range(len(dataset_list)):
        shutil.copy("char/{:}m100s3c100r{:}KId0{:}/".format(dataset_label_list[i], reproc_label_list[i], SCORE_TYPE)+"res.pickle", "pub_results/ROC_pickles/ROC_{:}_{:}".format(dataset_label_list[i], SCORE_TYPE))
    shutil.copy("char/MoG{:}m100s3c100r{:}KIbl0/".format(dataset_label_list[i], reproc_label_list[i])+"res.pickle", "pub_results/ROC_pickles/ROC_{:}_{:}".format(dataset_label_list[i], "GMM"))

#%% produce dimensions plot
shutil.copy("char/{:}m100s3c100r{:}KId0{:}/".format(dataset_label_list[i], reproc_label_list[i], SCORE_TYPE)+"res.pickle", "pub_results/ROC_pickles/ROC_{:}_{:}".format(dataset_label_list[i], SCORE_TYPE))


#%% produce correlation plots


#%% produce a table with k-means cluster parameters
create_table_of_kmeans_cluster_properties("pub_results/dim_table.tex")
