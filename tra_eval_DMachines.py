from k_means_train_experiments import k_means_process
import pickle
import matplotlib.pyplot as plt
from utilities import find_nearest_idx_above
signals_per_cahnel=[8, 7, 9, 10]  

for j, chanel in enumerate(["1", "2a", "2b", "3"]):  
    for i in range(signals_per_cahnel[j]):
        dataset="DM_"+chanel+"_"+str(i)
        k_means_process(dataset=dataset,
            n_clusters=100,
            SIGMA=0,
            crop=100000,
            preproc=None, #reprocessing.reproc_4rt,
            Id=0,
            knc=5,                        
            characterise=True,
            SAVE_CHAR=True,
            REVERSE=False,
            DO_TSNE=False,
            DO_TSNE_CENTROIDS=True, 
            train_mode="rob", 
            SCORE_TYPE="MinD",
            data=None,
            images=False,
            density="",
            plot_dim=True,
            save_plots=False)

#%%
plt.close("all")
method=1
for j, chanel in enumerate(["1", "2a", "2b", "3"]):
    for i in range(signals_per_cahnel[j]):
        res_path="char/BGchan{:}+SG{:}chan{:}m100s0c100rnoneKId0MinD/res.pickle".format(chanel, i, chanel)
        res=pickle.load(open(res_path, "rb"))
        plt.figure(1)
        plt.plot(res["AUC"], method, ".")
        plt.figure(2)
        plt.plot(res["tpr"][find_nearest_idx_above(1/res["fpr"], 100)], method, ".")
        plt.figure(3)
        plt.plot(res["tpr"][find_nearest_idx_above(1/res["fpr"], 1000)], method, ".")
        plt.figure(4)
        plt.plot(res["tpr"][find_nearest_idx_above(1/res["fpr"], 10000)], method, ".")
        
method=2
for j, chanel in enumerate(["1", "2a", "2b", "3"]):
    for i in range(signals_per_cahnel[j]):
        res_path="char/BGchan{:}+SG{:}chan{:}m100s0c100rnoneKId0KNC5/res.pickle".format(chanel, i, chanel)
        res=pickle.load(open(res_path, "rb"))
        plt.figure(1)
        plt.plot(res["AUC"], method, ".")
        plt.figure(2)
        plt.plot(res["tpr"][find_nearest_idx_above(1/res["fpr"], 100)], method, ".")
        plt.figure(3)
        plt.plot(res["tpr"][find_nearest_idx_above(1/res["fpr"], 1000)], method, ".")
        plt.figure(4)
        plt.plot(res["tpr"][find_nearest_idx_above(1/res["fpr"], 10000)], method, ".")
        
method=3
for j, chanel in enumerate(["1", "2a", "2b", "3"]):
    for i in range(signals_per_cahnel[j]):
        res_path="char/BGchan{:}+SG{:}chan{:}m100s0c100rnoneKIrob0MinD/res.pickle".format(chanel, i, chanel)
        res=pickle.load(open(res_path, "rb"))
        plt.figure(1)
        plt.plot(res["AUC"], method, ".")
        plt.figure(2)
        plt.plot(res["tpr"][find_nearest_idx_above(1/res["fpr"], 100)], method, ".")
        plt.figure(3)
        plt.plot(res["tpr"][find_nearest_idx_above(1/res["fpr"], 1000)], method, ".")
        plt.figure(4)
        plt.plot(res["tpr"][find_nearest_idx_above(1/res["fpr"], 10000)], method, ".")
        
method=4
for j, chanel in enumerate(["1", "2a", "2b", "3"]):
    for i in range(signals_per_cahnel[j]):
        res_path="char/BGchan{:}+SG{:}chan{:}m100s0c100rnoneKIrob0KNC5/res.pickle".format(chanel, i, chanel)
        res=pickle.load(open(res_path, "rb"))
        plt.figure(1)
        plt.plot(res["AUC"], method, ".")
        plt.figure(2)
        plt.plot(res["tpr"][find_nearest_idx_above(1/res["fpr"], 100)], method, ".")
        plt.figure(3)
        plt.plot(res["tpr"][find_nearest_idx_above(1/res["fpr"], 1000)], method, ".")
        plt.figure(4)
        plt.plot(res["tpr"][find_nearest_idx_above(1/res["fpr"], 10000)], method, ".")
        
plt.figure(2)
plt.xscale("log")     
plt.figure(3)
plt.xscale("log")  
plt.figure(4)
plt.xscale("log")  
