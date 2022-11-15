from k_means_train_experiments import k_means_process
import matplotlib.pyplot as plt
import numpy as np
X_tr, tr_scores, bg_scores, sg_scores = k_means_process(dataset="4f",
                    n_clusters=100,
                    SIGMA=3,
                    crop=100000,
                    preproc=None, #reprocessing.reproc_4rt,
                    Id=0,
                    knc=5,                        
                    characterise=False,
                    train_mode="d", 
                    SCORE_TYPE="logLrhn",
                    return_scores=True,
                    return_data=True)


X_tr, tr_scores_rh0, bg_scores_rh0, sg_scores_rh0 = k_means_process(dataset="4f",
                    n_clusters=100,
                    SIGMA=3,
                    crop=100000,
                    preproc=None, #reprocessing.reproc_4rt,
                    Id=0,
                    knc=5,    
                    data=X_tr,              
                    characterise=False,
                    train_mode="d", 
                    SCORE_TYPE="logLrh0sp",
                    return_scores=True,
                    return_data=True)

min_=np.min(tr_scores_rh0)
tr_scores_rh0-=min_
tr_scores-=min_
tr_scores_ns=tr_scores-tr_scores_rh0
ind = np.argsort(tr_scores_ns)

plt.plot(tr_scores[ind])
plt.plot(tr_scores_ns[ind])
#plt.plot(tr_scores_rh0[ind])