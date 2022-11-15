import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random

def TSNE_kmeans(X_tr, X_sg_val, tr_scores, sg_scores, tr_losses, sg_losses, kmeans, n_clusters, DO_TSNE_CENTROIDS, TSNE_scores):
     max_score=np.quantile(np.concatenate([tr_scores, sg_scores]), 0.99)#max(np.max(tr_scores), np.max(sg_scores))
     min_score=np.quantile(np.concatenate([tr_scores, sg_scores]), 0.01)#min(np.min(tr_scores), np.min(sg_scores))
     
     tr_scores_nrm=tr_scores#-min_score)/(max_score-min_score)
     sg_scores_nrm=sg_scores#-min_score)/(max_score-min_score)
     
    # %%%%%%%%%%%%%%%%%%%%%%%
     if n_clusters>=10:
         cl1, cl2 = 1, 9
         plt.figure("cluster{:}_{:}scores".format(cl1, cl2))
         plt.scatter((np.sqrt(tr_losses[kmeans.labels_==cl1, cl1]-np.min(tr_losses[:, cl1])))*2**0.5, 
                     (np.sqrt(tr_losses[kmeans.labels_==cl1, cl2]-np.min(tr_losses[:, cl2])))*2**0.5,
                     s=10, 
                     c=tr_scores[kmeans.labels_==cl1],
                     label="cluster {}, {}".format(cl1+1, np.sum(kmeans.labels_==cl1)),
                      cmap="turbo", vmin=min_score, vmax=max_score)
         plt.xlabel("cluster {:} dist from edge in sigma".format(cl1+1))
         plt.ylabel("cluster {:} dist from edge in sigma".format(cl2+1))
         plt.legend()
         #plt.scatter(bg_losses[test_bg_labels==cl2, cl1 ], bg_losses[test_bg_labels==cl2, cl2])
     #%%%%%%%%%%%%%%%%%%%%%%%%%
     plt.rcParams.update({'font.size': 22})
     n_sig=1000
     n_bg=1000
     random.seed(a=10, version=2)
     IDs_TSNE=np.random.randint(0, X_tr.shape[0]-1, n_bg, )
     IDs_TSNE_sig=np.random.randint(0, X_sg_val.shape[0]-1, n_sig, )
     if DO_TSNE_CENTROIDS:
         centoids=kmeans.cluster_centers_
         labels_TSNE=np.concatenate((kmeans.labels_[IDs_TSNE], -1*np.ones(n_sig), -2*np.ones(n_clusters)))
         if X_tr.shape[1]==2:
             Y =np.concatenate([X_tr[IDs_TSNE], X_sg_val[IDs_TSNE_sig], centoids])
         else:
             Y = TSNE(n_components=2, n_iter=1000, random_state=0).fit_transform(np.concatenate([X_tr[IDs_TSNE], X_sg_val[IDs_TSNE_sig], centoids]))
     else:
         labels_TSNE=np.concatenate((kmeans.labels_[IDs_TSNE], -1*np.ones(n_sig)))
         if X_tr.shape[1]==2:
             Y =np.concatenate([X_tr[IDs_TSNE], X_sg_val[IDs_TSNE_sig]])
         else:
             Y = TSNE(n_components=2, n_iter=1000, random_state=0).fit_transform(np.concatenate([X_tr[IDs_TSNE], X_sg_val[IDs_TSNE_sig]]))
     plt.figure(num="TSNE", figsize=(10, 10))

     u_labels = np.unique(kmeans.labels_)
     for i in u_labels:
         if n_clusters<=10:
             plt.scatter(Y[labels_TSNE==i, 0], Y[labels_TSNE==i, 1], label=i+1)
         else:
             plt.scatter(Y[labels_TSNE==i, 0], Y[labels_TSNE==i, 1])
     #plt.scatter(Y[labels_TSNE==-1, 0], Y[labels_TSNE==-1, 1], label="signal", marker="x")
     if DO_TSNE_CENTROIDS:
         plt.scatter(Y[labels_TSNE==-2, 0], Y[labels_TSNE==-2, 1], label="centroids", marker="X", color="black", s=70)
     #plt.legend()
     if X_tr.shape[1]==2:
             ax=plt.gca()
             ax.axis('equal')
     
     if TSNE_scores:
         plt.figure(num="TSNE_scores", figsize=(12, 10))
         plt.scatter(Y[labels_TSNE>=0, 0], Y[labels_TSNE>=0, 1], c=tr_scores_nrm[IDs_TSNE],  cmap="turbo", marker="o", vmin=min_score, vmax=max_score)
         plt.colorbar()
         plt.scatter(Y[labels_TSNE==-1, 0], Y[labels_TSNE==-1, 1], c=sg_scores_nrm[IDs_TSNE_sig],  marker="x", cmap="turbo", vmin=min_score, vmax=max_score)
         if X_tr.shape[1]==2:
             ax=plt.gca()
             ax.axis('equal')
             plt.xlim((-30, 30))
             plt.ylim((-5, 10))
         if DO_TSNE_CENTROIDS:
             plt.scatter(Y[labels_TSNE==-2, 0], Y[labels_TSNE==-2, 1], label="centroids", marker="X", color="black", s=90)
         plt.legend()

         
         plt.figure(num="TSNE_scores_tra", figsize=(12, 10))
         plt.scatter(Y[labels_TSNE>=0, 0], Y[labels_TSNE>=0, 1], c=tr_scores_nrm[IDs_TSNE], label="training", cmap="turbo", marker="o", vmin=min_score, vmax=np.max(tr_scores_nrm[IDs_TSNE]))
         if DO_TSNE_CENTROIDS:
             plt.scatter(Y[labels_TSNE==-2, 0], Y[labels_TSNE==-2, 1], label="centroids", marker="X", color="black", s=90)
         plt.legend()
         
         plt.figure(num="TSNE_scores_sig", figsize=(12, 10))
         plt.scatter(Y[labels_TSNE==-1, 0], Y[labels_TSNE==-1, 1], c=sg_scores_nrm[IDs_TSNE_sig], label="signal", marker="x", cmap="turbo", vmin=min_score, vmax=max_score)
         if DO_TSNE_CENTROIDS:
             plt.scatter(Y[labels_TSNE==-2, 0], Y[labels_TSNE==-2, 1], label="centroids", marker="X", color="black", s=90)
         plt.legend()
     print("done TSNE")