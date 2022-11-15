import matplotlib.pyplot as plt
import numpy as np
from dataset_path_and_pref import dataset_path_and_pref, prepare_data

def hist_and_av(full_mean_diffs, 
                non_smeared_mean, 
                n_clusters, 
                bg_losses, 
                sg_losses, 
                bg_min, 
                sg_min, 
                DI, 
                preproc, 
                standard, 
                images, 
                test_bg_labels,
                test_sg_labels,
                counts_train,
                cluster_counts_bg,
                cluster_counts_sg,
                kmeans,
                X_bg_val):
    cols=3+full_mean_diffs+non_smeared_mean*2
    num=min(10, n_clusters)
    fig, ax = plt.subplots(num, cols, figsize=(cols*4, num*4.2), squeeze=False)
    fig.set_label("clusters")
    max_dist=max(np.max(bg_losses), np.max(sg_losses))
    max_min_dist=max(np.max(bg_min), np.max(sg_min))
    min_dist=min(0, np.min(bg_losses), np.min(sg_losses))
    min_min_dist=min(0, np.min(bg_min), np.min(sg_min))
    bins=np.linspace(min_dist, max_dist, 40)
    bins2=np.linspace(min_min_dist, max_min_dist, 40)
    
    X_bg_val_no_sm=prepare_data(DI["bg_val_data_path"], field=DI["bg_val_data_field"], preproc=preproc, SIGMA=0, standard=standard)
    for i in range(num):
        #mean image
        plt.sca(ax[i][2])
        plt.xticks([])
        plt.yticks([])
        if images:
            plt.imshow(kmeans.cluster_centers_[i].reshape((40, 40)))
        #histogram of distances
        plt.sca(ax[i][0])
        plt.yticks([])
        plt.hist(bg_losses[:, i], bins=bins, histtype='step')
        plt.hist(sg_losses[:, i], bins=bins, histtype='step')
        plt.hist(bg_min[test_bg_labels==i], bins=bins, histtype='step')
        plt.hist(sg_min[test_sg_labels==i], bins=bins, histtype='step')
        if i<num-1:
            plt.xticks([])
        plt.sca(ax[i][1])
        plt.yticks([])
        plt.title("tr"+str(counts_train[i])+" bg"+str(cluster_counts_bg[i])+" sg"+str(cluster_counts_sg[i]))
        plt.hist(bg_min[test_bg_labels==i], bins=bins2, histtype='step')
        plt.hist(sg_min[test_sg_labels==i], bins=bins2, histtype='step')
        if i<num-1:
            plt.xticks([])
        if images:
            curr_col=2
            if non_smeared_mean:
                curr_col+=1
                plt.sca(ax[i][curr_col])
                plt.xticks([])
                plt.yticks([])
                mat=kmeans.cluster_centers_[i].reshape((40, 40))-np.mean(X_bg_val, 0).reshape((40, 40))
                max_mat=max(np.max(mat), -np.min(mat))
                plt.imshow(mat, vmin=-max_mat, vmax=max_mat, cmap="bwr")
            
            if full_mean_diffs:
                curr_col+=1
                plt.sca(ax[i][curr_col])
                plt.xticks([])
                plt.yticks([])
                mat=np.mean(X_bg_val_no_sm[kmeans.predict(X_bg_val)==i], 0).reshape((40, 40))
                plt.imshow(mat)
                curr_col+=1
                plt.sca(ax[i][curr_col])
                plt.xticks([])
                plt.yticks([])
                mat=np.mean(X_bg_val_no_sm[kmeans.predict(X_bg_val)==i], 0).reshape((40, 40))-np.mean(X_bg_val_no_sm, 0).reshape((40, 40))
                max_mat=max(np.max(mat), -np.min(mat))
                plt.imshow(mat, vmin=-max_mat, vmax=max_mat, cmap="bwr")
                plt.colorbar()