import numpy as np
import matplotlib.pyplot as plt

def anomalousity_histogram(X, kmeans, scores, res, sufix="", max_sigma_inclusions=5, max_cluster_inclusions=3):
            dist = kmeans.transform(X)
            
            n_sigmas_inclusion_arr=np.arange(max_sigma_inclusions+1)
            n_clusters_inclusion_arr=np.arange(max_cluster_inclusions)
            inclusions=-np.sum(dist<res["means"], axis=1)
            for n_sigmas_inclusion in n_sigmas_inclusion_arr:
                n_sigma_inclusions=np.sum(dist<(res["means"]+res["sigmas"]*n_sigmas_inclusion), axis=1)
                inclusions+=(n_sigma_inclusions>0)*(inclusions==0)*n_sigmas_inclusion
            
            res["inclusions"+sufix]=n_sigmas_inclusion_arr

            in_n_clusters_all=[]
            in_n_sigma_all=[]
            indices=np.argsort(scores)
            inclusions_sort=inclusions[indices]
            n=len(X)//10
            for i in range(len(X)//n):
                inclusions_part=inclusions_sort[i*n:(i*n+n)]
                in_n_sigma=[]
                for n_sigmas_inclusion in n_sigmas_inclusion_arr:
                    in_n_sigma.append(np.sum(inclusions_part<=n_sigmas_inclusion)) 
                in_n_sigma_all.append(in_n_sigma)
                    
            for i in range(len(X)//n):
                inclusions_part=inclusions_sort[i*n:(i*n+n)]
                in_n_clusters=[]
                for n_sigmas_inclusion in n_sigmas_inclusion_arr:
                    in_n_clusters.append(np.sum(inclusions_part<-n_sigmas_inclusion)) 
                in_n_clusters_all.append(in_n_clusters)
                
            in_n_clusters_all=np.array(in_n_clusters_all)
            in_n_sigma_all=np.array(in_n_sigma_all)
            plt.figure("anomaly_types"+sufix)
            for n_cluster_inclusion in n_clusters_inclusion_arr:
                plt.plot(in_n_clusters_all[:, n_cluster_inclusion-1], label="in "+(str)(n_cluster_inclusion-1)+" clusters", linestyle="--")
            for n_sigmas_inclusion in n_sigmas_inclusion_arr:
                plt.plot(in_n_sigma_all[:, n_sigmas_inclusion], label=(str)(n_sigmas_inclusion)+"sigma inclusion")
            #plt.show()
            res["n_sigmas_inclusion_arr"+sufix]=n_sigmas_inclusion_arr
            res["in_n_sigma_all"+sufix]=in_n_sigma_all
            res["n_clusters_inclusion_arr"+sufix]=n_clusters_inclusion_arr
            res["in_n_clusters_all"+sufix]=in_n_clusters_all