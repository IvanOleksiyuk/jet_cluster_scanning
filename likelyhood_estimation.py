import numpy as np 
import copy
from utilities import d_ball_volume, dm1_sphere_area
import matplotlib
import matplotlib.pyplot as plt
import set_matplotlib_default as smd

def infinity_to_min_max(bg_scores, sg_scores, tr_scores):
    max_score=max(np.max(tr_scores[np.isfinite(tr_scores)],initial=0), np.max(sg_scores[np.isfinite(sg_scores)],initial=0), np.max(bg_scores[np.isfinite(bg_scores)],initial=0))
    min_score=min(np.min(tr_scores[np.isfinite(tr_scores)],initial=0), np.min(sg_scores[np.isfinite(sg_scores)],initial=0), np.min(bg_scores[np.isfinite(bg_scores)],initial=0))
    #replacing inf with max*1.1 has no effect on the tagging performance (inf was 0 befor -log)
    if max_score>0:   
        bg_scores[bg_scores==np.inf]=max_score*1.1
        sg_scores[sg_scores==np.inf]=max_score*1.1
        tr_scores[tr_scores==np.inf]=max_score*1.1
    else:
        bg_scores[bg_scores==np.inf]=max_score*0.9
        sg_scores[sg_scores==np.inf]=max_score*0.9
        tr_scores[tr_scores==np.inf]=max_score*0.9
    if min_score>0:
        bg_scores[bg_scores==np.NINF]=min_score*0.9
        sg_scores[sg_scores==np.NINF]=min_score*0.9
        tr_scores[tr_scores==np.NINF]=min_score*0.9
    else:
        bg_scores[bg_scores==np.NINF]=min_score*1.1
        sg_scores[sg_scores==np.NINF]=min_score*1.1
        tr_scores[tr_scores==np.NINF]=min_score*1.1


def likelyhood_estimation_dim_Gauss(kmeans, crop, k, X_tr, X_bg_val, X_sg_val, density_function, LOG_LIKELIHOOD=True):
    dist_tr = kmeans.transform(X_tr)
    dist_bg_val = kmeans.transform(X_bg_val)
    dist_sg_val = kmeans.transform(X_sg_val)
    means=[]
    sigmas=[]
    weights=[]
    for i in range(k):
        dist=dist_tr[kmeans.labels_==i, i]
        means.append(np.mean(dist))
        sigmas.append(np.std(dist))
        weights.append(len(dist)/crop)

    means=np.array(means)
    sigmas=np.array(sigmas)
    weights=np.array(weights)
    s=sigmas/(2**0.5)
    a=means**2/s**2
    b=weights*crop
    b.sort()
    print(b)
    print("s", s)
    print("a", a)
    a=np.median(a)
    print("a_deside", a)
    part_L_bg=np.zeros(dist_bg_val.shape)
    part_L_sg=np.zeros(dist_sg_val.shape)
    part_L_tr=np.zeros(dist_tr.shape)
    bg_L, sg_L, tr_L=0, 0, 0
    for i in range(k):
        part_L_tr[:, i]=density_function(dist_tr[:, i], a, s[i], weights[i])
        part_L_bg[:, i]=density_function(dist_bg_val[:, i], a, s[i], weights[i])
        part_L_sg[:, i]=density_function(dist_sg_val[:, i], a, s[i], weights[i])
        bg_L-=part_L_bg[:, i]
        sg_L-=part_L_sg[:, i]
        tr_L-=part_L_tr[:, i]
    if LOG_LIKELIHOOD:
        tr_losses = -np.log(part_L_tr)
        bg_losses = -np.log(part_L_bg)
        sg_losses = -np.log(part_L_sg)
        bg_scores = -np.log(-bg_L)
        sg_scores = -np.log(-sg_L)
        tr_scores = -np.log(-tr_L)
        
        infinity_to_min_max(bg_scores, sg_scores, tr_scores)
        
        infinity_to_min_max(bg_losses, sg_losses, tr_losses)
    else:
        tr_losses = part_L_tr
        bg_losses = part_L_bg
        sg_losses = part_L_sg
        bg_scores = bg_L
        sg_scores = sg_L
        tr_scores = tr_L
    return tr_scores, bg_scores, sg_scores, tr_losses, bg_losses, sg_losses

def estimate_dim_uniform_point(dist_tr, cluster_i, r_i, c=1.1):
    """

    Parameters
    ----------
    dist_tr : np.array
        Array of the distances of training points to the cluster centres
    cluster_i : int
        Index of a cluster
    r_i : float
        distance to the cluster at whicj to find d(r)
    c : float, optional
        step size. The default is 1.1.

    Returns
    -------
    dim : float
        effective dimensionality of the dataset at scale r_i looking from the centre of cluster_i.
        See paper for formula
    """
    
    dists=dist_tr[:, cluster_i] #take distances of all points to a given cluster centre
    N_1=np.sum([dists<=r_i]) #find the number of points inside the ball of radius r_i
    N_2=np.sum([dists<=r_i*c]) #find the number of points inside the ball of radius r_i*c
    dim=np.log(N_2/N_1)/np.log(c) #calcualte the effective dimensionality as discussed in the paper
    return dim 

def likelyhood_estimation_dim_Uniform(kmeans, 
                                      crop, 
                                      k, X_tr, X_bg_val, X_sg_val, 
                                      density_function, 
                                      LOG_LIKELIHOOD=True, 
                                      res=None, 
                                      d=None,
                                      PLOT_DIM=True):
    #if d==None: density_function will get no dimension as a last argument
    #if type(d)=int: density_function will get d as this argument
    #if d=="med": density_function will get med(d_i) as this argument
    #if d=="ind": density_function will get d_i as this argument
    
    #find all the point-to-cluster distances
    dist_tr = kmeans.transform(X_tr)
    dist_bg_val = kmeans.transform(X_bg_val)
    dist_sg_val = kmeans.transform(X_sg_val)
    means=[]    #list of rho_i of the clusters
    sigmas=[]   #list of sigma_i of the clusters
    weights=[]  #list of N_i/N_tot of the clusters
    dims=[]     #list of dim_i of each cluster
    for i in range(k):
        dist=dist_tr[kmeans.labels_==i, i] #find distances to cluster i of points assigned to cluster i
        if len(dist)<=2: #for the case if clusteer has exactrly 1 point we set this mixture to be 0
            means.append(1)
            sigmas.append(1)
            weights.append(0)
        else:
            means.append(np.mean(dist))  #calculate rho_i
            sigmas.append(np.std(dist))  #calculate sigma_i
            weights.append(len(dist)/crop) #calculate N_i/N_tot
            if type(d)==str: #If d is not set by user find the effective dimensions d_i for each cluster
                dims.append(estimate_dim_uniform_point(dist_tr, i, means[-1], c=1.1))
    
    #plot for illustrating the dimension estimation for clusters
    if PLOT_DIM and type(d)==str:
        matplotlib.rcParams.update({'font.size': 14})
        plt.figure("dimensions", figsize=(8, 5))
        plt.xlabel("$R$")
        plt.xscale("log")
        plt.ylabel("$d_i(R)$")
        axs=plt.gca()
        axs.grid( which='both', alpha=0.5 )
        c_plot=1.01
        for cluster_i in [0, 1, 2, 3, 4]:
            dists=dist_tr[:, cluster_i]
            min_d=np.min(dists)
            max_d=np.max(dists) 
            r_arr=min_d*c_plot**np.arange(0, (int)(np.floor(np.log(max_d/(min_d+10**-10))/np.log(c_plot)))+3)
            dim_arr=[]
            for r in r_arr:
                dim_arr.append(estimate_dim_uniform_point(dist_tr, cluster_i, r, c=1.1))
            dim_arr=np.array(dim_arr)
            vline_color = next(axs._get_lines.prop_cycler)['color']
            plt.plot(r_arr, dim_arr, c=vline_color)
            plt.axvline(means[cluster_i], color=vline_color)
    
    #Transform lists into arrays
    means=np.array(means)
    sigmas=np.array(sigmas)
    weights=np.array(weights)
    dims=np.array(dims)

    #pint out some results
    print("means")
    print(means)
    print("sigmas")
    print(sigmas)
    print("dims")
    print(dims)
    
    if d=="med":
        d=np.median(dims) #If d is not set by user find the effective dimension d for the dataset        
    print("d=", d)
    
    # Some tests 
    # TODO: delete this part
    """
    N_0=d_ball_volume(d, means) #O((sigma/mean)^0)
    N_1=dm1_sphere_area(d, means)*sigmas*np.sqrt(np.pi/2) #O((sigma/mean)^1)
    rat=N_1/N_0
    print("ratios", rat.sort())
    print("max ratios =", np.max(rat))
    print("max sig/mu =", np.max(sigmas/means))
    """
    
    if res!=None: #put some results in the res dictionary to use later
        res["means"]=means
        res["sigmas"]=sigmas
        res["dims"]=dims
        res["d_med"]=np.median(dims)
        res["d_mean"]=np.mean(dims)
        res["mu_max"]=np.max(means)
        res["mu_mean"]=np.mean(means)
        res["mu_median"]=np.median(means)
        res["mu_min"]=np.min(means)
        res["sig_max"]=np.max(sigmas)
        res["sig_min"]=np.min(sigmas)
        res["sig/mu_max"]=np.max(sigmas/means)
        #res["V_0/V_1_max"]=np.max(rat)

    #define arrays for partial likelyhoods
    part_L_bg=np.zeros(dist_bg_val.shape) 
    part_L_sg=np.zeros(dist_sg_val.shape)
    part_L_tr=np.zeros(dist_tr.shape)
    bg_L, sg_L, tr_L=0, 0, 0
    for i in range(k):
        #calcualte partial likelyhoods for each cluster
        if d==None:
            part_L_tr[:, i]=density_function(dist_tr[:, i], means[i], sigmas[i], weights[i])
            part_L_bg[:, i]=density_function(dist_bg_val[:, i], means[i], sigmas[i], weights[i])
            part_L_sg[:, i]=density_function(dist_sg_val[:, i], means[i], sigmas[i], weights[i])
        else:
            if type(d)!=str:
                part_L_tr[:, i]=density_function(dist_tr[:, i], means[i], sigmas[i], weights[i], d)
                part_L_bg[:, i]=density_function(dist_bg_val[:, i], means[i], sigmas[i], weights[i], d)
                part_L_sg[:, i]=density_function(dist_sg_val[:, i], means[i], sigmas[i], weights[i], d)  
            else:
                part_L_tr[:, i]=density_function(dist_tr[:, i], means[i], sigmas[i], weights[i], dims[i])
                part_L_bg[:, i]=density_function(dist_bg_val[:, i], means[i], sigmas[i], weights[i], dims[i])
                part_L_sg[:, i]=density_function(dist_sg_val[:, i], means[i], sigmas[i], weights[i], dims[i])
        #Subtract a prtial likelyhood from a total negative likelyhood
        bg_L-=part_L_bg[:, i]
        sg_L-=part_L_sg[:, i]
        tr_L-=part_L_tr[:, i]
    if LOG_LIKELIHOOD: #if one wants to use negative log likelyhood as a score
        # calculate the negative partial log likelyhoods (useful for analyesis)
        tr_losses = -np.log(part_L_tr)
        bg_losses = -np.log(part_L_bg)
        sg_losses = -np.log(part_L_sg)
        # Calculate the negative log likelyhoods
        bg_scores = -np.log(-bg_L)
        sg_scores = -np.log(-sg_L)
        tr_scores = -np.log(-tr_L)
        #resolve isues with infinities arrising from log(0)
        infinity_to_min_max(bg_scores, sg_scores, tr_scores)
        infinity_to_min_max(bg_losses, sg_losses, tr_losses)
    else: #if one wants to use negative likelyhood as a score
        tr_losses = part_L_tr
        bg_losses = part_L_bg
        sg_losses = part_L_sg
        bg_scores = bg_L
        sg_scores = sg_L
        tr_scores = tr_L
    #Both scores give the same tagging results as they are connected by a monotonic log function
    return tr_scores, bg_scores, sg_scores, tr_losses, bg_losses, sg_losses



def likelyhood_estimation_dim_Uniform_special(kmeans, 
                                      crop, 
                                      k, X_tr, X_bg_val, X_sg_val, 
                                      density_function, 
                                      LOG_LIKELIHOOD=True, 
                                      res=None, 
                                      d=None,
                                      PLOT_DIM=True):
    #if d==None: density_function will get no dimension as a last argument
    #if type(d)=int: density_function will get d as this argument
    #if d=="med": density_function will get med(d_i) as this argument
    #if d=="ind": density_function will get d_i as this argument
    
    #find all the point-to-cluster distances
    dist_tr = kmeans.transform(X_tr)
    dist_bg_val = kmeans.transform(X_bg_val)
    dist_sg_val = kmeans.transform(X_sg_val)
    means=[]    #list of rho_i of the clusters
    sigmas=[]   #list of sigma_i of the clusters
    weights=[]  #list of N_i/N_tot of the clusters
    dims=[]     #list of dim_i of each cluster
    for i in range(k):
        dist=dist_tr[kmeans.labels_==i, i] #find distances to cluster i of points assigned to cluster i
        means.append(np.mean(dist))  #calculate rho_i
        sigmas.append(np.std(dist))  #calculate sigma_i
        weights.append(len(dist)/crop) #calculate N_i/N_tot
        if type(d)==str: #If d is not set by user find the effective dimensions d_i for each cluster
            dims.append(estimate_dim_uniform_point(dist_tr, i, means[-1], c=1.1))
    
    #plot for illustrating the dimension estimation for clusters
    if PLOT_DIM and type(d)==str:
        matplotlib.rcParams.update({'font.size': 14})
        plt.figure("dimensions", figsize=(8, 5))
        plt.xlabel("$R$")
        plt.xscale("log")
        plt.ylabel("$d_i(R)$")
        axs=plt.gca()
        axs.grid( which='both', alpha=0.5 )
        c_plot=1.01
        for cluster_i in [0, 1, 2, 3, 4]:
            dists=dist_tr[:, cluster_i]
            min_d=np.min(dists)
            max_d=np.max(dists) 
            r_arr=min_d*c_plot**np.arange(0, (int)(np.floor(np.log(max_d/(min_d+10**-10))/np.log(c_plot)))+3)
            dim_arr=[]
            for r in r_arr:
                dim_arr.append(estimate_dim_uniform_point(dist_tr, cluster_i, r, c=1.1))
            dim_arr=np.array(dim_arr)
            vline_color = next(axs._get_lines.prop_cycler)['color']
            plt.plot(r_arr, dim_arr, c=vline_color)
            plt.axvline(means[cluster_i], color=vline_color)
    
    #Transform lists into arrays
    means=np.array(means)
    sigmas=np.array(sigmas)
    weights=np.array(weights)
    dims=np.array(dims)

    #pint out some results
    print("means")
    print(means)
    print("sigmas")
    print(sigmas)
    print("dims")
    print(dims)
    
    if d=="med":
        d=np.median(dims) #If d is not set by user find the effective dimension d for the dataset        
    print("d=", d)
    
    # Some tests 
    # TODO: delete this part
    """
    N_0=d_ball_volume(d, means) #O((sigma/mean)^0)
    N_1=dm1_sphere_area(d, means)*sigmas*np.sqrt(np.pi/2) #O((sigma/mean)^1)
    rat=N_1/N_0
    print("ratios", rat.sort())
    print("max ratios =", np.max(rat))
    print("max sig/mu =", np.max(sigmas/means))
    """
    
    if res!=None: #put some results in the res dictionary to use later
        res["means"]=means
        res["sigmas"]=sigmas
        res["dims"]=dims
        res["d_med"]=np.median(dims)
        res["d_mean"]=np.mean(dims)
        res["mu_max"]=np.max(means)
        res["mu_mean"]=np.mean(means)
        res["mu_median"]=np.median(means)
        res["mu_min"]=np.min(means)
        res["sig_max"]=np.max(sigmas)
        res["sig_min"]=np.min(sigmas)
        res["sig/mu_max"]=np.max(sigmas/means)
        #res["V_0/V_1_max"]=np.max(rat)
    ind=np.argmin(means)
    N_0=d_ball_volume(d, means[ind]) #O((sigma/mean)^0)
    N_1=dm1_sphere_area(d, means[ind])*sigmas[ind]*np.sqrt(np.pi/2) #O((sigma/mean)^1)
    N=N_0+N_1
    #define arrays for partial likelyhoods
    part_L_bg=np.zeros(dist_bg_val.shape) 
    part_L_sg=np.zeros(dist_sg_val.shape)
    part_L_tr=np.zeros(dist_tr.shape)
    bg_L, sg_L, tr_L=0, 0, 0
    for i in range(k):
        #calcualte partial likelyhoods for each cluster
        if d==None:
            part_L_tr[:, i]=density_function(dist_tr[:, i], means[i], sigmas[i], weights[i]/N)
            part_L_bg[:, i]=density_function(dist_bg_val[:, i], means[i], sigmas[i], weights[i]/N)
            part_L_sg[:, i]=density_function(dist_sg_val[:, i], means[i], sigmas[i], weights[i]/N)
        else:
            if type(d)!=str:
                part_L_tr[:, i]=density_function(dist_tr[:, i], means[i], sigmas[i], weights[i]/N, d)
                part_L_bg[:, i]=density_function(dist_bg_val[:, i], means[i], sigmas[i], weights[i]/N, d)
                part_L_sg[:, i]=density_function(dist_sg_val[:, i], means[i], sigmas[i], weights[i]/N, d)  
            else:
                part_L_tr[:, i]=density_function(dist_tr[:, i], means[i], sigmas[i], weights[i]/N, dims[i])
                part_L_bg[:, i]=density_function(dist_bg_val[:, i], means[i], sigmas[i], weights[i]/N, dims[i])
                part_L_sg[:, i]=density_function(dist_sg_val[:, i], means[i], sigmas[i], weights[i]/N, dims[i])
        #Subtract a prtial likelyhood from a total negative likelyhood
        bg_L-=part_L_bg[:, i]
        sg_L-=part_L_sg[:, i]
        tr_L-=part_L_tr[:, i]
    if LOG_LIKELIHOOD: #if one wants to use negative log likelyhood as a score
        # calculate the negative partial log likelyhoods (useful for analyesis)
        tr_losses = -np.log(part_L_tr)
        bg_losses = -np.log(part_L_bg)
        sg_losses = -np.log(part_L_sg)
        # Calculate the negative log likelyhoods
        bg_scores = -np.log(-bg_L)
        sg_scores = -np.log(-sg_L)
        tr_scores = -np.log(-tr_L)
        #resolve isues with infinities arrising from log(0)
        infinity_to_min_max(bg_scores, sg_scores, tr_scores)
        infinity_to_min_max(bg_losses, sg_losses, tr_losses)
    else: #if one wants to use negative likelyhood as a score
        tr_losses = part_L_tr
        bg_losses = part_L_bg
        sg_losses = part_L_sg
        bg_scores = bg_L
        sg_scores = sg_L
        tr_scores = tr_L
    #Both scores give the same tagging results as they are connected by a monotonic log function
    return tr_scores, bg_scores, sg_scores, tr_losses, bg_losses, sg_losses

def likelyhood_estimation(kmeans, 
                        crop, 
                        k, X_tr, X_bg_val, X_sg_val, 
                        density_function, 
                        LOG_LIKELIHOOD=True, 
                        res=None, 
                        PLOT_DIM=True):
    return likelyhood_estimation_dim_Uniform(kmeans, 
                                      crop, 
                                      k, X_tr, X_bg_val, X_sg_val, 
                                      density_function, 
                                      LOG_LIKELIHOOD=True, 
                                      res=None, 
                                      d=None,
                                      PLOT_DIM=True)