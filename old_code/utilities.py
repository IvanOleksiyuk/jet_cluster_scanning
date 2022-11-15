import numpy as np
import pickle
from scipy.special import gamma

def standardize(X, standard=None):
    if standard==None:
        m=np.mean(X, axis=0)
        s=np.std(X, axis=0)
        standard=(m, s)

    X-=standard[0]
    X[:, standard[1]>0]/=standard[1][standard[1]>0]
    return standard

def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def find_nearest_idx_above(array, value):
    array = np.asarray(array)
    a=np.abs(array - value)
    a[(array-value)<0]=np.inf
    idx = a.argmin()
    return idx

def latex_table(arr, caption="", label=""):
    s=""
    (m, n) = arr.shape
    s+="\\begin{table} \n"
    s+="\\centering"
    s+="\\begin{tabular}{|"+"c|"*n+" } \n"
    s+="\\hline \n"
    for i in range(m):
        for j in range(n):
            s+=arr[i, j]+"&"
        s=s[:-1]
        s+="\\\\ \n"
        s+="\\hline \n"
    s+="\\end{tabular} \n"
    s+="\\caption{"
    s+=caption
    s+="}\n"
    s+="\\label{"
    s+=label
    s+="}\n"
    s+="\\end{table} \n"
    #s+="cell1 & cell2 & cell3 \\\\ \n" 
    return s
    
def inveB_at_eS(fpr, tpr, eS):
    return 1/fpr[find_nearest_idx(tpr, eS)]

def eS_at_inveB(fpr, tpr, ieB):
    return tpr[find_nearest_idx(fpr, 1/ieB)]
        
def get_metric_from_res(res, metric, mod=""):
    if metric[:3]=="AUC":
        return res[metric+mod]
    elif metric[:6]=="ieB@eS":
        return inveB_at_eS(res["fpr"+mod], res["tpr"+mod], (float)(metric[6:]))
    elif metric[:6]=="eS@ieB":
        return eS_at_inveB(res["fpr"+mod], res["tpr"+mod], (float)(metric[6:]))

def get_metric_from_char(char_path, metric, form=None):
    res=pickle.load(open(char_path+"/res.pickle", "rb"))
    if metric=="eB0.2":
        eps_s=0.2
        metric_val=1/res["fpr"][find_nearest_idx(res["tps"], eps_s)]
    else:
        metric_val=res[metric]
    if form==None:
        return metric_val
    else:
        return form.format(metric_val)
    
def d_ball_volume(d, R):
    return (np.pi**(d/2))/gamma(d/2+1)*R**d

def dm1_sphere_area(d, R):
    if d==0:
        return 0
    else:
        return 2*(np.pi**(d/2))/gamma(d/2)*R**(d-1)
    