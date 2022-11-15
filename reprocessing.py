import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy
from scipy.ndimage import gaussian_filter

def sum_1_norm(x):
    if len(x.shape)==4:
        return x/(np.sum(x, (1, 2, 3))).reshape((-1, 1, 1, 1))
    elif len(x.shape)==3:
        return x/(np.sum(x, (1, 2))).reshape((-1, 1, 1))
    elif len(x.shape)==2:
        return x/(np.sum(x, (1))).reshape((-1, 1))
    else:
        print("strange data format!")
        return x

def gaussian_smearing(x, SIGMA):
    return gaussian_filter(x, sigma=[0, SIGMA, SIGMA, 0]) 

def fft2_abs(x):
    return np.abs(np.fft.fft2(x, axes=(-3, -2)))

def fft2_absm1(x):
    return 1-np.abs(np.fft.fft2(x, axes=(-3, -2)))

def reproc_none(x):
       return x

def reproc_none_x10(x):
       return x*10

def reproc_none_x100(x):
       return x*100

def reproc_none_x10000(x):
       return x*10000

def reproc_sq(x):
    return sum_1_norm(x**2)

def boundMSE(x):
    return sum_1_norm(x)**0.5

def reproc_sin(x):
       return np.sin(x*np.pi/2)/(np.sum(np.sin(x*np.pi/2), (1, 2, 3))).reshape((-1, 1, 1, 1))

def reproc_sqrt(x):
    return sum_1_norm(x**0.5)

def reproc_sqrt_nonorm(x):
    return sum_1_norm(x**0.5)

def reproc_4rt(x):
    return sum_1_norm(x**0.25)

def reproc_heavi(x):
       x[x>0]=1
       return sum_1_norm(x)

def reproc_log(x, l):
       x = l*x
       x = np.log(x+1)
       return x/(np.sum(x, (1, 2, 3))).reshape((-1, 1, 1, 1))

def reproc_log1000(x):
       x = 1000*x
       x = np.log(x+1)
       return x/(np.sum(x, (1, 2, 3))).reshape((-1, 1, 1, 1))

def reproc_log1000_nonorm(x):
       x = 1000*x
       x = np.log(x+1)
       return x/np.log(1000+1)

def reproc_4rt_nonorm(x):
       return (x)**0.25

def reproc_sqrt_nonorm(x):
       return (x)**0.5

def reproc_heavi_nonorm(x):
       x[x>0]=1
       return x

def reproc_mean_std(x, mean, std, eps):
       return (x-mean)/(std+eps)

def reproc_sqnorm(x):
       return (x)/np.sqrt(np.sum((x)**2, (1, 2, 3))).reshape((-1, 1, 1, 1))

def reproc_log1000_sqnorm(x):
       x = 1000*x
       x = np.log(x+1)
       return x/np.sqrt(np.sum(x**2, (1, 2, 3))).reshape((-1, 1, 1, 1))

def reproc_4rt_sqnorm(x):
       return (x)**0.25/np.sqrt(np.sum(x**0.5, (1, 2, 3))).reshape((-1, 1, 1, 1))

def reproc_sqrt_sqnorm(x):
       return (x)**0.5/np.sqrt(np.sum(x, (1, 2, 3))).reshape((-1, 1, 1, 1))

def reproc_heavi_sqnorm(x):
       x[x>0]=1
       return x/np.sqrt(np.sum(x**2, (1, 2, 3))).reshape((-1, 1, 1, 1))


def xImageOfOnes(a):
    image1=np.ones((40, 40, 1))
    return a[0]*image1

def leaveNbrightest(x, N):
    shape=x.shape
    y=np.partition(x.reshape((shape[0],shape[1]*shape[2]*shape[3])), -N, axis=1)[:, -N]
    #print(np.expand_dims(y, 1))
    #print(np.apply_along_axis(xImageOfOnes, -1, np.expand_dims(y, 1)))
    #print((x>=np.apply_along_axis(xImageOfOnes, -1, np.expand_dims(y, 1))))
    x=x*(x>=np.apply_along_axis(xImageOfOnes, -1, np.expand_dims(y, 1)))
    #print(x)
    return x

def leaveMtoNbrightest(x, N, M=1):
    shape=x.shape
    y=np.partition(x.reshape((shape[0],shape[1]*shape[2]*shape[3])), -N, axis=1)[:, -N]
    y2=np.partition(x.reshape((shape[0],shape[1]*shape[2]*shape[3])), -M, axis=1)[:, -M]
    #print(np.expand_dims(y, 1))
    #print(np.apply_along_axis(xImageOfOnes, -1, np.expand_dims(y, 1)))
    #print((x>=np.apply_along_axis(xImageOfOnes, -1, np.expand_dims(y, 1))))
    x=x*(x>=np.apply_along_axis(xImageOfOnes, -1, np.expand_dims(y, 1)))*(x<=np.apply_along_axis(xImageOfOnes, -1, np.expand_dims(y2, 1)))
    #print(x)
    return x

def reproc_names(reproc):
    if reproc==None:
        return "none"
    if reproc==reproc_none:
        return "none"
    if reproc==reproc_sqrt:
        return "sqrt"
    if reproc==reproc_4rt:
        return "4rt"
    if reproc==reproc_log1000:
        return "log1000"
    if reproc==reproc_heavi:
        return "heavi"
    return "invalid"

def postpr_names(reproc):
    if reproc==None:
        return ""
    if reproc==reproc_none:
        return ""
    if reproc==boundMSE:
        return "None"