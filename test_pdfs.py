import numpy as np 
from utilities import d_ball_volume, dm1_sphere_area
import matplotlib.pyplot as plt

def half_gaussian_norm(x, mean, sigma, weight, smear=1):
    sigma_=sigma*smear
    out = 1/(sigma_*(2*np.pi)**0.5)*np.exp(-(x-mean)**2/(2*sigma_**2))
    out[x<mean]=np.max(out)
    out/=2*(0.5+1*mean*np.max(out))
    return weight*out

def half_gaussian_norm_d(x, mean, sigma, weight, d=1, dont_use_weights=False):
    # the likelyhood is comnputed to the O((sigma/mean)^1)
    N_0=d_ball_volume(d, mean) #O((sigma/mean)^0)
    N_1=dm1_sphere_area(d, mean)*sigma*np.sqrt(np.pi/2) #O((sigma/mean)^1)
    N=N_0+N_1
    out = np.exp(-(x-mean)**2/(2*sigma**2))
    out[x<mean]=1
    out/=np.max(out)
    out/=N
    if dont_use_weights:
        return out
    else:
        return out*weight
    
def d_slopes_norm(x, mean, sigma, weight, d=1, dont_use_weights=False, dont_use_volume=False):
    # the likelyhood is comnputed to the O((sigma/mean)^1)
    N_in=d_ball_volume(d, mean)
    N_sl=dm1_sphere_area(d, mean)*sigma*np.sqrt(np.pi/2)
    N=N_in+N_sl
    out=np.zeros(x.shape)
    out[x>=mean] =((mean/x[x>=mean])**(d-1))* np.exp(-(x[x>=mean]-mean)**2/(2*sigma**2))
    out[x<mean]=1
    out/=np.max(out)
    if dont_use_volume==False:
        out/=N
    if dont_use_weights:
        return out
    else:
        return out*weight
    
t=2
n_steps=1000
x=np.linspace(0, t, n_steps)
plt.plot(x, half_gaussian_norm(x, 1, 0.3, 1))
plt.plot(x, d_slopes_norm(x, 1, 0.3, 1))


plt.plot(x, half_gaussian_norm_d(x, 1, 0.3, 1, d=0))
plt.plot(x, d_slopes_norm(x, 1, 0.3, 1, d=1, dont_use_volume=True))
print(np.sum(half_gaussian_norm(x, 1, 0.3, 1))*t/n_steps)
