import numpy as np
import matplotlib.pyplot as plt
from utilities import d_ball_volume, dm1_sphere_area

def d_slopes_norm(x, mean, sigma, weight, d=1, dont_use_weights=False, dont_use_volume=False):
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
    
plt.axhline(0, color="black")
    
for d in [0, 1, 3, 5]:
    x=np.linspace(-4, 1, 5000)
    x1=np.abs(x-0)
    x2=np.abs(x-1.7)
    x3=np.abs(x+2)
    L1=d_slopes_norm(x1, 0.5, 0.1, 1, d=d, dont_use_weights=False, dont_use_volume=False)
    L2=d_slopes_norm(x2, 0.7, 0.14, 0, d=d, dont_use_weights=False, dont_use_volume=False)
    L3=d_slopes_norm(x3, 1, 0.2, 1, d=d, dont_use_weights=False, dont_use_volume=False)
    plt.plot(x, ((L1+L2+L3)/np.max(L1+L2+L3)), label="d={:}".format(d))
    
for d in [0, 1, 3, 5]:
    L1=d_slopes_norm(x1, 0.5, 0.1, 1, d=d, dont_use_weights=False, dont_use_volume=False)
    L2=d_slopes_norm(x2, 0.7, 0.14, 0, d=d, dont_use_weights=False, dont_use_volume=False)
    L3=d_slopes_norm(x3, 1, 0.2, 1, d=d, dont_use_weights=False, dont_use_volume=False)
    plt.plot(x[4650], (((L1+L2+L3)/np.max(L1+L2+L3))[4650]), "o", color="black")
    plt.plot(x[2500], (((L1+L2+L3)/np.max(L1+L2+L3))[2500]), "s", color="black")
    
plt.legend()
plt.ylabel("Likelihood/max(Likelihood)")
plt.xlabel("Axis in VS through two cluster centroids, distance units")
plt.annotate("$\mu_2$", [-2.05, -0.07])
plt.annotate("$\mu_1$", [-0.05, -0.07])
#plt.annotate("hiaerarchy", [-1.55, -0.07])
#plt.annotate("out-of-cluster", [0.65, -0.07])
plt.scatter([-2, 0], [0, 0], c="black", marker="x")
plt.ylim([-0.1, 1.1])

    
