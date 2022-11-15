import numpy as np
import pickle
import matplotlib.pyplot as plt

def g_r_dist(r, d, s=1, max_norm=True):
    N=1
    y=N*r**(d-1)*np.e**(-r**2/(s**2*2))
    if max_norm:
        return y/np.max(y)
    else:
        return y

def mc_curve()

x=np.linspace(0, 6, 1000)
plt.figure()
plt.grid("minor")
plt.plot(x, g_r_dist(x, 5), label="d=5, s=1")
plt.xlabel("r")
plt.legend()
plt.savefig("plots/multidim_Gaussian_r_dist.png")