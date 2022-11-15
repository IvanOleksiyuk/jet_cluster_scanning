import numpy as np
import matplotlib.pyplot as plt
def plot_roc(path):
    roc = np.load(path)
    plt.plot(roc[:,1], 1/(roc[:,0]+1e-16))
    plt.xlabel("$\epsilon_s$")
    plt.ylabel("$\epsilon_b^{-1}$")
    plt.yscale('log')
    plt.ylim((1,1000))
    plt.xlim((0,1))
    plt.show()
    
plot_roc("other_ROC/inn_roc_aachen_cubrt.npy")