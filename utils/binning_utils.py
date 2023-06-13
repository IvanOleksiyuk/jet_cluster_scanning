import numpy as np

def default_binning(W=100, lb=2600, rb=6000, steps=200):
    mjjmin_arr = np.linspace(lb, rb - W, steps)
    mjjmax_arr = mjjmin_arr + W
    binning = np.stack([mjjmin_arr, mjjmax_arr]).T
    return binning
