import numpy as np

def MLSnormal_positiv(bkg, sig):
    return np.max((sig-bkg)/np.sqrt(bkg))

def chi_square(bkg, sig):
    return np.sum((sig-bkg)**2/bkg)