import numpy as np

def MLSnormal_oneside(bkg, sig):
    return np.max((bkg-sig)/np.sqrt(bkg))