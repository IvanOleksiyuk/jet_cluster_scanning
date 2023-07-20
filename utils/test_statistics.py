import numpy as np

def MLSnormal_positiv(bkg, sig):
    return np.max((bkg-sig)/np.sqrt(bkg))