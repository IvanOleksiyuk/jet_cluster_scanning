import numpy as np
import copy
from robust_estimators import std_ignore_outliers, mean_ignore_outliers

class Spectra:
    def __init__(self, x, y, err=None, poisson=True):
        self.x = x
        self.y = y  # Has to be two dimensional where axis=0 loopes through all spectra
        if err is None:
            self.err = np.sqrt(y)  # take poisson error
        else:
            self.err = err
        self.poisson=poisson

    def copy(self):
        return copy.deepcopy(self)

    # None of the following methods showld modify x, or y of this class but only return copy with modifications
    def scale(self, scale):
        y = self.y * scale
        err = self.err * scale
        x = self.x
        return Spectra(x, y, err, poisson=False)

    def sum_norm(self):
        return self.scale(1 / np.sum(self.y, axis=1, keepdims=True))

    def max_norm(self):
        return self.scale(1 / np.max(self.y, axis=1, keepdims=True))
        
    def num_der(self):
        y = (self.y[:, 1:] - self.y[:, :-1]) / 2
        x = (self.x[1:] + self.x[:-1]) / 2
        err = 0  # correct this
        return Spectra(x, y, err, poisson=False)

    def sum_sp(self):
        y = np.sum(self.y, axis=0, keepdims=True)
        x = self.x
        if self.poisson:
            return Spectra(x, y)
        else:
            raise Exception("Why would you summ up non-poisson spectra?")

    def mean_sp(self):
        y = np.mean(self.y, axis=0, keepdims=True)
        x = self.x
        err=np.sqrt(np.mean(self.err**2, axis=0, keepdims=True))
        if self.poisson:
            return Spectra(x, y, err, poisson=False)

    def std_sp(self):
        y = np.std(self.y, axis=0, keepdims=True)
        x = self.x
        err=0 #TODO tTHIS IS FALSE IMPROVE IT
        if self.poisson:
            return Spectra(x, y, err, poisson=False)

    def mean_sp_rob(self):
        y = mean_ignore_outliers(self.y, axis=0, keepdims=True)
        x = self.x
        err=0 #TODO THIS IS FALSE IMPROVE IT
        if self.poisson:
            return Spectra(x, y, err, poisson=False)

    def std_sp_rob(self):
        y = std_ignore_outliers(self.y, axis=0, keepdims=True)
        x = self.x
        err=0 #TODO THIS IS FALSE IMPROVE IT
        if self.poisson:
            return Spectra(x, y, err, poisson=False)

    def subtract_bg(self, background):
        y = self.y-background.y
        x = self.x
        err = np.sqrt(self.err**2 + background.err**2)
        return Spectra(x, y, err, poisson=False)


    