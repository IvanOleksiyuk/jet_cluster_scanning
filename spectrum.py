import numpy as np
import copy


class Spectra:
    def __init__(self, x, y, err=None):
        self.x = x
        self.y = y  # Has to be two dimensional where axis=0 loopes through all spectra
        if err is None:
            self.err = np.sqrt(y)  # take poisson error
            self.poisson = True
        else:
            self.err = err
            self.false = True

    def copy(self):
        return copy.deepcopy(self)

    # None of the following methods showld modify x, or y of this class but only return copy with modifications
    def scale(self, scale):
        y = self.y * scale
        err = self.err * scale
        x = self.x
        return Spectra(x, y, err)

    def norm(self):
        return self.scale(1 / np.sum(self.y, axis=1, keepdims=True))

    def max_norm(self):
        return self.scale(1 / np.max(self.y, axis=1, keepdims=True))

    def num_der(self):
        y = (self.y[1:] - self.y[:-1]) / 2
        x = (self.x[1:] + self.x[:-1]) / 2
        err = 0  # correct this
        return Spectra(x, y, err)

    def sum_sp(self):
        y = np.sum(self.y, axis=0, keepdims=True)
        x = self.x
        if self.poisson:
            return Spectra(x, y)
        else:
            raise Exception("Why would you summ up non-poisson spectra?")
