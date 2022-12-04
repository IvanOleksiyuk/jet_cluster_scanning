import numpy as np
import copy


class Spectrum:
    def __init__(self, x, y):
        self.x
        self.y

    def copy(self):
        return Spectrum(x.copy(), y.copy())

    def norm(self):
        self.y = self.y / np.sum(self.y)

    def max_norm(self):
        self.y = self.y / np.max(self.y)

    def num_der(self):
        self.y = (self.y[1:] - self.y[:-1]) / 2
        self.x = (self.x[1:] + self.x[:-1]) / 2


class Spectra_simple:
    def __init__(self, x, y):
        self.spectra = []
        for line in y:
            self.spectra.append(Spectrum(x, y))

    def copy(self):
        return copy.deepcopy(self)

    def norm(self):
        self.y = self.y / np.sum(self.y)

    def max_norm(self):
        self.y = self.y / np.max(self.y)

    def num_der(self):
        self.y = (self.y[1:] - self.y[:-1]) / 2
        self.x = (self.x[1:] + self.x[:-1]) / 2
