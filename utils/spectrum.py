import numpy as np
import copy
from utils.robust_estimators import std_ignore_outliers, mean_ignore_outliers
from utils.lowpasfilter import butter_lowpass_filter
import scipy.signal


class Spectra:
    def __init__(self, x, y, err=None, poisson=True):
        """_summary_

        Args:
            x (np.ndarray): One dimensional array of x-coordinates of all the spectra
            y (np.ndarray): Two dimensional array, first index is the number of the spectrum the second is the index of y coordinate of this spectrum
            err (np.ndarray, optional): symmetric standard deviations for each point on each spectrum. If None the poisson error is taken
            poisson (bool, optional): tells if the error of current spectrum is poisson (namely equals square-root of spectrum or is approxiamtion of this)
        """
        self.x = x
        self.y = y  # Has to be two dimensional where axis=0 loopes through all spectra
        if err is None:
            self.err = np.sqrt(y)  # take poisson error
        else:
            self.err = err
        self.poisson = poisson

    def copy(self):
        return copy.deepcopy(self)

    # None of the following methods should modify x, or y of this class but only return copy with modifications
    def scale(self, scale):
        y = self.y * scale
        err = self.err * scale
        x = self.x
        return Spectra(x, y, err, poisson=False)

    def pscale(self, scale):
        y = self.y * scale
        err = self.err * np.sqrt(scale)
        x = self.x
        return Spectra(x, y, err, poisson=True)

    def sum_norm(self):
        return self.scale(1 / np.sum(self.y, axis=1, keepdims=True))

    def max_norm(self):
        return self.scale(1 / np.max(self.y, axis=1, keepdims=True))

    def num_der(self):
        y = (self.y[:, 1:] - self.y[:, :-1]) / 2
        x = (self.x[1:] + self.x[:-1]) / 2
        err = 0  # TODO tTHIS IS FALSE IMPROVE IT
        return Spectra(x, y, err, poisson=False)

    def sum_sp(self, choose=None):
        if choose is None:
            y = np.sum(self.y, axis=0, keepdims=True)
        else:
            y = np.sum(self.y[choose], axis=0, keepdims=True)
        x = self.x
        if self.poisson:
            return Spectra(x, y)
        else:
            raise Exception("Why would you summ up non-poisson spectra?")

    def mean_sp(self):
        y = np.mean(self.y, axis=0, keepdims=True)
        x = self.x
        err = np.sqrt(np.mean(self.err**2, axis=0, keepdims=True))
        return Spectra(x, y, err, poisson=False)

    def std_sp(self):
        y = np.std(self.y, axis=0, keepdims=True)
        x = self.x
        err = 0  # TODO tTHIS IS FALSE IMPROVE IT
        return Spectra(x, y, err, poisson=False)

    def mean_sp_rob(self):
        y = mean_ignore_outliers(self.y, keepdims=True)
        x = self.x
        err = 0  # TODO THIS IS FALSE IMPROVE IT
        return Spectra(x, y, err, poisson=False)

    def std_sp_rob(self):
        y = std_ignore_outliers(self.y, keepdims=True)
        x = self.x
        err = 0  # TODO THIS IS FALSE IMPROVE IT
        return Spectra(x, y, err, poisson=False)

    def subtract_bg(self, background):
        y = self.y - background.y
        x = self.x
        err = np.sqrt(self.err**2 + background.err**2)
        return Spectra(x, y, err, poisson=False)

    def subtract_sp(self, another):
        y = self.y - another.y
        x = self.x
        err = self.err
        return Spectra(x, y, err, poisson=False)

    def standardize(self):
        y = (self.y - np.mean(self.y, axis=0)) / np.std(self.y, axis=0)
        x = self.x
        err = 0  # TODO THIS IS FALSE IMPROVE IT
        return Spectra(x, y, err, poisson=False)

    def standardize_rob(self):  # TODO outlier factor should be a parameter
        y = (self.y - mean_ignore_outliers(self.y)) / std_ignore_outliers(self.y)
        x = self.x
        err = 0  # TODO THIS IS FALSE IMPROVE IT
        return Spectra(x, y, err, poisson=False)

    def butter_lowpas(self):
        order = 6
        fs = 1  # sample rate, Hz
        cutoff = 0.05  # desired cutoff frequency of the filter, Hz
        y = np.zeros_like(self.y)
        for i in range(len(self.y)):
            y[i] = butter_lowpass_filter(self.y[i], cutoff, fs, order)
        x = self.x
        err = 0  # TODO THIS IS FALSE IMPROVE IT
        return Spectra(x, y, err, poisson=False)

    def medfilt(self, window_size):
        y = np.zeros_like(self.y)
        for i in range(len(self.y)):
            y[i] = scipy.signal.medfilt(self.y[i], window_size)
        x = self.x
        err = 0  # TODO THIS IS FALSE IMPROVE IT
        return Spectra(x, y, err, poisson=False)

    def make_poiserr_another_sp_sumnorm(self, anothr_sp):
        self.err = np.sqrt(
            anothr_sp.y / np.sum(anothr_sp.y, axis=1) * np.sum(self.y, axis=1)
        )

    def make_poiserr_another_sp_sumnorm_where_low_stat(self, anothr_sp, low_stat=0):
        if not self.poisson:
            raise Exception("Self spectrum is not poisson")
        if not anothr_sp.poisson:
            raise Exception("Another spectrum is not poisson")
        if len(anothr_sp.y) == 1:
            for i in range(self.y.shape[0]):
                self.err[i][self.y[i] <= low_stat] = np.sqrt(
                    anothr_sp.y[0] / np.sum(anothr_sp.y[0]) * np.sum(self.y[i])
                )[self.y[i] <= low_stat]
        else:
            for i in range(self.y.shape[0]):
                self.err[i][self.y[i] <= low_stat] = np.sqrt(
                    anothr_sp.y[i] / np.sum(anothr_sp.y[i]) * np.sum(self.y[i])
                )[self.y[i] <= low_stat]

    def chisq_ndof(self, another=None):
        if len(self.y) > 1:
            axis = 1
        else:
            axis = None
        if another is None:
            return np.mean(self.y**2 / self.err**2, axis=axis)
        else:
            return np.mean(
                (self.y - another.y) ** 2 / (self.err**2 + another.err**2),
                axis=axis,
            )

    def max_diff_abs(self, another):
        return np.max(np.abs(self.y - another.y))

    def max_diff(self, another):
        return np.max(self.y - another.y)

    def max_dev_abs(self, another=None):
        if len(self.y) > 1:
            axis = 1
        else:
            axis = None
        if another is None:
            return np.max(np.abs(self.y) / self.err, axis=axis)
        else:
            return np.max(
                np.abs(self.y - another.y) / (self.err**2 + another.err**2) ** 0.5,
                axis=axis,
            )

    def max_dev(self, another=None):
        if len(self.y) > 1:
            axis = 1
        else:
            axis = None
        if another is None:
            return np.max(self.y / self.err, axis=axis)
        else:
            return np.max(
                (self.y - another.y) / (self.err**2 + another.err**2) ** 0.5,
                axis=axis,
            )
