import numpy as np
import random


def std_ignore_outliers(
    x, oulier_fraction=0.2, corerecting_factor=1.51, keepdims=False
):
    med = np.median(x, axis=0)
    x1 = np.abs(x - med)
    x2 = np.copy(x)
    q = np.quantile(x1, 1 - oulier_fraction, axis=0)
    x2[x1 > q] = np.nan
    return np.nanstd(x2, axis=0, keepdims=keepdims) * corerecting_factor


def mean_ignore_outliers(x, oulier_fraction=0.2, keepdims=False):
    med = np.median(x, axis=0)
    x1 = np.abs(x - med)
    x2 = np.copy(x)
    q = np.quantile(x1, 1 - oulier_fraction, axis=0)
    x2[x1 > q] = np.nan
    return np.nanmean(x2, axis=0, keepdims=keepdims)


def scaling_factor_robust_SD_MC(oulier_fraction):
    a = np.random.normal(size=(50, 10000))
    std1 = np.mean(np.std(a, axis=0))
    std2 = np.mean(
        std_ignore_outliers(
            a, oulier_fraction=oulier_fraction, corerecting_factor=1
        )
    )
    return std1 / std2
