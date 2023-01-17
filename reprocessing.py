"""
A library of different preprocessings of the jet images 
"""

import numpy as np
from scipy.ndimage import gaussian_filter


def sum_1_norm(x, batch=True):
    """Normalise each image in a batch to sum 1 (summing over chanels as well!!!)
    or a single image

    Args:
        x (numpy.array): batch of images of formats (N, W, H, C), (N, W, H) or (N, W*H) or a single image
        batch (bool, optional): Specify if a given array is a batch or a single image (not clear from the dimensionality). Defaults to True.

    Returns:
        numpy.array: normalised images/image
    """
    if batch:
        if len(x.shape) == 4:  #
            return x / (np.sum(x, (1, 2, 3))).reshape((-1, 1, 1, 1))
        elif len(x.shape) == 3:
            return x / (np.sum(x, (1, 2))).reshape((-1, 1, 1))
        elif len(x.shape) == 2:
            return x / (np.sum(x, (1))).reshape((-1, 1))
        else:
            print("strange data format!")  # TODO add exception here
            return x
    else:
        return x / (np.sum(x))


def gaussian_smearing(x, sigma, batch=True):
    """Smear images wiht a gaussian kernel

    Args:
        x (numpy.array): batch of images of formats (N, W, H, C), (N, W, H) or a single image of formats (W, H, C), (W, H)
        sigma (float): standard deviation of a gaussian
        batch (bool, optional): Specify if a given array is a batch or a single image (not clear from the dimensionality). Defaults to True.

    Returns:
        numpy.array: smeared images/image
    """
    if batch:
        if len(x.shape) == 4:  #
            return gaussian_filter(x, sigma=[0, sigma, sigma, 0])
        elif len(x.shape) == 3:
            return gaussian_filter(x, sigma=[0, sigma, sigma])
    else:
        if len(x.shape) == 3:  #
            return gaussian_filter(x, sigma=[sigma, sigma, 0])
        elif len(x.shape) == 2:
            return gaussian_filter(x, sigma=[sigma, sigma])


def reweighting(x, n):
    if n == 0:
        return x != 0
    else:
        return x**n


def reproc_heavi(x):
    x[x > 0] = 1
    return sum_1_norm(x)


def reproc_log(x, l):
    x = l * x
    x = np.log(x + 1)
    return x / (np.sum(x, (1, 2, 3))).reshape((-1, 1, 1, 1))


class Reprocessing:
    def __init__(self, arg_str=None):  # TODO: add parsing with a name argument!
        if (arg_str is None) or arg_str == "none":
            self.name = "none"
            self.function = lambda x: x
            return

        arg_str.split()
        self.arg_str = arg_str
        arg_list = arg_str.split()
        self.name = "".join(arg_list)

    # Only works for a batch of images, add possibility to apply to a single image

    def __call__(self, x):
        arg_list = self.arg_str.split()
        arg_list.reverse()
        while arg_list != []:
            fun_char = arg_list.pop()
            if fun_char == "N":
                x = sum_1_norm(x)
            elif fun_char == "w":
                x = reweighting(x, (float)(arg_list.pop()))
            elif fun_char == "s":
                x = gaussian_smearing(x, (float)(arg_list.pop()))
        return x
