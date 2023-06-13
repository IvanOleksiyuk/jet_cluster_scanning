import numpy as np


def squeeze_1d(x, f):
    x = x[: (len(x) // f) * f]
    x = x.reshape((-1, f))
    x = np.mean(x, axis=1)
    return x


def squeeze(x, f):
    if len(x.shape) == 1:
        x = x[: (len(x) // f) * f]
        x = x.reshape((-1, f))
        x = np.mean(x, axis=1)
    elif len(x.shape) == 2:
        xs = []
        for line in x:
            xs.append(squeeze_1d(line, f))
        x = np.stack(xs)
    return x
