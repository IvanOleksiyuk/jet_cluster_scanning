import numpy as np


def default_binning(W=100, lb=2600, rb=6000, steps=100):
    mjjmin_arr = np.linspace(lb, rb - W, steps)
    mjjmax_arr = mjjmin_arr + W
    window_centers = (mjjmin_arr + mjjmax_arr) / 2
    binning = np.stack([mjjmin_arr, mjjmax_arr]).T
    return binning


print(default_binning())

"""
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
    }
)

t = np.linspace(0.0, 1.0, 100)
s = np.cos(4 * np.pi * t) + 2

fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)
ax.plot(t, s)

ax.set_xlabel(r"$\mathbf{time, t (s)}$")
ax.set_ylabel(r"$Velocity, v (\phi/sec)$", fontsize=16)
ax.set_title(r"With rcParams", fontsize=16, color="r")

plt.show()
"""
