import matplotlib.pyplot as plt
import numpy
from sklearn.manifold import TSNE
import set_matplotlib_default as smd


def CS_TSNE(num_der_counts_windows, labels, save_path):
    X_embedded = TSNE().fit_transform(num_der_counts_windows)
    plt.figure()
    plt.grid()
    plt.plot(
        X_embedded[:, 0][labels == 1],
        X_embedded[:, 1][labels == 1],
        ".",
        color="red",
    )
    plt.plot(
        X_embedded[:, 0][labels == 0],
        X_embedded[:, 1][labels == 0],
        ".",
        color="blue",
    )
    plt.xlabel("embedding dim 0")
    plt.ylabel("embedding dim 1")
    plt.savefig(save_path + "eval/TSNE.png")
