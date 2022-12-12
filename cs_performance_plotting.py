import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import set_matplotlib_default as smd
from robust_estimators import std_ignore_outliers, mean_ignore_outliers


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


def plot_sum_over_bins_dist(counts, bin_widths, labels, save_path):
    plt.figure()
    plt.grid()
    plt.hist(
        [
            np.sum(counts * bin_widths, axis=1)[labels == 0],
            np.sum(counts * bin_widths, axis=1)[labels == 1],
        ],
        bins=20,
        histtype="bar",
        stacked=True,
    )
    plt.xlabel("integral under the curve")
    plt.ylabel("cluster n")
    plt.savefig(save_path + "eval/curves_integrals.png", bbox_inches="tight")


def two_class_curves(
    bin_centers,
    counts,
    labels,
    figsize,
    suffix="",
    xlabel="window centre $m_{jj}$ [GeV]",
    ylabel="",
    save_file="",
    marker="",
    linestyle="-",
):
    plt.figure(figsize=figsize)
    plt.grid()
    lab1 = "anomalous clusters" + suffix
    lab2 = "non-anomalous clusters" + suffix
    for j in range(len(counts)):
        if labels[j]:
            plt.plot(
                bin_centers,
                counts[j],
                color="red",
                marker=marker,
                linestyle=linestyle,
                label=lab1,
            )
            lab1 = None
        else:
            plt.plot(
                bin_centers,
                counts[j],
                color="blue",
                marker=marker,
                linestyle=linestyle,
                label=lab2,
            )
            lab2 = None
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(
        save_file,
        bbox_inches="tight",
    )


def plot_mean_deviat(x, middle, deviat, lwd=1.5, color="lime", fillb=False):
    plt.plot(
        x,
        middle,
        color=color,
        label="mean and SD",
        linewidth=lwd,
    )
    plt.plot(
        x,
        middle - deviat,
        color=color,
        linewidth=lwd,
    )
    plt.plot(
        x,
        middle + deviat,
        color=color,
        linewidth=lwd,
    )
    if fillb:
        plt.fill_between(
            x,
            middle - deviat,
            middle + deviat,
            alpha=0.4,
            color=color,
        )


###OBSOLETE CODE BELOW !!!!!!!!!
###OBSOLETE CODE BELOW !!!!!!!!!
###OBSOLETE CODE BELOW !!!!!!!!!
###OBSOLETE CODE BELOW !!!!!!!!!
###OBSOLETE CODE BELOW !!!!!!!!!
###OBSOLETE CODE BELOW !!!!!!!!!
###OBSOLETE CODE BELOW !!!!!!!!!


def plot_all_scalings(
    window_centers,
    counts_windows,
    countmax_windows,
    countnrm_windows,
    save_path,
    figsize,
):
    # Does the same thing as the ClusterScanning.plot() method but with given spectra (used only to test if cs_performance is dojing the right thing)
    k = counts_windows.shape[0]
    min_allowed_count = 100
    min_min_allowed_count = 10

    plt.figure(figsize=figsize)
    plt.grid()
    for j in range(k):
        plt.plot(window_centers, counts_windows[j])
    plt.xlabel("window centre $m_{jj}$ [GeV]")
    plt.ylabel("$N_i(m_{jj})$")
    plt.savefig(save_path + "kmeans_ni_mjj_total.png", bbox_inches="tight")
    smallest_cluster_count_window = np.min(counts_windows, axis=0)
    for i in range(len(window_centers)):
        if smallest_cluster_count_window[i] < min_allowed_count:
            if smallest_cluster_count_window[i] < min_min_allowed_count:
                plt.axvline(window_centers[i], color="black", alpha=0.6)
            else:
                plt.axvline(window_centers[i], color="black", alpha=0.3)

    plt.savefig(
        save_path + "kmeans_ni_mjj_total_statAllowed.png",
        bbox_inches="tight",
    )

    plt.figure(figsize=figsize)
    plt.grid()
    for j in range(k):
        plt.plot(window_centers, countmax_windows[j])
    plt.xlabel("window centre $m_{jj}$ [GeV]")
    plt.ylabel("$N_i(m_{jj})/max(N_i(m_{jj}))$")
    plt.savefig(save_path + "kmeans_ni_mjj_max.png", bbox_inches="tight")
    smallest_cluster_count_window = np.min(counts_windows, axis=0)
    for i in range(len(window_centers)):
        if smallest_cluster_count_window[i] < min_allowed_count:
            if smallest_cluster_count_window[i] < min_min_allowed_count:
                plt.axvline(window_centers[i], color="black", alpha=0.6)
            else:
                plt.axvline(window_centers[i], color="black", alpha=0.3)

    plt.savefig(
        save_path + "kmeans_ni_mjj_max_statAllowed.png",
        bbox_inches="tight",
    )

    plt.figure(figsize=figsize)
    plt.grid()
    for j in range(k):
        plt.plot(window_centers, countnrm_windows[j])
    plt.xlabel("window centre $m_{jj}$ [GeV]")
    plt.ylabel("$N_i(m_{jj})/sum(N_i(m_{jj}))$")
    plt.savefig(save_path + "kmeans_ni_mjj_norm.png", bbox_inches="tight")
    smallest_cluster_count_window = np.min(counts_windows, axis=0)
    for i in range(len(window_centers)):
        if smallest_cluster_count_window[i] < min_allowed_count:
            if smallest_cluster_count_window[i] < min_min_allowed_count:
                plt.axvline(window_centers[i], color="black", alpha=0.6)
            else:
                plt.axvline(window_centers[i], color="black", alpha=0.3)

    plt.savefig(
        save_path + "kmeans_ni_mjj_norm_statAllowed.png",
        bbox_inches="tight",
    )
