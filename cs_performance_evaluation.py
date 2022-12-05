import numpy as np
import random
import h5py
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN, Birch
from sklearn.manifold import TSNE
import time
import os
import reprocessing
import pickle
from scipy.ndimage import gaussian_filter
import copy
from lowpasfilter import butter_lowpass_filter
import datetime
import scipy.signal
import set_matplotlib_default as smd
import cs_performance_plotting as csp
import cluster_scanning
from robust_estimators import std_ignore_outliers, mean_ignore_outliers
from spectrum import Spectra
import matplotlib as mpl

mpl.rcParams.update(mpl.rcParamsDefault)

plt.close("all")


def num_der(x):
    return (x[1:] - x[:-1]) / 2


def middles(x):
    return (x[1:] + x[:-1]) / 2


def max_norm(x):
    return x / np.max(x)


def norm(x):
    return x / np.sum(x)


def squeeze(x, f):
    x = x[: (len(x) // f) * f]
    x = x.reshape((-1, f))
    x = np.mean(x, axis=1)
    return x


def cs_performance_evaluation(
    counts_windows=None,
    plotting=True,
    filterr="med",
    labeling="kmeans_der",  # , #,">5sigma"
    W=100,
    lb=2600,
    rb=6000,
    save_path="char/0kmeans_scan/26w60wk50ret0con0.05W100ste200rewsqrtsme1ID0/",
    save_load=True,
    verbous=True,
):
    os.makedirs(save_path, exist_ok=True)

    figsize = (6, 4.5)

    steps = counts_windows.shape[0]
    k = counts_windows.shape[1]

    Mjjmin_arr = np.linspace(lb, rb - W, steps)
    Mjjmax_arr = Mjjmin_arr + W
    window_centers = (Mjjmin_arr + Mjjmax_arr) / 2

    counts_windows = counts_windows.T

    sp_original = Spectra(window_centers, counts_windows)

    if verbous:
        print("minimal in training window:", np.min(sp_original.y[:, 0]))

    # produce maxnormed and normed versions of counts_windows

    sp_maxnorm = sp_original.max_norm()
    sp_sumnorm = sp_original.norm()

    countmax_windows = sp_maxnorm.y
    countnrm_windows = sp_sumnorm.y

    # total signal+background in windows
    count_sum = np.sum(sp_original.y, axis=0)
    count_sum_sigma = np.sqrt(count_sum)

    # normed vesions
    countmax_sum = count_sum / np.max(count_sum)
    countnrm_sum = norm(count_sum)

    # versions without respective background
    countnrm_windows_s = countnrm_windows - countnrm_sum

    baselinenrm = mean_ignore_outliers(countnrm_windows - countnrm_sum)
    countnrm_windows_sigma = std_ignore_outliers(countnrm_windows)
    countnrm_windows_std = (
        countnrm_windows - countnrm_sum - baselinenrm
    ) / countnrm_windows_sigma

    num_der_counts_windows = num_der(countmax_windows.T).T
    num_der_countmax_sum = num_der(countmax_sum)

    def filter_spectra(sp, filterr):
        if filterr == "lowpass":
            order = 6
            fs = 1  # sample rate, Hz
            cutoff = 0.05  # desired cutoff frequency of the filter, Hz
            for i in range(k):
                sp[i] = butter_lowpass_filter(sp[i], cutoff, fs, order)

        elif filterr == "med5":
            for i in range(k):
                sp[i] = scipy.signal.medfilt(sp[i], [5])

        elif filterr == "med":
            for i in range(k):
                sp[i] = scipy.signal.medfilt(sp[i], [7])

        elif filterr == "med11":
            for i in range(k):
                sp[i] = scipy.signal.medfilt(sp[i], [11])

    filter_spectra(num_der_counts_windows, filterr)

    if labeling == "kmeans_der":
        kmeans = KMeans(2)
        kmeans.fit(num_der_counts_windows)
        as_vectors = "derivatives"
        if np.sum(kmeans.labels_) < len(kmeans.labels_) // 2:
            labels = kmeans.labels_
        else:
            labels = 1 - kmeans.labels_

    elif labeling == "kmeans_cur":
        kmeans = KMeans(2)
        kmeans.fit(countmax_windows)
        as_vectors = "curves"
        if np.sum(kmeans.labels_) < len(kmeans.labels_) // 2:
            labels = kmeans.labels_
        else:
            labels = 1 - kmeans.labels_

    elif labeling == ">5sigma":
        labels = np.zeros(k)
        for j in range(k):
            if np.any(countnrm_windows_std[j] > 5):
                labels[j] = 1
            else:
                labels[j] = 0

    elif labeling == "random":
        labels = np.random.uniform(size=k) < 0.5

    # Estimate trial factor depending on the bin overlap
    tf = ((rb - lb) / W) / steps

    # Combine curves and
    anomaly_poor = np.sum(counts_windows[labels == 0], axis=0) * tf
    if np.any(labels):
        anomaly_rich = np.sum(counts_windows[labels == 1], axis=0) * tf
    else:
        anomaly_rich = anomaly_poor  # to give a 0 test_statistic

    if verbous:
        print("trial_factor", tf)

    #
    anomaly_poor_sigma = np.sqrt(anomaly_poor)
    anomaly_rich_sigma = np.sqrt(
        anomaly_poor / sum(anomaly_poor) * sum(anomaly_rich)
    )
    anomaly_poor_sigma = anomaly_poor_sigma * (
        np.sum(anomaly_rich) / np.sum(anomaly_poor)
    )
    anomaly_poor = anomaly_poor * (np.sum(anomaly_rich) / np.sum(anomaly_poor))

    def chi_square_ndof(
        anomaly_rich, anomaly_poor, anomaly_rich_sigma, anomaly_poor_sigma
    ):
        # calculate the test statistic (chi_square/degrees of freedom)
        return np.mean(
            (anomaly_rich - anomaly_poor) ** 2
            / (anomaly_rich_sigma**2 + anomaly_poor_sigma**2)
        )

    chisq_ndof = chi_square_ndof(
        anomaly_rich, anomaly_poor, anomaly_rich_sigma, anomaly_poor_sigma
    )

    n_dof = len(window_centers) / (
        W * steps / (Mjjmin_arr[-1] - Mjjmin_arr[0])
    )

    if verbous:
        print("n_dof=", n_dof)

    res = {}
    res["chisq_ndof"] = chisq_ndof
    res["ndof"] = n_dof
    res["deviation"] = (chisq_ndof - 1) * n_dof / np.sqrt(2 * n_dof)

    if plotting:
        plt.rcParams["lines.linewidth"] = 1

        os.makedirs(save_path + "eval/", exist_ok=True)

        plt.figure()
        plt.grid()
        plt.hist(np.sum(countmax_windows, axis=0), bins=20)
        plt.xlabel("integral under the curve")
        plt.ylabel("cluster n")
        plt.savefig(
            save_path + "eval/curves_integrals.png", bbox_inches="tight"
        )

        # subtracted curves max
        plt.figure(figsize=figsize)
        plt.grid()
        lab1 = "anomalous clusters \n method 1"
        lab2 = "non-anomalous clusters \n method 1"
        for j in range(k):
            if labels[j]:
                plt.plot(
                    window_centers,
                    countnrm_windows_s[j],
                    color="red",
                    label=lab1,
                )
                lab1 = None
            else:
                plt.plot(
                    window_centers,
                    countnrm_windows_s[j],
                    color="blue",
                    label=lab2,
                )
                lab2 = None

        ccccc = "lime"
        lwd = 1.5
        plt.plot(
            window_centers,
            np.mean(countnrm_windows_s, axis=0),
            color=ccccc,
            label="mean and SD",
            linewidth=lwd,
        )
        plt.plot(
            window_centers,
            np.mean(countnrm_windows_s, axis=0)
            - np.std(countnrm_windows, axis=0),
            color=ccccc,
            linewidth=lwd,
        )
        plt.plot(
            window_centers,
            np.mean(countnrm_windows_s, axis=0)
            + np.std(countnrm_windows, axis=0),
            color=ccccc,
            linewidth=lwd,
        )

        ccccc2 = "orange"
        plt.plot(
            window_centers,
            baselinenrm,
            color=ccccc2,
            label="outlier robust\n mean and SD",
            linewidth=lwd,
        )
        plt.plot(
            window_centers,
            baselinenrm - std_ignore_outliers(countnrm_windows),
            color=ccccc2,
            linewidth=lwd,
        )
        plt.plot(
            window_centers,
            baselinenrm + std_ignore_outliers(countnrm_windows),
            color=ccccc2,
            linewidth=lwd,
        )
        plt.fill_between(
            window_centers,
            baselinenrm - std_ignore_outliers(countnrm_windows),
            baselinenrm + std_ignore_outliers(countnrm_windows),
            alpha=0.4,
            color="lime",
        )
        plt.legend()
        plt.xlabel("window centre $m_{jj}$ [GeV]")
        plt.ylabel("$N_i(m_{jj})/sum(N_i(m_{jj}))$-background")
        plt.savefig(save_path + "eval/norm-toatal.png", bbox_inches="tight")

        # all curves standardized
        plt.figure(figsize=figsize)
        plt.grid()
        lab1 = "anomalous clusters \n method 1"
        lab2 = "non-anomalous clusters \n method 1"
        for j in range(k):
            if labels[j]:
                plt.plot(
                    window_centers,
                    countnrm_windows_std[j],
                    color="red",
                    label=lab1,
                )
                lab1 = None
            else:
                plt.plot(
                    window_centers,
                    countnrm_windows_std[j],
                    color="blue",
                    label=lab2,
                )
                lab2 = None

        plt.axhline(5, color="red", alpha=0.2)
        plt.xlabel("window centre $m_{jj}$ [GeV]")
        plt.ylabel("deviation in SD")
        plt.legend()
        plt.savefig(
            save_path + "eval/norm-toatal-sigmas.png", bbox_inches="tight"
        )

        # all curves standardized
        plt.figure(figsize=figsize)
        plt.grid()

        for j in range(k):
            if labels[j]:
                plt.plot(
                    window_centers,
                    countnrm_windows_std[j] * np.sqrt(countnrm_windows[j]),
                    color="red",
                )
            else:
                plt.plot(
                    window_centers,
                    countnrm_windows_std[j] * np.sqrt(countnrm_windows[j]),
                    color="blue",
                )

        plt.xlabel("window centre $m_{jj}$ [GeV]")
        plt.ylabel("deviation in SD")
        plt.savefig(
            save_path + "eval/norm-toatal-sigmas-special.png",
            bbox_inches="tight",
        )

        # all curves standardized squeezed
        plt.figure(figsize=figsize)
        plt.grid()
        sqf = 20
        for j in range(k):
            if labels[j]:
                plt.plot(
                    squeeze(window_centers, sqf),
                    squeeze(countnrm_windows_std[j], sqf),
                    color="red",
                )
            else:
                plt.plot(
                    squeeze(window_centers, sqf),
                    squeeze(countnrm_windows_std[j], sqf),
                    color="blue",
                )

        plt.xlabel("window centre $m_{jj}$ [GeV]")
        plt.ylabel("deviation in SD")
        plt.savefig(
            save_path + "eval/norm-toatal-sigmas-squeezed.png",
            bbox_inches="tight",
        )

        plt.figure(figsize=figsize)
        plt.grid()
        for j in range(k):
            if labels[j]:
                plt.plot(
                    np.mean(countnrm_windows_std[j] ** 2),
                    np.random.normal(),
                    "r.",
                )
            else:
                plt.plot(
                    np.mean(countnrm_windows_std[j] ** 2),
                    np.random.normal(),
                    "b.",
                )
        plt.xlabel("npd")
        plt.ylabel("random_variable")
        plt.savefig(save_path + "eval/npd_each.png", bbox_inches="tight")

        # all curves standardized points
        plt.figure()
        plt.grid()
        ys = []
        for j in range(k):
            if labels[j]:
                plt.plot(window_centers, countnrm_windows_std[j], "r.")
                ys.append(list(countnrm_windows_std[j]))
            else:
                plt.plot(window_centers, countnrm_windows_std[j], "b.")
                ys.append(list(countnrm_windows_std[j]))

        plt.xlabel("window centre $m_{jj}$ [GeV]")
        plt.ylabel("n_clusater/sum(n_cluster)-bg")
        plt.savefig(
            save_path + "eval/norm-toatal-sigmas-points.png",
            bbox_inches="tight",
        )

        # all curves standardized distribution
        fig, axs = plt.subplots(
            1, 2, figsize=(9, 3), gridspec_kw={"width_ratios": [2, 1]}
        )
        plt.sca(axs[0])
        ys = np.array(ys)
        ys = ys.reshape((-1,))
        xs = np.tile(window_centers, k)
        h, xedges, yedges, image = plt.hist2d(
            xs, ys, bins=(200, 40), cmap="gist_heat_r"
        )
        plt.xlabel("window centre $m_{jj}$ [GeV]")
        plt.ylabel("deviation in SD")
        plt.colorbar()
        plt.sca(axs[1])
        yy = (yedges[:-1] + yedges[1:]) / 2
        # for i in range(200):
        plt.step(yy, np.mean(h, axis=0), where="mid", color="darkorange")
        plt.xlabel("deviations in SD")
        plt.ylabel("avarage curve points per bin")
        plt.savefig(save_path + "eval/2d_hist.png", bbox_inches="tight")

        # subtracted curves norm
        plt.figure(figsize=figsize)
        plt.grid()

        for j in range(k):
            if labels[j]:
                plt.plot(
                    window_centers,
                    countmax_windows[j] - countmax_sum,
                    color="red",
                )
            else:
                plt.plot(
                    window_centers,
                    countmax_windows[j] - countmax_sum,
                    color="blue",
                )

        plt.plot(
            window_centers,
            np.mean(countmax_windows - countmax_sum, axis=0),
            color="lime",
        )
        plt.plot(
            window_centers,
            -std_ignore_outliers(countmax_windows),
            color="lime",
        )
        plt.plot(
            window_centers, std_ignore_outliers(countmax_windows), color="lime"
        )
        # plt.fill_between(window_centers, -np.std(countmax_windows, axis=0), np.std(countmax_windows, axis=0), alpha=0.2, color="lime")
        plt.fill_between(
            window_centers,
            np.mean(countmax_windows - countmax_sum, axis=0)
            - np.std(countmax_windows, axis=0),
            np.mean(countmax_windows - countmax_sum, axis=0)
            + np.std(countmax_windows, axis=0),
            alpha=0.4,
            color="lime",
        )
        plt.xlabel("window centre $m_{jj}$ [GeV]")
        plt.ylabel("n_clusater/max(n_cluster)-bg")
        plt.savefig(save_path + "eval/maxnorm-toatal.png", bbox_inches="tight")

        # all center curves with found labels
        plt.figure(figsize=figsize)
        plt.grid()
        for j in range(k):
            if labels[j]:
                plt.plot(window_centers, countmax_windows[j], color="red")
            else:
                plt.plot(window_centers, countmax_windows[j], color="blue")
        plt.xlabel("window centre $m_{jj}$ [GeV]")
        plt.ylabel("n_clusater/max(n_cluster)")
        plt.savefig(
            save_path + "eval/countmax_windows.png", bbox_inches="tight"
        )

        # all center curve derivatives with found labels
        plt.figure(figsize=figsize)
        plt.grid()
        lab1 = "anomalous clusters \n method 1"
        lab2 = "non-anomalous clusters \n method 1"
        for j in range(k):
            if labels[j]:
                plt.plot(
                    middles(window_centers),
                    num_der_counts_windows[j],
                    color="red",
                    label=lab1,
                )
                lab1 = None
            else:
                plt.plot(
                    middles(window_centers),
                    num_der_counts_windows[j],
                    color="blue",
                    label=lab2,
                )
                lab2 = None
        plt.xlabel("window centre $m_{jj}$ [GeV] between adjacent windows")
        plt.ylabel(r"$\Delta(N_i(m_{jj})/sum(N_i(m_{jj}))$)")
        plt.savefig(
            save_path + "eval/num_der_counts_windows.png", bbox_inches="tight"
        )

        # all center curve derivatives with found labels
        plt.figure(figsize=figsize)
        plt.grid()
        for j in range(k):
            if labels[j]:
                plt.plot(
                    middles(window_centers),
                    num_der_counts_windows[j] - num_der_countmax_sum,
                    color="red",
                )
            else:
                plt.plot(
                    middles(window_centers),
                    num_der_counts_windows[j] - num_der_countmax_sum,
                    color="blue",
                )
        plt.xlabel("window centre $m_{jj}$ [GeV]")
        plt.ylabel(r"$Median filtered \Delta[N_i(m_{jj})/sum(N_i(m_{jj}))]$")
        plt.legend()
        plt.savefig(
            save_path + "eval/num_der_counts_windows_-bg.png",
            bbox_inches="tight",
        )

        # TSNE
        csp.CS_TSNE(num_der_counts_windows, labels, save_path)

        # combinations
        plt.figure(figsize=figsize)
        plt.grid()
        sigmas = 2
        plt.fill_between(
            window_centers,
            anomaly_rich - anomaly_rich_sigma * sigmas,
            anomaly_rich + anomaly_rich_sigma * sigmas,
            alpha=0.2,
            color="red",
        )
        plt.fill_between(
            window_centers,
            anomaly_poor - anomaly_poor_sigma * sigmas,
            anomaly_poor + anomaly_poor_sigma * sigmas,
            alpha=0.2,
            color="blue",
        )

        plt.plot(
            window_centers,
            anomaly_poor,
            label=r"$\tilde{N}_l$normalised sum of anomaly poor clusters",
            color="blue",
        )
        # plt.plot(window_centers, anomaly_rich, label="sum of cluster 1 curves \n $\chi^2/n_{dof}$={:.3f}\n sigmas={:.3f}".format(chisq, (chisq-1)*n_dof/np.sqrt(2*n_dof)), color="red")
        plt.plot(
            window_centers,
            anomaly_rich,
            label=r"$\tilde{N}_l$ sum of anomaly rich clusters $\tilde{\chi}^2/n_{dof}=$"
            + "{:.3f}".format(res["chisq_ndof"]),
            color="red",
        )
        # r"sum of cluster 1 curves \n $\tilde{\chi}^2/n_d _o _f=$"+"{:.3f}".format(chisq)
        # plt.plot(window_centers, max_norm(count_sum), "--", label="all")
        plt.xlabel("window centre $m_{jj}$ [GeV]")
        plt.ylabel("jets in 16.58GeV window")

        # curvefit thingy
        bg = scipy.interpolate.interp1d(window_centers, anomaly_poor)
        p0_mu = window_centers[np.argmax(anomaly_rich - anomaly_poor)]

        def f(x, w, n, mu, sig):
            return w * bg(x) + n * W * tf / np.sqrt(2 * np.pi) / sig * np.exp(
                -((x - mu) ** 2) / 2 / sig**2
            )

        p0 = (1, 0, p0_mu, 20)
        print("p0_mu", p0_mu)
        rrr = scipy.optimize.curve_fit(
            f,
            window_centers,
            anomaly_rich,
            sigma=np.sqrt(anomaly_rich_sigma**2 + anomaly_poor_sigma**2),
            p0=p0,
            bounds=([0, 0, lb, 10], [2, 10000, rb, (rb - lb) / 2]),
        )
        print(rrr[0])
        # likelyhood spectrum
        # plt.plot(window_centers, f(window_centers, *p0), color="green")
        chisq_fit = np.mean(
            (anomaly_rich - f(window_centers, *rrr[0])) ** 2
            / (anomaly_rich_sigma**2 + anomaly_poor_sigma**2)
        )
        plt.plot(
            window_centers,
            f(window_centers, *rrr[0]),
            color="green",
            label="Curvefit $w=${:.03f}, $n=${:.01f}, \n $\mu=${:.01f}, $\sigma=${:.01f}, \n".format(
                *rrr[0]
            )
            + r"$\tilde{\chi}^2/n_{dof}=$"
            + "{:.3f}".format(chisq_fit),
        )
        plt.legend()
        plt.savefig(save_path + "eval/comb.png", bbox_inches="tight")

    return res


if __name__ == "__main__":

    config_file_path = "config/default_MB.yml"
    cs = cluster_scanning.ClusterScanning(config_file_path)
    cs.load_mjj()
    cs.load_results()
    cs.sample_signal_events()
    counts_windows = cs.perform_binning()
    cs.make_plots()

    cs_performance_evaluation(
        counts_windows=counts_windows,
        plotting=True,
        filterr="med",
        labeling=">5sigma",
        W=100,
        lb=2600,
        rb=6000,
        save_path="char/0kmeans_scan/example1/",
        save_load=True,
        verbous=True,
    )

    cs_performance_evaluation(
        counts_windows=counts_windows,
        plotting=True,
        filterr="med",
        labeling="kmeans_der",  # , #,">5sigma"
        W=100,
        lb=2600,
        rb=6000,
        save_path="char/0kmeans_scan/example2/",
        save_load=True,
        verbous=True,
    )

    print("Executed when invoked directly")
