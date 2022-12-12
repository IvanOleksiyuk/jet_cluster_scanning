import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN, Birch
import time
import os
import pickle
from lowpasfilter import butter_lowpass_filter
import scipy.signal
import set_matplotlib_default as smd
import cs_performance_plotting as csp
import cluster_scanning
from robust_estimators import std_ignore_outliers, mean_ignore_outliers
from spectrum import Spectra
import matplotlib as mpl
from squeeze_array import squeeze

mpl.rcParams.update(mpl.rcParamsDefault)
plt.close("all")


def middles(x):
    return (x[1:] + x[:-1]) / 2


def default_binning(W=100, lb=2600, rb=6000, steps=200):
    mjjmin_arr = np.linspace(lb, rb - W, steps)
    mjjmax_arr = mjjmin_arr + W
    binning = np.stack([mjjmin_arr, mjjmax_arr]).T
    return binning


def cs_performance_evaluation(
    counts_windows=None,
    plotting=True,
    filterr="med",
    labeling="kmeans_der",  # , #,">5sigma"
    binning=default_binning(),
    save_path="cs_performance_evaluation/",
    verbous=True,
    sid=0,
):
    # Usuallly oriented incorrectly #TODO correct this
    counts_windows = counts_windows.T

    # Create folder if not yet exists
    os.makedirs(save_path, exist_ok=True)

    # Useful shortcuts and variables
    window_centers = (binning.T[1] + binning.T[0]) / 2
    bin_widths = binning.T[1] - binning.T[0]
    k = counts_windows.shape[0]

    # Original spectra
    sp_original = Spectra(window_centers, counts_windows)

    # produce maxnormed and normed versions of counts_windows
    sp_maxn = sp_original.max_norm()
    sp_sumn = sp_original.sum_norm()

    # produce maxnormed and normed versions of counts_windows
    countmax_windows = sp_maxn.y
    countnrm_windows = sp_sumn.y

    # total signal+background spectrum in windows
    sum_sp = sp_original.sum_sp()

    # normed vesions
    sum_sp_maxn = sum_sp.max_norm()
    sum_sp_sumn = sum_sp.sum_norm()

    # DO I NEED THESE?
    countnrm_sum = sum_sp_sumn.y[0]

    # versions without respective background
    sp_maxn_s = sp_maxn.subtract_bg(sum_sp_maxn)
    sp_sumn_s = sp_sumn.subtract_bg(sum_sp_sumn)
    countnrm_windows_s = sp_sumn_s.y
    countmax_windows_s = sp_maxn_s.y

    baselinenrm = mean_ignore_outliers(countnrm_windows - countnrm_sum)
    countnrm_windows_sigma = std_ignore_outliers(countnrm_windows)
    countnrm_windows_std = (
        countnrm_windows - countnrm_sum - baselinenrm
    ) / countnrm_windows_sigma

    num_der_counts_windows = sp_maxn.num_der().y
    num_der_countmax_sum = sum_sp_maxn.num_der().y[0]

    if filterr == "lowpass":
        order = 6
        fs = 1  # sample rate, Hz
        cutoff = 0.05  # desired cutoff frequency of the filter, Hz
        for i in range(k):
            num_der_counts_windows[i] = butter_lowpass_filter(
                num_der_counts_windows[i], cutoff, fs, order
            )

    elif filterr == "med5":
        for i in range(k):
            num_der_counts_windows[i] = scipy.signal.medfilt(
                num_der_counts_windows[i], [5]
            )

    elif filterr == "med":
        for i in range(k):
            num_der_counts_windows[i] = scipy.signal.medfilt(
                num_der_counts_windows[i], [7]
            )

    elif filterr == "med11":
        for i in range(k):
            num_der_counts_windows[i] = scipy.signal.medfilt(
                num_der_counts_windows[i], [11]
            )

    ###
    ###

    if labeling == "kmeans_der":
        np.random.seed(sid)
        kmeans = KMeans(2)
        kmeans.fit(num_der_counts_windows)
        as_vectors = "derivatives"
        if np.sum(kmeans.labels_) < len(kmeans.labels_) // 2:
            labels = kmeans.labels_
        else:
            labels = 1 - kmeans.labels_

    elif labeling == "kmeans_cur":
        np.random.seed(sid)
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

    # total width of all (overlaing) bins devided by width of the covered area (approximation for number of time each point is counted)
    tf = (binning.T[1][-1] - binning.T[0][0]) / np.sum(bin_widths)

    anomaly_poor = np.sum(counts_windows[labels == 0], axis=0) * tf
    if np.any(labels):
        anomaly_rich = np.sum(counts_windows[labels == 1], axis=0) * tf
    else:
        anomaly_rich = anomaly_poor

    if verbous:
        print("trial_factor", tf)
    anomaly_poor_sigma = np.sqrt(anomaly_poor)
    anomaly_rich_sigma = np.sqrt(
        anomaly_poor / sum(anomaly_poor) * sum(anomaly_rich)
    )

    anomaly_poor_sigma = anomaly_poor_sigma * (
        np.sum(anomaly_rich) / np.sum(anomaly_poor)
    )
    anomaly_poor = anomaly_poor * (np.sum(anomaly_rich) / np.sum(anomaly_poor))

    chisq = np.mean(
        (anomaly_rich - anomaly_poor) ** 2
        / (anomaly_rich_sigma**2 + anomaly_poor_sigma**2)
    )
    mean_repetition = np.sum(bin_widths) / (binning.T[0][-1] - binning.T[0][0])
    # total width of all (overlaing) bins devided by width of the covered area (approximation for number of time each point is counted)
    # Assuming all windows have equal width we take the covered region from min of the firs to min of the last (not to max of the last) to compencete for the points close to edges not being counted sevral times
    # Most likely a totaly false formula but actualy obsolete

    n_dof = len(window_centers) / mean_repetition

    if verbous:
        print("n_dof=", n_dof)

    res = {}
    res["chisq_ndof"] = chisq
    res["ndof"] = n_dof
    res["deviation"] = (chisq - 1) * n_dof / np.sqrt(2 * n_dof)

    if plotting:
        # Some matplotlib stuff
        plt.rcParams["lines.linewidth"] = 1
        figsize = (6, 4.5)

        # Duplicate plots from the cluster scanning OBSOLETE
        csp.plot_all_scalings(
            window_centers,
            counts_windows,
            countmax_windows,
            countnrm_windows,
            save_path,
            figsize,
        )

        # Make a directory for evaluation
        os.makedirs(save_path + "eval/", exist_ok=True)

        # Distribution of spectra integrals after max normalisation
        csp.plot_sum_over_bins_dist(
            countmax_windows, bin_widths, labels, save_path
        )

        # subtracted curves norm
        csp.two_class_curves(
            window_centers,
            countnrm_windows_s,
            labels,
            figsize,
        )
        csp.plot_mean_deviat(
            window_centers,
            np.mean(countnrm_windows_s, axis=0),
            np.std(countnrm_windows, axis=0),
            fillb=True,
        )
        csp.plot_mean_deviat(
            window_centers,
            baselinenrm,
            std_ignore_outliers(countnrm_windows),
            color="orange",
            fillb=True,
        )
        plt.legend()
        plt.xlabel("window centre $m_{jj}$ [GeV]")
        plt.ylabel("$N_i(m_{jj})/sum(N_i(m_{jj}))$-background")
        plt.savefig(save_path + "eval/norm-toatal.png", bbox_inches="tight")

        # all curves standardized
        csp.two_class_curves(
            window_centers,
            countnrm_windows_std,
            labels,
            figsize,
        )
        plt.axhline(5, color="red", alpha=0.2)
        plt.xlabel("window centre $m_{jj}$ [GeV]")
        plt.ylabel("deviation in SD")
        plt.legend()
        plt.savefig(
            save_path + "eval/norm-toatal-sigmas.png", bbox_inches="tight"
        )

        # all curves standardized
        csp.two_class_curves(
            window_centers,
            countnrm_windows_std * np.sqrt(countnrm_windows),
            labels,
            figsize,
            ylabel="deviation in SD",
            save_file=save_path + "eval/norm-toatal-sigmas-special.png",
        )

        # all curves standardized squeezed
        sqf = 20
        csp.two_class_curves(
            squeeze(window_centers, sqf),
            squeeze(countnrm_windows_std, sqf),
            labels,
            figsize,
            ylabel="deviation in SD",
            save_file=save_path + "eval/norm-toatal-sigmas-squeezed.png",
        )

        # distribution of total MSE after standartisation
        csp.two_class_curves(
            window_centers,
            countnrm_windows_std,
            labels,
            figsize,
            ylabel="deviation in SD",
            save_file=save_path + "eval/norm-toatal-sigmas-special.png",
            marker=".",
            linestyle="",
        )

        # all curves standardized points
        csp.two_class_curves(
            window_centers,
            countnrm_windows_std * np.sqrt(countnrm_windows),
            labels,
            figsize,
            ylabel="deviation in SD",
            save_file=save_path + "eval/norm-toatal-sigmas-special.png",
        )

        # all curves standardized distribution
        fig, axs = plt.subplots(
            1, 2, figsize=(9, 3), gridspec_kw={"width_ratios": [2, 1]}
        )
        plt.sca(axs[0])
        ys = []
        for j in range(k):
            ys.append(list(countnrm_windows_std[j]))
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
                    countmax_windows_s[j],
                    color="red",
                )
            else:
                plt.plot(
                    window_centers,
                    countmax_windows_s[j],
                    color="blue",
                )

        plt.plot(
            window_centers,
            np.mean(countmax_windows_s, axis=0),
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
            np.mean(countmax_windows_s, axis=0)
            - np.std(countmax_windows, axis=0),
            np.mean(countmax_windows_s, axis=0)
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
            return w * bg(x) + n * (bin_widths) * tf / np.sqrt(
                2 * np.pi
            ) / sig * np.exp(-((x - mu) ** 2) / 2 / sig**2)

        p0 = (1, 0, p0_mu, 20)
        print("p0_mu", p0_mu)
        rrr = scipy.optimize.curve_fit(
            f,
            window_centers,
            anomaly_rich,
            sigma=np.sqrt(anomaly_rich_sigma**2 + anomaly_poor_sigma**2),
            p0=p0,
            bounds=(
                [0, 0, binning.min(), 10],
                [2, 10000, binning.max(), (binning.max() - binning.min()) / 2],
            ),
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
    sliding_cluster_performance_evaluation()
    print("Executed when invoked directly")
