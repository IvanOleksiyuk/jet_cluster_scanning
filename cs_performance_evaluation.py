import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN, Birch
import time
import os
import pickle
import set_matplotlib_default as smd
import cs_performance_plotting as csp
import cluster_scanning
from robust_estimators import std_ignore_outliers, mean_ignore_outliers
from spectrum import Spectra
import matplotlib as mpl
from squeeze_array import squeeze
from curvefit_eval import curvefit_eval
from binning_utils import default_binning
from config_utils import Config

# mpl.rcParams.update(mpl.rcParamsDefault)
plt.close("all")


def middles(x):
    return (x[1:] + x[:-1]) / 2


class CS_evaluation_process:
    def __init__(
        self,
        config=None,
        config_file_path=None,
        counts_windows=None,
        binning=default_binning(),
        ID="",
    ):

        self.config_file_path = config_file_path
        if config is not None:
            self.config = config
        elif config_file_path is not None:
            self.config = Config(config_file_path)
        self.cfg = self.config.get_dotmap()

        self.counts_windows = counts_windows.T
        self.binning = binning
        self.ID = ID

        # Useful shortcuts and variables
        self.k = self.counts_windows.shape[0]

    def prepare_spectra(self, prepare_spectra=[]):
        """function to prepare the spectra for the evaluation process
        saves spectra after each preparation step so thet they may be used for plotting later on"""

        # Parameters that are not given via config
        counts_windows = self.counts_windows
        binning = self.binning

        # Useful shortcuts and variables
        window_centers = (binning.T[1] + binning.T[0]) / 2

        # Original spectra
        sp_original = Spectra(window_centers, counts_windows)
        previous = sp_original
        for i, action in enumerate(prepare_spectra):
            if action == "sumn":
                new = previous.sum_norm()
            elif action == "maxn":
                new = previous.max_norm()
            elif action == "-bsummaxn":
                sum_sp = sp_original.sum_sp()
                sum_sp_maxn = sum_sp.max_norm()
                new = previous.subtract_bg(sum_sp_maxn)
            elif action == "-bsumsumn":
                sum_sp = sp_original.sum_sp()
                sum_sp_sumn = sum_sp.sum_norm()
                new = previous.subtract_bg(sum_sp_sumn)
            elif action == "der":
                new = previous.num_der()
            elif action == "lowpass":
                new = previous.butter_lowpas()
            elif action == "standrob":
                new = previous.standardize_rob()
            elif action[:3] == "med":
                if action[3:] == "":
                    filter_size = 7
                else:
                    filter_size = int(action[3:])
                new = previous.medfilt(filter_size)
            previous = new

        return previous

    def flip_labels_if_majority_positive(self, labels):
        """function to flip the labels if the majority of the labels is positive"""

        # Useful shortcuts and variables
        k = labels.shape[0]

        # Flip labels if majority is positive
        if np.sum(labels) > k / 2:
            labels = 1 - labels

        return labels

    def label_spectra(self):
        """function to label the spectra for the evaluation process"""

        # parameters from config file
        labeling = self.cfg.labeling

        if labeling == "2meansder":
            sp_y = self.prepare_spectra(["maxn", "der", "med7"]).y
            np.random.seed(self.cfg.sid)
            kmeans = KMeans(2)
            kmeans.fit(sp_y)
            labels = kmeans.labels_
            labels = self.flip_labels_if_majority_positive(labels)

        elif labeling[:6] == "maxdev":
            sp_sumn_standrob = self.prepare_spectra(["sumn", "-bsumsumn", "standrob"])
            threshold = float(labeling[6:])
            labels = np.zeros(self.k)
            for j in range(self.k):
                if np.any(sp_sumn_standrob.y[j] > threshold):
                    labels[j] = 1
                else:
                    labels[j] = 0

        elif labeling == "random":
            labels = np.random.uniform(size=self.k) < 0.5

        else:
            raise ValueError("labeling method not recognized")

        return labels

    def chi_squared_metric(self):

        # Parameters that are not given via config
        binning = self.binning

        # parameters from config file
        plotting = self.cfg.plotting
        save = self.cfg.save
        verbous = self.cfg.verbous
        save_path = self.cfg.save_path

        steal_sd_anomalypoor = self.cfg.steal_sd_anomalypoor

        # Create folder if not yet exists
        if save:
            os.makedirs(save_path, exist_ok=True)

        # Useful shortcuts and variables
        window_centers = (binning.T[1] + binning.T[0]) / 2
        bin_widths = binning.T[1] - binning.T[0]

        # Original spectra
        sp_original = self.prepare_spectra()

        labels = self.label_spectra()
        # total width of all (overlaing) bins devided by width of the covered area (approximation for number of time each point is counted)
        tf = (binning.T[1][-1] - binning.T[0][0]) / np.sum(bin_widths)
        if verbous:
            print("trial_factor", tf)

        # Aggregate clusters using labels
        anomaly_poor_sp = sp_original.sum_sp(np.logical_not(labels)).pscale(tf)
        if np.any(labels):
            anomaly_rich_sp = sp_original.sum_sp(labels.astype(dtype=np.bool_)).pscale(
                tf
            )
        else:
            anomaly_rich_sp = anomaly_poor_sp

        # Steal statistics from poor to rich spectrum if requested
        if steal_sd_anomalypoor == 1:
            # Stal only in low stat regions
            anomaly_rich_sp.make_poiserr_another_sp_sumnorm_where_low_stat(
                anomaly_poor_sp
            )
        elif steal_sd_anomalypoor == 2:
            # Stal everywhere
            anomaly_rich_sp.make_poiserr_another_sp_sumnorm(anomaly_poor_sp)

        anomaly_poor_sp = anomaly_poor_sp.scale(
            np.sum(anomaly_rich_sp.y) / np.sum(anomaly_poor_sp.y)
        )

        chisq_ndof = anomaly_poor_sp.chisq_ndof(anomaly_rich_sp)
        max_sumnorm_dev = np.max(
            anomaly_poor_sp.sum_norm().y - anomaly_rich_sp.sum_norm().y
        )
        max_maxnorm_dev = np.max(
            anomaly_poor_sp.max_norm().y - anomaly_rich_sp.max_norm().y
        )

        res = {}
        res["chisq_ndof"] = chisq_ndof
        res["max-sumnorm-dev"] = max_sumnorm_dev
        res["max-maxnorm-dev"] = max_maxnorm_dev

        # theoretical interpretation
        mean_repetition = np.sum(bin_widths) / (binning.T[0][-1] - binning.T[0][0])
        n_dof = len(window_centers) / mean_repetition
        res["ndof"] = n_dof
        if verbous:
            print("n_dof=", n_dof)
        res["deviation"] = (chisq_ndof - 1) * n_dof / np.sqrt(2 * n_dof)

        if np.any(anomaly_rich_sp.err < 0):
            print("WARNING: rich anomaly error has negative values")
        elif np.any(anomaly_rich_sp.err == 0):
            print("WARNING: rich anomaly error has zero values")
        return res

    def run(self):
        """function to run the evaluation process of a test statistic or a given metric"""
        if self.cfg.test_statistic == "chisq_ndof":
            return self.chi_squared_metric()["chisq_ndof"]
        elif self.cfg.test_statistic == "max-sumnorm-dev":
            return self.chi_squared_metric()["max-sumnorm-dev"]
        elif self.cfg.test_statistic == "max-maxnorm-dev":
            return self.chi_squared_metric()["max-maxnorm-dev"]

        # if plotting:
        #     # Some matplotlib stuff
        #     plt.rcParams["lines.linewidth"] = 1
        #     figsize = (6, 4.5)

        #     # Duplicate plots from the cluster scanning OBSOLETE
        #     csp.plot_all_scalings(
        #         window_centers,
        #         counts_windows,
        #         countmax_windows,
        #         countnrm_windows,
        #         save_path + f"plots{ID}/",
        #         figsize,
        #     )

        #     # Make a directory for evaluation
        #     eval_path = save_path + f"eval{ID}/"
        #     os.makedirs(eval_path, exist_ok=True)

        #     # Distribution of spectra integrals after max normalisation
        #     csp.plot_sum_over_bins_dist(countmax_windows, bin_widths, labels, eval_path)

        #     # subtracted curves norm
        #     csp.two_class_curves(
        #         sp_sumn_s.x,
        #         sp_sumn_s.y,
        #         labels,
        #         figsize,
        #     )
        #     csp.plot_mean_deviat(
        #         sp_sumn_s.x,
        #         sp_sumn_s.mean_sp().y[0],
        #         sp_sumn_s.std_sp().y[0],
        #         fillb=True,
        #     )
        #     csp.plot_mean_deviat(
        #         sp_sumn_s.x,
        #         sp_sumn_s.mean_sp_rob().y[0],
        #         sp_sumn_s.std_sp_rob().y[0],
        #         color="orange",
        #         fillb=True,
        #     )
        #     plt.legend()
        #     plt.xlabel("window centre $m_{jj}$ [GeV]")
        #     plt.ylabel("$N_i(m_{jj})/sum(N_i(m_{jj}))$-background")
        #     plt.savefig(eval_path + "norm-toatal.png", bbox_inches="tight")

        #     # all curves standardized
        #     csp.two_class_curves(
        #         window_centers,
        #         sp_sumn_standrob.y,
        #         labels,
        #         figsize,
        #     )
        #     plt.axhline(5, color="red", alpha=0.2)
        #     plt.xlabel("window centre $m_{jj}$ [GeV]")
        #     plt.ylabel("deviation in SD")
        #     plt.legend()
        #     plt.savefig(eval_path + "norm-toatal-sigmas.png", bbox_inches="tight")

        #     # all curves standardized
        #     csp.two_class_curves(
        #         window_centers,
        #         sp_sumn_standrob.y * np.sqrt(countnrm_windows),
        #         labels,
        #         figsize,
        #         ylabel="deviation in SD",
        #         save_file=eval_path + "norm-toatal-sigmas-special.png",
        #     )

        #     # all curves standardized squeezed
        #     sqf = 20
        #     csp.two_class_curves(
        #         squeeze(window_centers, sqf),
        #         squeeze(sp_sumn_standrob.y, sqf),
        #         labels,
        #         figsize,
        #         ylabel="deviation in SD",
        #         save_file=eval_path + "norm-toatal-sigmas-squeezed.png",
        #     )

        #     # distribution of total MSE after standartisation
        #     csp.two_class_curves(
        #         window_centers,
        #         sp_sumn_standrob.y,
        #         labels,
        #         figsize,
        #         ylabel="deviation in SD",
        #         save_file=eval_path + "norm-toatal-sigmas-special.png",
        #         marker=".",
        #         linestyle="",
        #     )

        #     # all curves standardized points
        #     csp.two_class_curves(
        #         window_centers,
        #         sp_sumn_standrob.y * np.sqrt(countnrm_windows),
        #         labels,
        #         figsize,
        #         ylabel="deviation in SD",
        #         save_file=eval_path + "norm-toatal-sigmas-special.png",
        #     )

        #     # all curves standardized distribution
        #     fig, axs = plt.subplots(
        #         1, 2, figsize=(9, 3), gridspec_kw={"width_ratios": [2, 1]}
        #     )
        #     plt.sca(axs[0])
        #     ys = []
        #     for j in range(k):
        #         ys.append(list(sp_sumn_standrob.y[j]))
        #     ys = np.array(ys)
        #     ys = ys.reshape((-1,))
        #     xs = np.tile(window_centers, k)
        #     h, xedges, yedges, image = plt.hist2d(
        #         xs, ys, bins=(200, 40), cmap="gist_heat_r"
        #     )
        #     plt.xlabel("window centre $m_{jj}$ [GeV]")
        #     plt.ylabel("deviation in SD")
        #     plt.colorbar()
        #     plt.sca(axs[1])
        #     yy = (yedges[:-1] + yedges[1:]) / 2
        #     # for i in range(200):
        #     plt.step(yy, np.mean(h, axis=0), where="mid", color="darkorange")
        #     plt.xlabel("deviations in SD")
        #     plt.ylabel("avarage curve points per bin")
        #     plt.savefig(eval_path + "2d_hist.png", bbox_inches="tight")

        #     # subtracted curves norm
        #     csp.two_class_curves(
        #         window_centers,
        #         sp_maxn_s.y,
        #         labels,
        #         figsize,
        #         ylabel="n_clusater/max(n_cluster)-bg",
        #     )

        #     plt.plot(
        #         window_centers,
        #         np.mean(sp_maxn_s.y, axis=0),
        #         color="lime",
        #     )
        #     plt.plot(
        #         window_centers,
        #         -std_ignore_outliers(countmax_windows),
        #         color="lime",
        #     )
        #     plt.plot(
        #         window_centers, std_ignore_outliers(countmax_windows), color="lime"
        #     )
        #     plt.fill_between(
        #         sp_maxn_s.x,
        #         np.mean(sp_maxn_s.y, axis=0) - np.std(sp_maxn_s.y, axis=0),
        #         np.mean(sp_maxn_s.y, axis=0) + np.std(sp_maxn_s.y, axis=0),
        #         alpha=0.4,
        #         color="lime",
        #     )
        #     plt.xlabel("window centre $m_{jj}$ [GeV]")
        #     plt.ylabel("n_clusater/max(n_cluster)-bg")
        #     plt.savefig(eval_path + "maxnorm-toatal.png", bbox_inches="tight")

        #     # all center curves with found labels
        #     plt.figure(figsize=figsize)
        #     plt.grid()
        #     for j in range(k):
        #         if labels[j]:
        #             plt.plot(window_centers, countmax_windows[j], color="red")
        #         else:
        #             plt.plot(window_centers, countmax_windows[j], color="blue")
        #     plt.xlabel("window centre $m_{jj}$ [GeV]")
        #     plt.ylabel("n_clusater/max(n_cluster)")
        #     plt.savefig(eval_path + "countmax_windows.png", bbox_inches="tight")

        #     # all center curve derivatives with found labels
        #     plt.figure(figsize=figsize)
        #     plt.grid()
        #     lab1 = "anomalous clusters \n method 1"
        #     lab2 = "non-anomalous clusters \n method 1"
        #     for j in range(k):
        #         if labels[j]:
        #             plt.plot(
        #                 middles(window_centers),
        #                 num_der_counts_windows[j],
        #                 color="red",
        #                 label=lab1,
        #             )
        #             lab1 = None
        #         else:
        #             plt.plot(
        #                 middles(window_centers),
        #                 num_der_counts_windows[j],
        #                 color="blue",
        #                 label=lab2,
        #             )
        #             lab2 = None
        #     plt.xlabel("window centre $m_{jj}$ [GeV] between adjacent windows")
        #     plt.ylabel(r"$\Delta(N_i(m_{jj})/sum(N_i(m_{jj}))$)")
        #     plt.savefig(eval_path + "num_der_counts_windows.png", bbox_inches="tight")

        #     # all center curve derivatives with found labels
        #     plt.figure(figsize=figsize)
        #     plt.grid()
        #     for j in range(k):
        #         if labels[j]:
        #             plt.plot(
        #                 middles(window_centers),
        #                 num_der_counts_windows[j] - num_der_countmax_sum,
        #                 color="red",
        #             )
        #         else:
        #             plt.plot(
        #                 middles(window_centers),
        #                 num_der_counts_windows[j] - num_der_countmax_sum,
        #                 color="blue",
        #             )
        #     plt.xlabel("window centre $m_{jj}$ [GeV]")
        #     plt.ylabel(r"$Median filtered \Delta[N_i(m_{jj})/sum(N_i(m_{jj}))]$")
        #     plt.legend()
        #     plt.savefig(
        #         eval_path + "num_der_counts_windows_-bg.png",
        #         bbox_inches="tight",
        #     )

        #     # TSNE
        #     csp.CS_TSNE(num_der_counts_windows, labels, eval_path)

        #     # aggregations

        #     csp.plot_aggregation(anomaly_poor_sp, anomaly_rich_sp, figsize, res)
        #     curvefit_eval(anomaly_poor_sp, anomaly_rich_sp, binning, tf)
        #     plt.savefig(eval_path + "comb.png", bbox_inches="tight")

        #     csp.plot_aggregation(
        #         anomaly_poor_sp.subtract_sp(anomaly_rich_sp),
        #         anomaly_rich_sp.subtract_sp(anomaly_rich_sp),
        #         figsize,
        #         res,
        #     )
        #     plt.savefig(eval_path + "comb_dev.png", bbox_inches="tight")


def cs_performance_evaluation(*args, **kwargs):
    CSE = CS_evaluation_process(*args, **kwargs)
    res = CSE.run()
    return res


if __name__ == "__main__":
    cs_performance_evaluation()
    print("Executed when invoked directly")
