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
        path=None,
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
        self.window_centers = (binning.T[1] + binning.T[0]) / 2
        self.bin_widths = binning.T[1] - binning.T[0]

        # plotting configurations
        plt.rcParams["lines.linewidth"] = 1
        self.figsize = (6, 4.5)

        # Paths
        if path is not None:
            self.cfg.save_path = path
        self.eval_path = self.cfg.save_path + f"eval{ID}/"

        # other variables that have to be initialized
        self.labels = None

    def prepare_spectra(self, prepare_spectra=[]):
        """function to prepare the spectra for the evaluation process
        saves spectra after each preparation step so thet they may be used for plotting later on"""

        # Original spectra
        sp_original = Spectra(self.window_centers, self.counts_windows)
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
        self.labels = labels
        return labels

    def chi_squared_metric(self):

        # Parameters that are not given via config
        binning = self.binning

        # parameters from config file
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
        res["tf"] = tf
        self.agg_sp = {}
        self.agg_sp["rich"] = anomaly_rich_sp
        self.agg_sp["poor"] = anomaly_poor_sp
        self.agg_sp["res"] = res
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
            res = self.chi_squared_metric()["chisq_ndof"]
        elif self.cfg.test_statistic == "max-sumnorm-dev":
            res = self.chi_squared_metric()["max-sumnorm-dev"]
        elif self.cfg.test_statistic == "max-maxnorm-dev":
            res = self.chi_squared_metric()["max-maxnorm-dev"]

        if self.cfg.plotting:
            self.plot()

        return res

    def plot(self):
        os.makedirs(self.cfg.save_path, exist_ok=True)
        self.redo_cs_plots()
        os.makedirs(self.eval_path, exist_ok=True)
        if self.labels is not None:
            csp.plot_sum_over_bins_dist(
                self.prepare_spectra(["maxn"]).y,
                self.bin_widths,
                self.labels,
                self.eval_path,
            )
            self.plot_standardisation_step(
                self.prepare_spectra(["sumn", "-bsumsumn"]),
                self.labels,
                y_label="$N_i(m_{jj})/sum(N_i(m_{jj}))$-background",
                savefile="sumn-toatal.png",
            )
            self.plot_standardisation_step(
                self.prepare_spectra(["maxn", "-bsummaxn"]),
                self.labels,
                y_label="$N_i(m_{jj})/max(N_i(m_{jj}))$-background",
                savefile="maxm-toatal.png",
            )
            self.plot_labeled_spectra(
                self.prepare_spectra(["sumn", "-bsumsumn", "standrob"]),
                self.labels,
                add_line=True,
                ylabel="deviation in SD",
                filename="norm-toatal-sigmas.png",
            )
            self.plot_labeled_spectra(
                self.prepare_spectra(["sumn"]),
                self.labels,
                ylabel="$N_i(m_{jj})/sum(N_i(m_{jj}))$",
                filename="sumn.png",
            )
            self.plot_labeled_spectra(
                self.prepare_spectra(["maxn"]),
                self.labels,
                ylabel="$N_i(m_{jj})/max(N_i(m_{jj}))$",
                filename="maxn.png",
            )
            self.plot_labeled_spectra(
                self.prepare_spectra(["maxn", "der", "med7"]),
                self.labels,
                ylabel=r"$\Delta(N_i(m_{jj})/max(N_i(m_{jj}))$)",
                filename="num-der.png",
            )
            # TSNE
            csp.CS_TSNE(
                self.prepare_spectra(["maxn", "der", "med7"]).y,
                self.labels,
                self.eval_path,
            )
            # aggregations
            csp.plot_aggregation(
                self.agg_sp["poor"],
                self.agg_sp["rich"],
                self.figsize,
                self.agg_sp["res"],
            )
            curvefit_eval(
                self.agg_sp["poor"],
                self.agg_sp["rich"],
                self.binning,
                self.agg_sp["res"]["tf"],
            )
            plt.savefig(self.eval_path + "comb.png", bbox_inches="tight")
            csp.plot_aggregation(
                self.agg_sp["poor"].subtract_sp(self.agg_sp["poor"]),
                self.agg_sp["poor"].subtract_sp(self.agg_sp["rich"]),
                self.figsize,
                self.agg_sp["res"],
            )
            plt.savefig(self.eval_path + "comb_dev.png", bbox_inches="tight")

    def redo_cs_plots(self):
        # Duplicate plots from the cluster scanning OBSOLETE
        os.makedirs(self.cfg.save_path + f"plots{self.ID}/", exist_ok=True)
        csp.plot_all_scalings(
            self.window_centers,
            self.counts_windows,
            self.prepare_spectra(["maxn"]).y,
            self.prepare_spectra(["sumn"]).y,
            self.cfg.save_path + f"plots{self.ID}/",
            self.figsize,
        )

    def plot_labeled_spectra(
        self, sp, labels, ylabel="y", filename="test.png", add_line=False
    ):
        csp.two_class_curves(
            sp.x,
            sp.y,
            labels,
            self.figsize,
        )
        if add_line:
            plt.axhline(5, color="red", alpha=0.2)
        plt.xlabel("window centre $m_{jj}$ [GeV]")
        plt.ylabel(ylabel)
        plt.legend()
        plt.savefig(self.eval_path + filename, bbox_inches="tight")

    def plot_standardisation_step(self, sp, labels, y_label="y", savefile="test.png"):

        csp.two_class_curves(
            sp.x,
            sp.y,
            labels,
            self.figsize,
        )
        csp.plot_mean_deviat(
            sp.x,
            sp.mean_sp().y[0],
            sp.std_sp().y[0],
            fillb=True,
        )
        csp.plot_mean_deviat(
            sp.x,
            sp.mean_sp_rob().y[0],
            sp.std_sp_rob().y[0],
            color="orange",
            fillb=True,
        )
        plt.legend()
        plt.xlabel("window centre $m_{jj}$ [GeV]")
        plt.ylabel(y_label)
        plt.savefig(self.eval_path + savefile, bbox_inches="tight")


def cs_performance_evaluation(*args, **kwargs):
    CSE = CS_evaluation_process(*args, **kwargs)
    res = CSE.run()
    return res


if __name__ == "__main__":
    config_path = ["config/cs_eval/maxdev5.yaml", "config/cs_eval/plotting.yaml"]
    jj = 0
    path = "char/sig_reg/k50Trueret0con0.05W3450_3650_w0.5s1Nrest/"
    # path1 = path + "binnedW100s200ei26006000/"
    path1 = path + "binnedW100s16ei30004600/"
    counts_windows = pickle.load(open(path1 + f"bres{jj}.pickle", "rb"))
    binning = pickle.load(open(path1 + "binning.pickle", "rb"))
    cs_performance_evaluation(
        counts_windows=counts_windows,
        binning=binning,
        path=path1,
        ID=jj,
        config_file_path=config_path,
    )
    print("Executed when invoked directly")

# Concepts
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
