import numpy as np
import copy
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN, Birch
import time
import os
import pickle
import cs_performance_plotting as csp
import cluster_scanning
from utils.robust_estimators import std_ignore_outliers, mean_ignore_outliers
import utils.set_matplotlib_default
from utils.spectrum import Spectra
from utils.squeeze_array import squeeze
import matplotlib as mpl
from curvefit_eval import curvefit_eval
from utils.binning_utils import default_binning
from utils.config_utils import Config
import logging
from scipy import stats

logging.basicConfig(level=logging.INFO)
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

        if isinstance(counts_windows, list):
            self.separate_binning = True
            if len(counts_windows) == 2:
                self.counts_windows_bg = counts_windows[0].T
                self.counts_windows_sg = counts_windows[1].T
                self.counts_windows = self.counts_windows_bg + self.counts_windows_sg
            elif len(counts_windows) == 1:
                self.counts_windows_bg = counts_windows[0].T
                self.counts_windows_sg = np.zeros_like(self.counts_windows_bg)
                self.counts_windows = self.counts_windows_bg
        else:
            self.counts_windows = counts_windows.T
            self.separate_binning = False

        non_empty_bins = np.sum(self.counts_windows, axis=1) > 0
        self.counts_windows = self.counts_windows[non_empty_bins]
        if self.separate_binning:
            self.counts_windows_bg = self.counts_windows_bg[non_empty_bins]
            self.counts_windows_sg = self.counts_windows_sg[non_empty_bins]

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
            if action == "fix_low_stat_error":
                new = copy.deepcopy(previous)
                new.make_poiserr_another_sp_sumnorm_where_low_stat(
                    sp_original.sum_sp(), low_stat=10
                )
            elif action == "sumn":
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
            elif action == "-fit2jet3par":
                new = previous.subtract_fit("3_param")
            elif action == "-fit2jet4par":
                new = previous.subtract_fit("4_param")
            elif action == "-fit2jet5par":
                new = previous.subtract_fit("5_param")
            elif action == "der":
                new = previous.num_der()
            elif action == "lowpass":
                new = previous.butter_lowpas()
            elif action == "standrob":
                new = previous.standardize_rob()
            elif action == "standardize":
                new = previous.standardize()
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

    def significance_improvement(self, cluster=None):
        initial_significance = np.sum(self.counts_windows_sg) / np.sqrt(
            np.sum(self.counts_windows_bg)
        )
        #print(initial_significance)
        final_significance = np.sum(self.counts_windows_sg[cluster]) / np.sqrt(
            np.sum(self.counts_windows_bg[cluster])
        )
        #print(final_significance)
        return final_significance / initial_significance
    
    def significance_improvement_full(self):
        if self.labels is None:
            self.labels = self.label_spectra()
        initial_significance = np.sum(self.counts_windows_sg) / np.sqrt(
            np.sum(self.counts_windows_bg)
        )
        #print(initial_significance)
        final_significance = np.sum(self.counts_windows_sg[self.labels == 1]) / np.sqrt(
            np.sum(self.counts_windows_bg[self.labels == 1])
        )
        #print(final_significance)
        return final_significance / initial_significance

    def signal_efficiency(self):
        """function to calculate the purity of the labels"""
        # Calculate purity
        if self.labels is None:
            self.labels = self.label_spectra()
        fraction_signal = np.sum(self.counts_windows_sg[self.labels == 1]) / np.sum(
            self.counts_windows_sg
        )
        return fraction_signal

    def background_efficiency(self):
        """function to calculate the purity of the labels"""
        # Calculate purity
        if self.labels is None:
            self.labels = self.label_spectra()
        fraction_signal = np.sum(self.counts_windows_bg[self.labels == 1]) / np.sum(
            self.counts_windows_bg
        )
        return fraction_signal

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

        elif labeling == "maxdev3_3fit":
            sp_sumn_standrob = self.prepare_spectra(["sumn", "-fit2jet3par", "standrob"])
            logging.debug(str(sp_sumn_standrob.y / sp_sumn_standrob.err))
            self.threshold = float(labeling[6:7])
            labels = np.zeros(self.k)
            for j in range(self.k):
                if np.any(sp_sumn_standrob.y[j] > self.threshold):
                    labels[j] = 1
                else:
                    labels[j] = 0

        elif labeling[:6] == "maxdev":
            sp_sumn_standrob = self.prepare_spectra(["sumn", "-bsumsumn", "standrob"])
            logging.debug(str(sp_sumn_standrob.y / sp_sumn_standrob.err))
            self.threshold = float(labeling[6:])
            labels = np.zeros(self.k)
            for j in range(self.k):
                if np.any(sp_sumn_standrob.y[j] > self.threshold):
                    labels[j] = 1
                else:
                    labels[j] = 0
        
        elif labeling == "random":
            labels = np.random.uniform(size=self.k) < 0.5

        else:
            raise ValueError("labeling method not recognized")
        self.labels = labels

        if self.separate_binning:
            SIs = []
            for i in range(self.k):
                SIs.append(self.significance_improvement(cluster=[i]))
            # logging.info("-------------------------------------------------")
            # logging.info(f"{SIs} max of SI's of separate clusters")
            # logging.info(f"{int(np.sum(labels))} clusters were chosen for signal, aggregation gives SI of {self.significance_improvement():.4f}")
            # print(f"{self.background_efficiency():},")
            # logging.info(f"signal efficiency: {self.signal_efficiency():.4f},, background efficiency: {self.background_efficiency():.4f}")
        return labels

    def aggregation_based_TS(self):
        """calculates several test statistics (TS) based on choosing signal and background clusters, aggregating them and compluting the test statistic on between the two curves"""

        # save options?
        save = self.cfg.save
        save_path = self.cfg.save_path
        if save:
            os.makedirs(save_path, exist_ok=True)

        # Original spectra
        sp_original = self.prepare_spectra()

        labels = self.label_spectra()
        # total width of all (overlaing) bins devided by width of the covered area (approximation for number of time each point is counted)
        tf = (self.binning.T[1][-1] - self.binning.T[0][0]) / np.sum(self.bin_widths)
        logging.debug(f"trial_factor {tf}")

        # Aggregate clusters using labels
        anomaly_poor_sp = sp_original.sum_sp(np.logical_not(labels)).pscale(tf)
        if np.any(labels):
            anomaly_rich_sp = sp_original.sum_sp(labels.astype(dtype=np.bool_)).pscale(
                tf
            )
        else:
            anomaly_rich_sp = anomaly_poor_sp

        # Steal statistics from poor to rich spectrum if requested
        if self.cfg.steal_sd_anomalypoor == 1:
            # Stal only in low stat regions
            anomaly_rich_sp.make_poiserr_another_sp_sumnorm_where_low_stat(
                anomaly_poor_sp
            )
        elif self.cfg.steal_sd_anomalypoor == 2:
            # Stal everywhere
            anomaly_rich_sp.make_poiserr_another_sp_sumnorm(anomaly_poor_sp)

        if hasattr(self.cfg, "background_estim"):
            if self.cfg.background_estim == "anomaly_poor_scled":
                anomaly_poor_sp = anomaly_poor_sp.scale(
                    np.sum(anomaly_rich_sp.y) / np.sum(anomaly_poor_sp.y)
                )
            elif self.cfg.background_estim == "4_param_fit":
                anomaly_poor_sp = anomaly_rich_sp.fit("4_param")
            elif self.cfg.background_estim == "3_param_fit":
                print("doing 3 param")
                anomaly_poor_sp = anomaly_rich_sp.fit("3_param")
        else:
            anomaly_poor_sp = anomaly_poor_sp.scale(
                np.sum(anomaly_rich_sp.y) / np.sum(anomaly_poor_sp.y)
            )

        # Calculate test statistics on aggregated spectra
        chisq_ndof = anomaly_poor_sp.chisq_ndof(anomaly_rich_sp)
        max_sumnorm_diff = anomaly_poor_sp.sum_norm().max_diff_abs(
            anomaly_rich_sp.sum_norm()
        )
        max_maxnorm_diff = anomaly_poor_sp.max_norm().max_diff_abs(
            anomaly_rich_sp.max_norm()
        )
        max_sumnorm_dev = anomaly_poor_sp.sum_norm().max_dev_abs(
            anomaly_rich_sp.sum_norm()
        )
        max_maxnorm_dev = anomaly_poor_sp.max_norm().max_dev_abs(
            anomaly_rich_sp.max_norm()
        )
        max_sumnorm_dev_rs = anomaly_poor_sp.max_dev_abs(
            anomaly_rich_sp, error_style="self_sqrt"
        )

        # Save results
        res = {}
        res["chisq_ndof"] = chisq_ndof
        res["max-sumnorm-dev"] = max_sumnorm_dev
        res["max-maxnorm-dev"] = max_maxnorm_dev
        res["max-sumnorm-dev-rs"] = max_sumnorm_dev_rs
        res["max-sumnorm-diff"] = max_sumnorm_diff
        res["max-maxnorm-diff"] = max_maxnorm_diff
        res["tf"] = tf
        self.agg_sp = {}
        self.agg_sp["rich"] = anomaly_rich_sp
        self.agg_sp["poor"] = anomaly_poor_sp
        self.agg_sp["res"] = res
        # theoretical interpretation
        mean_repetition = np.sum(self.bin_widths) / (
            self.binning.T[0][-1] - self.binning.T[0][0]
        )
        n_dof = len(self.window_centers) / mean_repetition
        res["ndof"] = n_dof
        logging.debug(f"n_dof={n_dof}")
        res["deviation"] = (chisq_ndof - 1) * n_dof / np.sqrt(2 * n_dof)

        if np.any(anomaly_rich_sp.err < 0):
            logging.warning("rich anomaly error has negative values")
        elif np.any(anomaly_rich_sp.err == 0):
            logging.warning("rich anomaly error has zero values")
        return res

    def non_aggregation_based_TS(self, prepare_sp, cluster_sc, mjj_sc):

        sp_sumnorm = self.prepare_spectra(prepare_spectra=prepare_sp)

        if mjj_sc == "max":
            per_cluster_score = sp_sumnorm.max_dev()
        elif mjj_sc == "chi":
            per_cluster_score = sp_sumnorm.chisq_ndof()

        if cluster_sc == "max":
            return np.max(per_cluster_score)
        elif cluster_sc == "chi":
            if mjj_sc == "max":
                return np.mean(per_cluster_score**2)
            elif mjj_sc == "chi":
                return np.mean(per_cluster_score)

    def run(self):
        """function to run the evaluation process of a test statistic or a given metric"""
        prepr = ["fix_low_stat_error", "sumn", "-bsumsumn"]

        aggr_based_stats = [
            "chisq_ndof",
            "max-sumnorm-dev",
            "max-maxnorm-dev",
            "max-sumnorm-diff",
            "max-maxnorm-diff",
            "max-sumnorm-dev-rs",
            "max-maxnorm-dev-rs",
        ]
        if self.cfg.test_statistic in aggr_based_stats:
            res = self.aggregation_based_TS()[self.cfg.test_statistic]

        # print(self.cfg.test_statistic)
        # print(self.cfg.test_statistic in aggr_based_stats)
        # print(res)
        # exit()

        elif self.cfg.test_statistic == "NA-sumnorm-maxC-maxM":
            res = self.non_aggregation_based_TS(
                prepare_sp=prepr, cluster_sc="max", mjj_sc="max"
            )
        elif self.cfg.test_statistic == "NA-sumnorm-maxC-chiM":
            res = self.non_aggregation_based_TS(
                prepare_sp=prepr, cluster_sc="max", mjj_sc="chi"
            )
        elif self.cfg.test_statistic == "NA-sumnorm-chiC-maxM":
            res = self.non_aggregation_based_TS(
                prepare_sp=prepr, cluster_sc="chi", mjj_sc="max"
            )
        elif self.cfg.test_statistic == "NA-sumnorm-chiC-chiM":
            res = self.non_aggregation_based_TS(
                prepare_sp=prepr, cluster_sc="chi", mjj_sc="chi"
            )
        elif self.cfg.test_statistic == "signal_efficiency":
            res = self.signal_efficiency()
        elif self.cfg.test_statistic == "background_efficiency":
            res = self.background_efficiency()
        elif self.cfg.test_statistic == "significance_improvement":
            res = self.significance_improvement_full()

        if self.cfg.plotting:
            self.plot()

        return res

    def plot(self, format=".pdf"):
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
                y_label="$N_{i,b}/\Sigma_b N_{i,b}-N_{orig, b}/\Sigma_b N_{orig, b}$",
                savefile="sumn-toatal"+format,
            )
            self.plot_standardisation_step(
                self.prepare_spectra(["maxn", "-bsummaxn"]),
                self.labels,
                y_label="$N_{i,b}/max_b(N_{i,b})-N_{orig, b}/max_b(N_{orig, b})$",
                savefile="maxm-toatal"+format,
            )
            self.plot_labeled_spectra(
                self.prepare_spectra(["sumn", "-bsumsumn", "standrob"]),
                self.labels,
                add_line=True,
                ylabel="deviation in SD",
                filename="norm-toatal-sigmas"+format,
            )
            self.plot_labeled_spectra(
                self.prepare_spectra(["sumn"]),
                self.labels,
                ylabel="$N_i(m_{jj})/sum(N_i(m_{jj}))$",
                filename="sumn"+format,
            )
            self.plot_labeled_spectra(
                self.prepare_spectra(["maxn"]),
                self.labels,
                ylabel="$N_i(m_{jj})/max(N_i(m_{jj}))$",
                filename="maxn"+format,
            )
            self.plot_labeled_spectra(
                self.prepare_spectra(["maxn", "der", "med7"]),
                self.labels,
                ylabel=r"$\Delta(N_i(m_{jj})/max(N_i(m_{jj}))$)",
                filename="num-der"+format,
            )
            # TSNE
            csp.CS_TSNE(
                self.prepare_spectra(["maxn", "der", "med7"]).y,
                self.labels,
                self.eval_path,
            )
            # aggregations
            self.agg_sp["poor"].err = np.sqrt(self.agg_sp["poor"].y)
            self.agg_sp["rich"].err = np.sqrt(self.agg_sp["poor"].y)*0
            csp.plot_aggregation(
                self.agg_sp["poor"],
                self.agg_sp["rich"],
                self.figsize,
                self.agg_sp["res"],
                ts="max-sumnorm-dev-sr"
            )
            # curvefit_eval(
            #     self.agg_sp["poor"],
            #     self.agg_sp["rich"],
            #     self.binning,
            #     self.agg_sp["res"]["tf"],
            # )
            plt.savefig(self.eval_path + "comb"+format, bbox_inches="tight")
            csp.plot_aggregation(
                self.agg_sp["poor"].subtract_sp(self.agg_sp["poor"]),
                self.agg_sp["rich"].subtract_sp(self.agg_sp["poor"]),
                self.figsize,
                self.agg_sp["res"],
            )
            plt.savefig(self.eval_path + "comb_dev"+format, bbox_inches="tight")
            self.plot_gaussianity_checks(self.prepare_spectra(["sumn", "standardize"]), self.labels)
        else:
            prepr = ["fix_low_stat_error", "sumn", "-bsumsumn"]
            plt.figure()
            sp = self.prepare_spectra(prepr)
            for i in range(len(sp.y)):
                plt.plot(sp.x, sp.y[i])
            plt.savefig(self.eval_path + "spectra_diffs"+format, bbox_inches="tight")

            plt.figure()
            for i in range(len(sp.y)):
                plt.plot(
                    sp.x,
                    sp.y[i] / sp.err[i],
                )
            plt.savefig(self.eval_path + "spectra_devs"+format, bbox_inches="tight")

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
        self, sp, labels, ylabel="y", filename="test.pdf", add_line=False
    ):
        csp.two_class_curves(
            sp.x,
            sp.y,
            labels,
            self.figsize,
        )
        if add_line:
            plt.axhline(self.threshold, color="red", alpha=0.2)
        plt.xlabel("Bin centre $m_{jj}$ [GeV]")
        plt.ylabel(ylabel)
        plt.legend()
        plt.savefig(self.eval_path + filename, bbox_inches="tight")

    def plot_standardisation_step(self, sp, labels, y_label="y", savefile="test.pdf"):

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
            label="mean with SD",
        )
        csp.plot_mean_deviat(
            sp.x,
            sp.mean_sp_rob().y[0],
            sp.std_sp_rob().y[0],
            label="robust mean with \n robust SD",
            color="orange",
            fillb=True,
        )
        plt.legend()
        plt.xlabel("Bin centre $m_{jj}$ [GeV]")
        plt.ylabel(y_label)
        plt.savefig(self.eval_path + savefile, bbox_inches="tight")

    def plot_gaussianity_checks(self, sp, labels):
        mass_bins = np.array([3000 + i*100 for i in range(17)])
        bg_spectra = sp.y[labels == 0]
        sg_spectra = sp.y[labels == 1]
            
        for bin in range(len(sp.x)):
            plt.figure()
            bins = np.linspace(-4, 4, 20+1)
            shapiro_p_value, ks_p_value, jarque_bera_p_value = self.test_sample_gausiianity(sp.y[:, bin])
            plt.hist([bg_spectra[:, bin], sg_spectra[:, bin]], bins=bins, density=True, stacked=True, alpha=0.5, label="Bkg+sig $p_{SW}$"+f"={shapiro_p_value:.2f}\n"+" $p_{KS}$"+f"={ks_p_value:.2f}"+" $p_{JB}$"+f"={jarque_bera_p_value:.2f}")
            x=np.linspace(-4, 4, 100)
            plt.hist(np.random.normal(loc=0, scale=1, size=10_000_000), bins=bins, label='Unit Gaussian', color='red', histtype='step', linewidth=1, density=True)
            plt.legend()
            #plt.yscale("log")
            plt.ylabel("Density")
            plt.title(f"bin {mass_bins[bin]}"+"$<m_{jj}<$"+f"{mass_bins[bin]}")
            plt.xlabel("Standardized normalized counts")
            plt.savefig(self.eval_path +f"gausianity_check_bin{bin}.png", bbox_inches="tight", dpi=250)

    @staticmethod
    def test_sample_gausiianity(sample):
        # Shapiro-Wilk Test
        shapiro_statistic, shapiro_p_value = stats.shapiro(sample)
        print(f"Shapiro-Wilk Test - Statistic: {shapiro_statistic}, p-value: {shapiro_p_value}")

        # Kolmogorov-Smirnov Test
        ks_statistic, ks_p_value = stats.kstest(sample, 'norm')
        print(f"Kolmogorov-Smirnov Test - Statistic: {ks_statistic}, p-value: {ks_p_value}")

        # Jarque-Bera Test
        jarque_bera_statistic, jarque_bera_p_value = stats.jarque_bera(sample)
        print(f"Jarque-Bera Test - Statistic: {jarque_bera_statistic}, p-value: {jarque_bera_p_value}")
        return shapiro_p_value, ks_p_value, jarque_bera_p_value

def cs_performance_evaluation(*args, **kwargs):
    CSE = CS_evaluation_process(*args, **kwargs)
    res = CSE.run()
    return res

def load_counts_windows(path):
    res = pickle.load(open(path, "rb"))
    if isinstance(res, list) or isinstance(res, np.ndarray):
        return res
    else:
        return res["counts_windows"]

def cs_performance_evaluation_single(experiment_path, binning_folder, bres_id, config_path, tstat_name):
    path1 = experiment_path + binning_folder
    if os.path.exists(path1 + f"{tstat_name}/t_stat{bres_id}.npy"):
        print("already exists")
        return 
    if os.path.exists(path1 + f"bres_{bres_id}.pickle"):
        counts_windows = load_counts_windows(path1 + f"bres_{bres_id}.pickle")
    else:
        print(f"{path1}bres_{bres_id}.pickle file not found")
        return
    binning = pickle.load(open(path1 + "binning.pickle", "rb"))
    t_stat = cs_performance_evaluation(
        counts_windows=counts_windows,
        binning=binning,
        config_file_path=[config_path],
    )
    os.makedirs(path1 + f"{tstat_name}/", exist_ok=True)
    np.save(path1 + f"{tstat_name}/t_stat{bres_id}.npy", t_stat)



if __name__ == "__main__":
    cs_performance_evaluation_single(experiment_path="char/one_run_experiments/k50MB2048_1iret0con0.05W3000_3100_w0.5s1Nboot__/", 
                                     binning_folder="binnedW100s16ei30004600/", 
                                     bres_id="b0_s0_i0", 
                                     config_paths="config/cs_eval/maxdev5.yaml",
                                     tstat_name="maxdev5")
    
    # config_path = ["config/cs_eval/maxdev5.yaml", "config/cs_eval/plotting.yaml"]
    # jj = 0
    # path = "char/sig_reg/k50Trueret0con0.05W3450_3650_w0.5s1Nrest/"
    # # path1 = path + "binnedW100s200ei26006000/"
    # path1 = path + "binnedW100s16ei30004600/"
    # counts_windows = pickle.load(open(path1 + f"bres{jj}.pickle", "rb"))
    # binning = pickle.load(open(path1 + "binning.pickle", "rb"))
    # cs_performance_evaluation(
    #     counts_windows=counts_windows,
    #     binning=binning,
    #     path=path1,
    #     ID=jj,
    #     config_file_path=config_path,
    # )
    # print("Executed when invoked directly")

# Concepts
#     # all curves standardized
#     csp.two_class_curves(
#         window_centers,
#         sp_sumn_standrob.y * np.sqrt(countnrm_windows),
#         labels,
#         figsize,
#         ylabel="deviation in SD",
#         save_file=eval_path + "norm-toatal-sigmas-special.pdf",
#     )

#     # all curves standardized squeezed
#     sqf = 20
#     csp.two_class_curves(
#         squeeze(window_centers, sqf),
#         squeeze(sp_sumn_standrob.y, sqf),
#         labels,
#         figsize,
#         ylabel="deviation in SD",
#         save_file=eval_path + "norm-toatal-sigmas-squeezed.pdf",
#     )

#     # distribution of total MSE after standartisation
#     csp.two_class_curves(
#         window_centers,
#         sp_sumn_standrob.y,
#         labels,
#         figsize,
#         ylabel="deviation in SD",
#         save_file=eval_path + "norm-toatal-sigmas-special.pdf",
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
#         save_file=eval_path + "norm-toatal-sigmas-special.pdf",
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
#     plt.xlabel("Bin centre $m_{jj}$ [GeV]")
#     plt.ylabel("deviation in SD")
#     plt.colorbar()
#     plt.sca(axs[1])
#     yy = (yedges[:-1] + yedges[1:]) / 2
#     # for i in range(200):
#     plt.step(yy, np.mean(h, axis=0), where="mid", color="darkorange")
#     plt.xlabel("deviations in SD")
#     plt.ylabel("avarage curve points per bin")
#     plt.savefig(eval_path + "2d_hist.pdf", bbox_inches="tight")
