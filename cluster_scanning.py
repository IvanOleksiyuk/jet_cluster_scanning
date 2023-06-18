# imports
import sys
import os
import pickle
import random
import time
import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans, MiniBatchKMeans
import reprocessing
from utils.config_utils import Config
import shutil
import logging
import re


class ColoredFormatter(logging.Formatter):
    COLOR_CODE = {
        logging.DEBUG: "\033[36m",  # Cyan
        logging.INFO: "\033[32m",  # Green
        logging.WARNING: "\033[33m",  # Yellow
        logging.ERROR: "\033[31m",  # Red
        logging.CRITICAL: "\033[35m",  # Magenta
    }
    RESET_CODE = "\033[0m"  # Reset color

    def format(self, record):
        color = self.COLOR_CODE.get(record.levelno, "")
        reset = self.RESET_CODE
        message = super().format(record)
        return f"{color}{message}{reset}"


# Configure the logger
logger = logging.getLogger()

# Create a custom colored handler
handler = logging.StreamHandler()
handler.setFormatter(ColoredFormatter())

# Add the colored handler to the logger
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)
logging.getLogger("matplotlib").setLevel(logging.WARNING)


class ClusterScanning:
    def __init__(self, config_file_path):
        self.config_file_path = config_file_path
        self.config = Config(config_file_path)
        self.cofdict = self.config.get_dict()
        if not "n_init" in self.cofdict:
            self.cofdict["n_init"] = "auto"
        self.cfg = self.config.get_dotmap()
        self.reproc = reprocessing.Reprocessing(self.cfg.reproc_arg_string)
        self.cfg.reproc_name = self.reproc.name

        if self.cfg.MiniBatch:
            MB_str = f"MB{self.cfg.batch_size}"
        else:
            MB_str = "Lloyd"

        self.save_path = self.cfg.save_path + (
            f"k{self.cfg.k}"
            f"{MB_str}"
            f"_{self.cfg.n_init}i"
            f"ret{self.cfg.retrain}"
            f"con{self.cfg.signal_fraction}"
            f"W{self.cfg.train_interval[0]}_{self.cfg.train_interval[1]}_"
            f"{self.cfg.reproc_name}"
        )

        if self.cfg.bootstrap:
            self.save_path += "boot/"
        else:
            self.save_path += "rest/"

        self.Mjjmin_arr = np.linspace(
            self.cfg.eval_interval[0],
            self.cfg.eval_interval[1] - self.cfg.W,
            self.cfg.steps,
        )
        self.Mjjmax_arr = self.Mjjmin_arr + self.cfg.W
        HP = self.config.get_dict()

        self.bootstrap_bg = None  # by default when bootstrap resampling is needed a corresponding function will be called

        if hasattr(self.cfg, "bootstrap_ID"):
            self.def_IDb = self.cfg.bootstrap_ID  # bootstrap ID
        else:
            self.def_IDb = -1

        if hasattr(self.cfg, "signal_sample_ID"):
            self.def_IDs = self.cfg.signal_sample_ID  # signal sample ID
        else:
            self.def_IDs = 0

        if hasattr(self.cfg, "ID"):
            self.def_IDi = self.cfg.ID
        else:
            self.def_IDi = 0  # signal ID

    @staticmethod
    def seed(ID):
        if isinstance(ID, int):
            random.seed(a=ID, version=2)  # set a seed corresponding to the ID
            np.random.seed(ID)
        else:
            # a simple way of generating a seed from a list of IDs
            # UNFORTUNATELY HASHES ARE NOT UNIQUE TO THE LIST OF IDS but should do for now
            isinstance(ID, list)
            sum_hash = 0
            for i in ID:
                random.seed(a=ID[i], version=2)
                sum_hash += random.randint(0, 10e6)
            random.seed(sum_hash)
            np.random.seed(sum_hash)

    @staticmethod
    def IDstr(IDb, IDs, IDi):
        return f"_b{IDb}_s{IDs}_i{IDi}"

    def load_mjj(self):
        self.mjj_bg = np.load(self.cfg.data_path + "mjj_bkg_sort.npy")
        self.mjj_sg = np.load(self.cfg.data_path + "mjj_sig_sort.npy")
        if not np.all(self.mjj_bg[:-1] <= self.mjj_bg[1:]):
            logging.error("background masses are not sorted!")
            sys.exit()
        if not np.all(self.mjj_sg[:-1] <= self.mjj_sg[1:]):
            logging.error("signal masses are not sorted!")
            sys.exit()

    def stack_event(self, x):
        print("NOT IMPLEMENTED YET")  # TODO
        return

    def flatten_event(self, x):
        print("NOT IMPLEMENTED YET")  # TODO
        return x.reshape()

    def flatten_image(self):
        print("NOT IMPLEMENTED YET")  # TODO
        pass

    def de_flatten_image(self):
        print("NOT IMPLEMENTED YET")  # TODO
        pass

    def load_data(self, show_example=True):
        logging.info("loading data")
        im_bg_file = h5py.File(self.cfg.data_path + "v2JetImSort_bkg.h5", "r")
        im_sg_file = h5py.File(self.cfg.data_path + "v2JetImSort_sig.h5", "r")
        self.im_bg = im_bg_file["data"]
        self.im_sg = im_sg_file["data"]
        if self.cfg.memory_intensive:
            # if self.load_record and os.path.exists("data_record"+self.cfg.reproc_arg_string+".npy"
            self.im_bg = self.reproc(
                im_bg_file["data"][:].reshape((-1, self.cfg.image_w, self.cfg.image_h))
            ).reshape(
                (
                    -1,
                    self.cfg.jet_per_event,
                    self.cfg.image_w * self.cfg.image_h,
                )
            )
            # np.save("data_record"+self.cfg.reproc_arg_string, self.im_bg)
            self.im_sg = self.reproc(
                im_sg_file["data"][:].reshape((-1, self.cfg.image_w, self.cfg.image_h))
            ).reshape(
                (
                    -1,
                    self.cfg.jet_per_event,
                    self.cfg.image_size * self.cfg.image_size,
                )
            )
            # np.save("data_record"+self.cfg.reproc_arg_string, self.im_sg)
            im_bg_file.close()
            im_sg_file.close()
            if show_example:
                plt.figure()
                plt.imshow(
                    self.im_bg[1:3].reshape(
                        (-1, self.cfg.image_size, self.cfg.image_size)
                    )[0]
                )
        self.bootstrap_bg = None
        logging.info("loading data complete")

    def data_mjj_slise(self, Mjjmin, Mjjmax):
        """Returns the background an signal jets in a given Mjj window

        Args:
            Mjjmin (float): lower Mjj interval limit
            Mjjmax (float): upper Mjj interval limit
        Returns:
            _type_: _description_
        """
        logging.info(f"loading window {Mjjmin} {Mjjmax}")
        indexing_bg = np.logical_and(self.mjj_bg >= Mjjmin, self.mjj_bg <= Mjjmax)
        indexing_bg = np.where(indexing_bg)[0]

        indexing_sg = np.logical_and(self.mjj_sg >= Mjjmin, self.mjj_sg <= Mjjmax)
        indexing_sg = np.where(indexing_sg)[0]

        # TODO: DELETE logs below
        # logging.info(len(indexing_bg), "bg events found in interval")
        # logging.info(len(indexing_sg), "sg events found in interval")

        start_time = time.time()
        logging.info("start data extraction")
        if self.bootstrap_bg is None:
            bg = self.im_bg[indexing_bg[0] : indexing_bg[-1]]
        else:
            if self.cfg.verbose:
                logging.info(len(self.im_bg[indexing_bg[0] : indexing_bg[-1]]))
                logging.info(len(self.bootstrap_bg[indexing_bg[0] : indexing_bg[-1]]))
            bg = np.repeat(
                self.im_bg[indexing_bg[0] : indexing_bg[-1]],
                self.bootstrap_bg[indexing_bg[0] : indexing_bg[-1]],
                axis=0,
            )

        if self.allowed is not None:
            sg = np.repeat(
                self.im_sg[indexing_sg[0] : indexing_sg[-1]],
                self.allowed[indexing_sg[0] : indexing_sg[-1]],
                axis=0,
            )
            logging.info(f"only {len(sg)} sg events taken")
        # test if chnaging to this spares some time
        # sg=im_sg[indexing_sg[0]:indexing_sg[-1]]
        # sg=sg[allowed[indexing_sg[0]:indexing_sg[-1]]]
        logging.info(f"only {len(bg)} bg events taken")
        logging.debug("load --- %s seconds ---" % (time.time() - start_time))
        start_time = time.time()

        if self.allowed is not None:
            data = np.concatenate((bg, sg))
        else:
            data = bg

        data = data.reshape(
            (
                len(data) * self.cfg.jet_per_event,
                self.cfg.image_size,
                self.cfg.image_size,
            )
        )
        if not self.cfg.memory_intensive:
            logging.debug("concat --- %s seconds ---" % (time.time() - start_time))
            start_time = time.time()
            data = self.reproc(data)
        data = data.reshape((len(data), self.cfg.image_size**2))
        logging.info("reproc --- %s seconds ---" % (time.time() - start_time))
        return data

    def train_k_means(self, ID):
        start_time = time.time()
        self.seed(ID)
        self.__ID = ID[-1]
        if self.cfg.MiniBatch:
            self.kmeans = MiniBatchKMeans(
                self.cfg.k,
                batch_size=self.cfg.batch_size,
                n_init=self.cfg.n_init,
                init="k-means++",
            )
        else:
            self.kmeans = KMeans(self.cfg.k, n_init=self.cfg.n_init, init="k-means++")

        # Train k-means in the training window
        data = self.data_mjj_slise(
            self.cfg.train_interval[0], self.cfg.train_interval[1]
        )
        self.kmeans.fit(data)
        counts = np.bincount(self.kmeans.labels_)
        counts.sort()
        logging.info(f"sorted cluster counts {counts}")
        logging.info(f"iterations {self.kmeans.n_iter_}")
        logging.info("training --- %s seconds ---" % (time.time() - start_time))

    def bootstrap_resample(self, ID=None):
        if ID is None:
            ID = self.__bsID

        if self.cfg.bootstrap and ID != -1:
            self.seed(ID)
            self.__bsID = ID
            n = len(self.mjj_bg)
            np.sort(
                np.random.randint(0, n, (n,))
            )  # this is the line that takes time and does nothing but if I remove it the incosistensies will begin because the random seed will be different
            a = np.arange(n)
            self.bootstrap_bg = np.bincount(np.random.choice(a, (n,)), minlength=n)
            logging.debug(f"performed bootstrap resampling with ID {ID}")
        else:
            self.__bsID = -1
            if not self.cfg.bootstrap:
                logging.debug(
                    "bootstrap is not set to true in config file => ignoring bootstrap"
                )
            else:
                if ID < -1:
                    logging.error("bootstrap ID is smaller -1 which is not allowed!")
                elif ID == -1:
                    logging.debug(
                        "bootstrap ID is set to -1 in config file => ignoring bootstrap"
                    )

    def cancel_bootstrap_resampling(self):
        self.bg_bootstrap = None

    def sample_signal_events(self, ID=None):
        if ID is None:
            ID = self.__sigID
        self.seed(ID)
        self.__sigID = ID
        if self.cfg.signal_fraction > 0:
            num_true = int(np.rint(self.cfg.signal_fraction * len(self.mjj_sg)))
            self.allowed = np.concatenate(
                (
                    np.zeros(len(self.mjj_sg) - num_true, dtype=bool),
                    np.ones(num_true, dtype=bool),
                )
            )
            np.random.shuffle(self.allowed)
        else:
            self.allowed = None

    def evaluate_whole_dataset(self):
        start_time = time.time()
        if self.cfg.memory_intensive:
            self.bg_lab = self.kmeans.predict(
                self.im_bg.reshape((-1, self.cfg.image_w * self.cfg.image_h))
            )
            self.bg_lab = self.bg_lab.reshape((-1, self.cfg.jet_per_event))
            self.sg_lab = self.kmeans.predict(
                self.im_sg.reshape((-1, self.cfg.image_w * self.cfg.image_h))
            )
            self.sg_lab = self.sg_lab.reshape((-1, self.cfg.jet_per_event))
        else:
            self.bg_lab = []
            batch_size = 10000
            for i in range(int(np.ceil(len(self.im_bg) / batch_size))):
                logging.debug(
                    i * batch_size, min((i + 1) * batch_size, len(self.im_bg))
                )
                self.bg_lab.append(
                    self.kmeans.predict(
                        self.reproc(
                            self.im_bg[
                                i
                                * batch_size : min(
                                    (i + 1) * batch_size, len(self.im_bg)
                                )
                            ].reshape((-1, self.cfg.image_w, self.cfg.image_h))
                        ).reshape((-1, self.cfg.image_w * self.cfg.image_h))
                    ).reshape((-1, self.cfg.jet_per_event))
                )
            self.bg_lab = np.concatenate(self.bg_lab)

            self.sg_lab = []
            batch_size = 10000
            for i in range(int(np.ceil(len(self.im_sg) / batch_size))):
                logging.debug(
                    i * batch_size,
                    min((i + 1) * batch_size, len(self.im_bg)),
                )
                self.sg_lab.append(
                    self.kmeans.predict(
                        self.reproc(
                            self.im_sg[
                                i
                                * batch_size : min(
                                    (i + 1) * batch_size, len(self.im_sg)
                                )
                            ].reshape((-1, self.cfg.image_w, self.cfg.image_h))
                        ).reshape((-1, self.cfg.image_w * self.cfg.image_h))
                    ).reshape((-1, self.cfg.jet_per_event))
                )
            self.sg_lab = np.concatenate(self.sg_lab)
        logging.info("label_eval --- %s seconds ---" % (time.time() - start_time))

    def count_bin(self, mjjmin, mjjmax, allowed, bootstrap_bg):
        """Counts a number of events for all classes in a given Mjj window

        Args:
            mjjmin (float): lower Mjj interval limit
            mjjmax (float): upper Mjj interval limit
            allowed (list/array of integers or bool of size len(im_sg)):
                Indicates which and how any times each signal image is chosen for the dataset
            bootstrap_bg (list/array of integers of size len(im_bg)):
                Indicates which and how any times each background image is chosen for the dataset

        Returns:
            np.array(dtype=int): number of jets in each cluster in this bin.
        """
        indexing_bg = np.logical_and(self.mjj_bg >= mjjmin, self.mjj_bg < mjjmax)
        indexing_bg = np.where(indexing_bg)[0]
        indexing_sg = np.logical_and(self.mjj_sg >= mjjmin, self.mjj_sg < mjjmax)
        indexing_sg = np.where(indexing_sg)[0]

        # TODO: DELETE logs  below
        # logging.info(len(indexing_bg), "bg events found in interval")
        # logging.info(len(indexing_sg), "sg events found in interval")

        if indexing_bg != []:
            if bootstrap_bg is None:
                bg = self.bg_lab[indexing_bg[0] : indexing_bg[-1] + 1]
            else:
                logging.debug(len(self.bg_lab[indexing_bg[0] : indexing_bg[-1]]))
                logging.debug(len(bootstrap_bg[indexing_bg[0] : indexing_bg[-1]]))
                bg = np.repeat(
                    self.bg_lab[indexing_bg[0] : indexing_bg[-1]],
                    bootstrap_bg[indexing_bg[0] : indexing_bg[-1]],
                    axis=0,
                )
        else:
            bg = np.array([])
            logger.warning("no background events in this window")

        if allowed is not None:
            sg = np.repeat(
                self.sg_lab[indexing_sg[0] : indexing_sg[-1]],
                allowed[indexing_sg[0] : indexing_sg[-1]],
                axis=0,
            )

            if self.cfg.separate_binning:
                return [
                    np.array([np.sum(bg == j) for j in range(self.cfg.k)]),
                    np.array([np.sum(sg == j) for j in range(self.cfg.k)]),
                ]
            else:
                all_lab = np.concatenate((bg, sg))
                return np.array([np.sum(all_lab == j) for j in range(self.cfg.k)])
        else:
            all_lab = bg
            return np.array([np.sum(all_lab == j) for j in range(self.cfg.k)])

    def perform_binning(self):
        counts_windows = []
        for i in range(self.cfg.steps):
            counts_windows.append(
                self.count_bin(
                    self.Mjjmin_arr[i],
                    self.Mjjmax_arr[i],
                    self.allowed,
                    self.bootstrap_bg,
                )
            )

        if self.cfg.separate_binning:
            self.counts_windows_bg = np.stack([x[0] for x in counts_windows])
            self.counts_windows_sg = np.stack([x[1] for x in counts_windows])
            self.counts_windows = [self.counts_windows_bg, self.counts_windows_sg]
            self.counts_windows_sum = sum(self.counts_windows)
        else:
            self.counts_windows = [np.stack(counts_windows)]
            self.counts_windows_sum = np.stack(counts_windows)
        return self.counts_windows

    def make_plots(self):
        # Some plotting
        plots_path = self.save_path + f"plots{self.get_IDstr()}/"
        os.makedirs(plots_path, exist_ok=True)

        window_centers = (self.Mjjmin_arr + self.Mjjmax_arr) / 2
        min_allowed_count = 100
        min_min_allowed_count = 10
        plt.figure()
        plt.grid()
        for j in range(self.cfg.k):
            plt.plot(window_centers, self.counts_windows_sum[:, j])
        plt.xlabel("m_jj")
        plt.ylabel("n points from window")
        plt.savefig(plots_path + "kmeans_ni_mjj_total.png")
        smallest_cluster_count_window = np.min(self.counts_windows_sum, axis=1)
        for i in range(len(window_centers)):
            if smallest_cluster_count_window[i] < min_allowed_count:
                if smallest_cluster_count_window[i] < min_min_allowed_count:
                    plt.axvline(window_centers[i], color="black", alpha=0.6)
                else:
                    plt.axvline(window_centers[i], color="black", alpha=0.3)

        plt.savefig(plots_path + "kmeans_ni_mjj_total_statAllowed.png")

        partials_windows = np.zeros(self.counts_windows_sum.shape)
        for i in range(len(self.Mjjmin_arr)):
            partials_windows[i, :] = self.counts_windows_sum[i, :] / np.sum(
                self.counts_windows_sum[i, :]
            )

        plt.figure()
        plt.grid()
        for j in range(self.cfg.k):
            plt.plot(window_centers, partials_windows[:, j])
        plt.xlabel("m_jj")
        plt.ylabel("fraction of points in window")
        plt.savefig(plots_path + "kmeans_xi_mjj_total.png")

        countmax_windows = np.zeros(self.counts_windows_sum.shape)
        for i in range(self.cfg.k):
            countmax_windows[:, i] = self.counts_windows_sum[:, i] / np.max(
                self.counts_windows_sum[:, i]
            )

        plt.figure()
        plt.grid()
        for j in range(self.cfg.k):
            plt.plot(window_centers, countmax_windows[:, j])

        conts_bg = []
        conts_sg = []
        for Mjjmin, Mjjmax in zip(self.Mjjmin_arr, self.Mjjmax_arr):
            conts_bg.append(
                np.sum(np.logical_and(self.mjj_bg >= Mjjmin, self.mjj_bg <= Mjjmax))
            )
            if self.allowed is not None:
                conts_sg.append(
                    np.sum(
                        np.logical_and(
                            np.logical_and(
                                self.mjj_sg >= Mjjmin, self.mjj_sg <= Mjjmax
                            ),
                            self.allowed,
                        )
                    )
                )
            else:
                conts_sg.append(0)
        conts_bg = np.array(conts_bg)
        conts_sg = np.array(conts_sg)
        conts = conts_bg + conts_sg
        plt.plot(window_centers, conts / np.max(conts), "--")
        plt.xlabel("m_jj")
        plt.ylabel("n points from window/max(...)")
        plt.savefig(plots_path + "kmeans_xi_mjj_maxn.png")

        for i in range(len(window_centers)):
            if smallest_cluster_count_window[i] < min_allowed_count:
                if smallest_cluster_count_window[i] < min_min_allowed_count:
                    plt.axvline(window_centers[i], color="black", alpha=0.6)
                else:
                    plt.axvline(window_centers[i], color="black", alpha=0.3)

        plt.savefig(plots_path + "kmeans_xi_mjj_maxn_statAllowed.png")

        countnrm_windows = np.zeros(self.counts_windows_sum.shape)
        for i in range(self.cfg.k):
            countnrm_windows[:, i] = self.counts_windows_sum[:, i] / np.sum(
                self.counts_windows_sum[:, i]
            )

        plt.figure()
        plt.grid()
        for j in range(self.cfg.k):
            plt.plot(window_centers, countnrm_windows[:, j])
        plt.xlabel("window centre $m_{jj}$ [GeV]")
        plt.ylabel("$N_i(m_{jj})/sum(N_i(m_{jj}))$")
        plt.savefig(plots_path + "kmeans_ni_mjj_norm.png", bbox_inches="tight")
        for i in range(len(window_centers)):
            if smallest_cluster_count_window[i] < min_allowed_count:
                if smallest_cluster_count_window[i] < min_min_allowed_count:
                    plt.axvline(window_centers[i], color="black", alpha=0.6)
                else:
                    plt.axvline(window_centers[i], color="black", alpha=0.3)

        plt.savefig(
            plots_path + "kmeans_ni_mjj_norm_statAllowed.png",
            bbox_inches="tight",
        )

    def run(self):
        os.makedirs(self.save_path, exist_ok=True)
        if isinstance(self.config_file_path, str):
            shutil.copy2(self.config_file_path, self.save_path + "config.yaml")
        elif isinstance(self.config_file_path, list):
            for i, path in enumerate(self.config_file_path):
                shutil.copy2(path, self.save_path + f"config_{i}.yaml")
        self.config.write(self.save_path + "confsum.yaml")

        if self.cfg.restart or self.cfg.bootstrap or self.cfg.resample_signal:
            self.multi_run()
        else:
            self.single_run()

    def save_results(self):
        with open(self.save_path + f"lab{self.get_IDstr()}.pickle", "wb") as file:
            pickle.dump(
                {"bg": self.bg_lab, "sg": self.sg_lab, "k_means": self.kmeans},
                file,
            )
        logging.debug(f"saved results to {self.save_path}")

    def load_results(self, IDb=None, IDs=None, IDi=None, IDstr=None):
        if IDstr is not None:
            IDb, IDs, IDi = self.IDstr_to_IDs(IDstr)

        if IDb is None:
            self.__bsID = self.def_IDb
        else:
            self.__bsID = IDb

        if IDs is None:
            self.__sigID = self.def_IDs
        else:
            self.__sigID = IDs
        if IDi is None:
            self.__ID = self.def_IDi
        else:
            self.__ID = IDi
        with open(self.save_path + f"lab{self.get_IDstr()}.pickle", "rb") as file:
            res = pickle.load(file)
        self.bg_lab = res["bg"]
        self.sg_lab = res["sg"]
        self.kmeans = res["k_means"]
        # TODO add loading of trained k-means

    def single_run(self):
        start_time = time.time()
        self.load_mjj()
        self.load_data()
        self.bootstrap_resample(self.def_IDb)
        # will do nothing if self.cfg.bootstrap=False or self.def_IDb=-1
        self.sample_signal_events(self.def_IDs)
        # will do nothing if self.cfg.resample_signal=False
        IDs = [self.def_IDb, self.def_IDs, self.def_IDi]
        self.train_k_means(IDs)
        self.evaluate_whole_dataset()
        self.save_results()
        self.perform_binning()
        self.make_plots()
        # plt.show()
        logging.info("Run completed in ### %s seconds ###" % (time.time() - start_time))

    def multi_run(self):
        start_time = time.time()
        self.load_mjj()
        self.load_data()
        logging.info(
            f"Starting multirun with {len(self.list_runs_to_be_done())} runs to be done"
        )
        for IDs in self.list_runs_to_be_done():
            logging.debug(
                f"Starting {self.IDstr(*IDs)} ### %s seconds ###"
                % (time.time() - start_time)
            )
            self.bootstrap_resample(
                IDs[0]
            )  # will do nothing if IDs[0]=-1 or self.cfg.bootstrap=False
            self.sample_signal_events(
                IDs[1]
            )  # will do nothing if IDs[1]=0 or self.cfg.resample_signal=False
            self.train_k_means(IDs)
            self.evaluate_whole_dataset()
            self.save_results()
            logging.info(
                f"Done {self.get_IDstr()} ### %s seconds ###"
                % (time.time() - start_time)
            )

    def list_runs_to_be_done(self):
        ID_tuple_list = []

        if self.cfg.bootstrap:
            IDb_arr = [i for i in range(self.cfg.IDb_start, self.cfg.IDb_finish)]
        else:
            IDb_arr = [self.def_IDb]

        if self.cfg.resample_signal:
            IDs_arr = [i for i in range(self.cfg.IDs_start, self.cfg.IDs_finish)]
        else:
            IDs_arr = [self.def_IDs]

        if self.cfg.restart:
            IDi_arr = [i for i in range(self.cfg.IDi_start, self.cfg.IDi_finish)]
        else:
            IDi_arr = [self.def_IDi]

        for IDb in IDb_arr:
            for IDs in IDs_arr:
                for IDi in IDi_arr:
                    if not os.path.exists(
                        self.save_path + f"lab{self.IDstr(IDb, IDs, IDi)}.pickle"
                    ):
                        ID_tuple_list.append([IDb, IDs, IDi])
        if len(ID_tuple_list) == 0:
            logging.info("No runs to be done")
        else:
            logging.debug(f"First in the list to do is {ID_tuple_list[0]}")
            logging.debug(f"Last in the list to do is {ID_tuple_list[-1]}")
        return ID_tuple_list

    def IDstr_to_IDs(self, IDstr):
        integers = re.findall(r"-?\d+", IDstr)
        integers = [int(num) for num in integers]
        return integers

    def get_IDstr(self):
        return self.IDstr(self.__bsID, self.__sigID, self.__ID)

    def counts_windows_path(self, directory=False, IDstr=None):
        pathh = (
            self.save_path
            + f"binnedW{self.cfg.W}s{self.cfg.steps}ei{self.cfg.eval_interval[0]}{self.cfg.eval_interval[1]}"
        )
        if self.bootstrap_bg is not None:
            pathh += f"boot/"
        else:
            pathh += "/"
        if directory:
            return pathh
        else:
            if IDstr is None:
                return pathh + f"bres{self.get_IDstr()}.pickle"
            else:
                return pathh + f"bres{IDstr}.pickle"

    def save_counts_windows(self):
        os.makedirs(self.counts_windows_path(directory=True), exist_ok=True)
        res = {}
        res["counts_windows"] = self.counts_windows
        res["inertia"] = self.kmeans.inertia_
        with open(self.counts_windows_path(), "wb") as file:
            pickle.dump(res, file)

    def available_IDstr(self):
        # TODO redo with glob.glob?
        IDstr_avail = []
        for file in os.listdir(self.save_path):
            if file.startswith("lab"):
                IDstr_avail.append(file[3:-7])
        return IDstr_avail

    def check_if_binning_exist(self, IDstr):
        pathh = self.counts_windows_path(IDstr=IDstr)
        return os.path.exists(pathh)

    def save_binning_array(self):
        os.makedirs(self.counts_windows_path(directory=True), exist_ok=True)
        with open(
            self.counts_windows_path(directory=True) + "binning.pickle", "wb"
        ) as file:
            binning = np.stack([self.Mjjmin_arr, self.Mjjmax_arr]).T
            pickle.dump(binning, file)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        config_file_path = [
            "config/s0_0.5_1_MB_new.yaml",
            "config/sig_frac/0.01.yaml",
            "config/multirun/i0_10.yaml",
            "config/tra_reg/3000_3100.yaml",
            "config/binning/CURTAINS.yaml",
            # "config/tra_reg/sig_reg.yaml",
            "config/v4.yaml",
        ]
    else:
        config_file_path = sys.argv[1:]
    logging.info("starting" + str(config_file_path))
    cs = ClusterScanning(config_file_path)
    cs.run()
