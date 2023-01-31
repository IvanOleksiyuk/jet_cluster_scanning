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
from config_utils import Config
import shutil


class ClusterScanning:
    def __init__(self, config_file_path):
        self.config_file_path = config_file_path
        self.config = Config(config_file_path)
        self.cfg = self.config.get_dotmap()
        self.reproc = reprocessing.Reprocessing(self.cfg.reproc_arg_string)
        self.cfg.reproc_name = self.reproc.name

        self.save_path = self.cfg.save_path + (
            f"k{self.cfg.k}"
            f"{self.cfg.MiniBatch}"
            f"ret{self.cfg.retrain}"
            f"con{self.cfg.signal_fraction}"
            f"W{self.cfg.train_interval[0]}_{self.cfg.train_interval[1]}_"
            f"{self.cfg.reproc_name}"
        )

        if self.cfg.bootstrap:
            self.save_path += "boot/"
        else:
            self.save_path += "rest/"

        if not self.cfg.restart:
            self.ID = self.cfg.ID

        self.Mjjmin_arr = np.linspace(
            self.cfg.eval_interval[0],
            self.cfg.eval_interval[1] - self.cfg.W,
            self.cfg.steps,
        )
        self.Mjjmax_arr = self.Mjjmin_arr + self.cfg.W
        HP = self.config.get_dict()

        self.bootstrap_bg = None  # by default when bootstrap resampling is needed a corresponding function will be called

    def seed(self):
        random.seed(a=self.ID, version=2)  # set a seed corresponding to the ID
        np.random.seed(self.ID)

    def load_mjj(self):
        self.mjj_bg = np.load(self.cfg.data_path + "mjj_bkg_sort.npy")
        self.mjj_sg = np.load(self.cfg.data_path + "mjj_sig_sort.npy")

    def stack_event(self, x):
        return

    def flatten_event(self, x):
        return x.reshape()

    def flatten_image(self):
        pass

    def de_flatten_image(self):
        pass

    def load_data(self, show_example=True):
        im_bg_file = h5py.File(self.cfg.data_path + "v2JetImSort_bkg.h5", "r")
        im_sg_file = h5py.File(self.cfg.data_path + "v2JetImSort_sig.h5", "r")
        self.im_bg = im_bg_file["data"]
        self.im_sg = im_sg_file["data"]
        if self.cfg.memory_intensive:
            self.im_bg = self.reproc(
                im_bg_file["data"][:].reshape((-1, self.cfg.image_w, self.cfg.image_h))
            ).reshape(
                (
                    -1,
                    self.cfg.jet_per_event,
                    self.cfg.image_w * self.cfg.image_h,
                )
            )
            self.im_sg = self.reproc(
                im_sg_file["data"][:].reshape((-1, self.cfg.image_w, self.cfg.image_h))
            ).reshape(
                (
                    -1,
                    self.cfg.jet_per_event,
                    self.cfg.image_size * self.cfg.image_size,
                )
            )
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

    def data_mjj_slise(self, Mjjmin, Mjjmax):
        """Returns the background an signal jets in a given Mjj window

        Args:
            Mjjmin (float): lower Mjj interval limit
            Mjjmax (float): upper Mjj interval limit
        Returns:
            _type_: _description_
        """
        print("loading window", Mjjmin, Mjjmax)
        indexing_bg = np.logical_and(self.mjj_bg >= Mjjmin, self.mjj_bg <= Mjjmax)
        indexing_bg = np.where(indexing_bg)[0]

        indexing_sg = np.logical_and(self.mjj_sg >= Mjjmin, self.mjj_sg <= Mjjmax)
        indexing_sg = np.where(indexing_sg)[0]

        # TODO: DELETE prints  below
        # print(len(indexing_bg), "bg events found in interval")
        # print(len(indexing_sg), "sg events found in interval")

        start_time = time.time()
        print("start data extraction")
        if self.bootstrap_bg is None:
            bg = self.im_bg[indexing_bg[0] : indexing_bg[-1]]
        else:
            if self.cfg.verbose:
                print(len(self.im_bg[indexing_bg[0] : indexing_bg[-1]]))
                print(len(self.bootstrap_bg[indexing_bg[0] : indexing_bg[-1]]))
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
            print("only", len(sg), "sg events taken")
        # test if chnaging to this spares some time
        # sg=im_sg[indexing_sg[0]:indexing_sg[-1]]
        # sg=sg[allowed[indexing_sg[0]:indexing_sg[-1]]]
        print("only", len(bg), "bg events taken")
        print("load --- %s seconds ---" % (time.time() - start_time))
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
            print("concat --- %s seconds ---" % (time.time() - start_time))
            start_time = time.time()
            data = self.reproc(data)
        data = data.reshape((len(data), self.cfg.image_size**2))
        print("reproc --- %s seconds ---" % (time.time() - start_time))
        return data

    def train_k_means(self):
        self.seed()
        if self.cfg.MiniBatch:
            self.kmeans = MiniBatchKMeans(self.cfg.k)
        else:
            self.kmeans = KMeans(self.cfg.k)

        # Train k-means in the training window
        start_time = time.time()
        data = self.data_mjj_slise(
            self.cfg.train_interval[0], self.cfg.train_interval[1]
        )
        self.kmeans.fit(data)
        counts = np.bincount(self.kmeans.labels_)
        counts.sort()
        print("sorted cluster counts", counts)
        print("trained --- %s seconds ---" % (time.time() - start_time))

    def bootstrap_resample(self):
        if self.cfg.bootstrap:
            self.seed()
            n = len(self.mjj_bg)
            np.sort(
                np.random.randint(0, n, (n,))
            )  # this is the line that takes time and does nothing but if I remove it the incosistensies will begin because the random seed will be different
            a = np.arange(n)
            self.bootstrap_bg = np.bincount(np.random.choice(a, (n,)), minlength=n)
        else:
            print("bootstrap is not set to true in config file => ignoring bootstrap")

    def cancel_bootstrap_resampling(self):
        self.bg_bootstrap = None

    def sample_signal_events(self):
        self.seed()
        num_true = int(np.rint(self.cfg.signal_fraction * len(self.mjj_sg)))
        self.allowed = np.concatenate(
            (
                np.zeros(len(self.mjj_sg) - num_true, dtype=bool),
                np.ones(num_true, dtype=bool),
            )
        )
        np.random.shuffle(self.allowed)

    def evaluate_whole_dataset(self):
        if self.cfg.memory_intensive:
            self.bg_lab = self.kmeans.predict(
                self.im_bg.reshape((-1, self.cfg.image_w * self.cfg.image_h))
            )
            self.sg_lab = self.kmeans.predict(
                self.im_sg.reshape((-1, self.cfg.image_w * self.cfg.image_h))
            )
        else:
            self.bg_lab = []
            batch_size = 10000
            for i in range(int(np.ceil(len(self.im_bg) / batch_size))):
                if self.cfg.verbose:
                    print(i * batch_size, min((i + 1) * batch_size, len(self.im_bg)))
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
                if self.cfg.verbose:
                    print(
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
        indexing_bg = np.logical_and(self.mjj_bg >= mjjmin, self.mjj_bg <= mjjmax)
        indexing_bg = np.where(indexing_bg)[0]
        indexing_sg = np.logical_and(self.mjj_sg >= mjjmin, self.mjj_sg <= mjjmax)
        indexing_sg = np.where(indexing_sg)[0]

        # TODO: DELETE prints  below
        # print(len(indexing_bg), "bg events found in interval")
        # print(len(indexing_sg), "sg events found in interval")

        if bootstrap_bg is None:
            bg = self.bg_lab[indexing_bg[0] : indexing_bg[-1]]
        else:
            print(len(self.bg_lab[indexing_bg[0] : indexing_bg[-1]]))
            print(len(bootstrap_bg[indexing_bg[0] : indexing_bg[-1]]))
            bg = np.repeat(
                self.bg_lab[indexing_bg[0] : indexing_bg[-1]],
                bootstrap_bg[indexing_bg[0] : indexing_bg[-1]],
                axis=0,
            )

        if allowed is not None:
            sg = np.repeat(
                self.sg_lab[indexing_sg[0] : indexing_sg[-1]],
                allowed[indexing_sg[0] : indexing_sg[-1]],
                axis=0,
            )
            all_lab = np.concatenate((bg, sg))
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

        # print(len(counts_windows))
        self.counts_windows = np.stack(counts_windows)
        return self.counts_windows

    def make_plots(self):
        # Some plotting
        plots_path = self.save_path + f"plots{self.ID}/"
        os.makedirs(plots_path, exist_ok=True)

        window_centers = (self.Mjjmin_arr + self.Mjjmax_arr) / 2
        min_allowed_count = 100
        min_min_allowed_count = 10
        plt.figure()
        plt.grid()
        for j in range(self.cfg.k):
            plt.plot(window_centers, self.counts_windows[:, j])
        plt.xlabel("m_jj")
        plt.ylabel("n points from window")
        plt.savefig(plots_path + "kmeans_ni_mjj_total.png")
        smallest_cluster_count_window = np.min(self.counts_windows, axis=1)
        for i in range(len(window_centers)):
            if smallest_cluster_count_window[i] < min_allowed_count:
                if smallest_cluster_count_window[i] < min_min_allowed_count:
                    plt.axvline(window_centers[i], color="black", alpha=0.6)
                else:
                    plt.axvline(window_centers[i], color="black", alpha=0.3)

        plt.savefig(plots_path + "kmeans_ni_mjj_total_statAllowed.png")

        partials_windows = np.zeros(self.counts_windows.shape)
        for i in range(len(self.Mjjmin_arr)):
            partials_windows[i, :] = self.counts_windows[i, :] / np.sum(
                self.counts_windows[i, :]
            )

        plt.figure()
        plt.grid()
        for j in range(self.cfg.k):
            plt.plot(window_centers, partials_windows[:, j])
        plt.xlabel("m_jj")
        plt.ylabel("fraction of points in window")
        plt.savefig(plots_path + "kmeans_xi_mjj_total.png")

        countmax_windows = np.zeros(self.counts_windows.shape)
        for i in range(self.cfg.k):
            countmax_windows[:, i] = self.counts_windows[:, i] / np.max(
                self.counts_windows[:, i]
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
            conts_sg.append(
                np.sum(
                    np.logical_and(
                        np.logical_and(self.mjj_sg >= Mjjmin, self.mjj_sg <= Mjjmax),
                        self.allowed,
                    )
                )
            )
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

        countnrm_windows = np.zeros(self.counts_windows.shape)
        for i in range(self.cfg.k):
            countnrm_windows[:, i] = self.counts_windows[:, i] / np.sum(
                self.counts_windows[:, i]
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

        if self.cfg.restart:
            self.multi_run()
        else:
            self.single_run()

    def save_results(self):
        with open(self.save_path + f"lab{self.ID}.pickle", "wb") as file:
            pickle.dump(
                {"bg": self.bg_lab, "sg": self.sg_lab, "k_means": self.kmeans},
                file,
            )

    def load_results(self, IDb=""):
        with open(self.save_path + f"lab{IDb}.pickle", "rb") as file:
            res = pickle.load(file)
        self.bg_lab = res["bg"]
        self.sg_lab = res["sg"]
        # TODO add loading of trained k-means

    def single_run(self):
        start_time = time.time()
        self.load_mjj()
        self.load_data()
        self.sample_signal_events()
        if self.cfg.bootstrap:
            self.bootstrap_resample()
        self.train_k_means()
        self.evaluate_whole_dataset()
        self.save_results()
        self.perform_binning()
        self.make_plots()
        # plt.show()
        print("All done ### %s seconds ###" % (time.time() - start_time))

    def multi_run(self):
        start_time = time.time()
        self.load_mjj()
        self.load_data()
        for IDb in range(self.cfg.restart_ID_start, self.cfg.restart_ID_finish):
            if os.path.exists(self.save_path + f"lab{IDb}.pickle"):
                print(f"IDb {IDb} already exists")
                continue
            self.ID = IDb
            self.sample_signal_events()
            if self.cfg.bootstrap:
                self.bootstrap_resample()
            self.train_k_means()
            self.evaluate_whole_dataset()
            self.save_results()
            print(f"Done IDb {IDb} ### %s seconds ###" % (time.time() - start_time))

    def counts_windows_path(self, directory=False):
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
            return pathh + f"bres{self.ID}.pickle"

    def save_counts_windows(self):
        os.makedirs(self.counts_windows_path(directory=True), exist_ok=True)
        with open(self.counts_windows_path(), "wb") as file:
            pickle.dump(self.counts_windows, file)

    def available_IDs(self):
        # TODO redo with glob.glob?
        IDs = []
        for file in os.listdir(self.save_path):
            if file.startswith("lab"):
                IDs.append(int(file[3:-7]))
        return IDs

    def check_if_binning_exist(self):
        pathh = self.counts_windows_path()
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
            "config/s0_0.5_1_MB.yaml",
            "config/sig_frac/0.05.yaml",
            "config/restart/0_10.yaml",
            # "config/binning/CURTAINS.yaml",
            "config/tra_reg/sig_reg.yaml",
        ]
    else:
        config_file_path = sys.argv[1:]
    print("sarting", config_file_path)
    cs = ClusterScanning(config_file_path)
    cs.run()
