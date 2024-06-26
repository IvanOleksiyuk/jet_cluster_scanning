# global imports
import sys
import os
import pickle
import random
import time
import h5py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
import shutil
import logging
import re
import copy
#local imports
import utils.set_matplotlib_default
from utils.config_utils import Config
from preproc import reprocessing

pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.WARNING)


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

        if not "ignore_signal_in_training" in self.cofdict:
            self.ignore_signal_in_training = False
        else:
            self.ignore_signal_in_training = self.cofdict["ignore_signal_in_training"]
        
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
                random.seed(a=i, version=2)
                sum_hash += random.randint(0, 10e6)
            random.seed(sum_hash)
            np.random.seed(sum_hash)

    @staticmethod
    def IDstr(IDb, IDs, IDi):
        return f"_b{IDb}_s{IDs}_i{IDi}"

    @staticmethod
    def IDstr_to_IDs(IDstr):
        integers = re.findall(r"-?\d+", IDstr)
        integers = [int(num) for num in integers]
        return integers

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
        logging.error("NOT IMPLEMENTED YET")  # TODO
        return

    def flatten_event(self, x):
        logging.error("NOT IMPLEMENTED YET")  # TODO
        return x.reshape()

    def flatten_image(self):
        logging.error("NOT IMPLEMENTED YET")  # TODO
        pass

    def de_flatten_image(self):
        logging.error("NOT IMPLEMENTED YET")  # TODO
        pass

    def load_data(self, show_example=True):
        logging.info("loading data")
        im_bg_file = h5py.File(self.cfg.data_path + "v2JetImSort_bkg.h5", "r")
        im_sg_file = h5py.File(self.cfg.data_path + "v2JetImSort_sig.h5", "r")
        self.im_bg = im_bg_file["data"]
        self.im_sg = im_sg_file["data"]
        if self.cfg.memory_intensive:
            logging.info("Using memory intensive mode")
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
            logging.info("bg load and reproc complete")
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
            if np.any(np.isnan(self.im_bg)) or np.any(np.isnan(self.im_sg)):
                logging.warning("There are "+str(np.sum(np.isnan(self.im_bg)))+" nan in bg data")
                logging.warning("There are "+str(np.sum(np.isnan(self.im_sg)))+" nan in sg data")
            logging.debug("Shape of background image array: "+str(self.im_bg.shape))
            logging.debug(f"First background image: {self.im_bg[0]}")
        else:
            logging.info("Using memory saving mode")
        self.bootstrap_bg = None
        logging.info("loading data complete")

    def data_mjj_slise(self, Mjjmin, Mjjmax, ignore_signal=False):
        """Returns the background and signal jets in a given Mjj window

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
                logging.debug("jets in window originally:", len(self.im_bg[indexing_bg[0] : indexing_bg[-1]]))
                logging.debug("jets in window bootstrapped:", np.sum(self.bootstrap_bg[indexing_bg[0] : indexing_bg[-1]]))
            bg = np.repeat(
                self.im_bg[indexing_bg[0] : indexing_bg[-1]],
                self.bootstrap_bg[indexing_bg[0] : indexing_bg[-1]],
                axis=0,
            )

        if not ignore_signal:
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
            if not ignore_signal:
                data = np.concatenate((bg, sg))
            else:
                data = bg
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
            self.cfg.train_interval[0], self.cfg.train_interval[1], ignore_signal=self.ignore_signal_in_training
        )
        self.kmeans.fit(data)
        logging.info("training --- %s seconds ---" % (time.time() - start_time))
        counts = self.get_counts_train()
        counts.sort()
        logging.info(f"sorted cluster counts {counts}")
        logging.info(f"iterations {self.kmeans.n_iter_}")

    def get_counts_train(self):
        counts = np.bincount(self.kmeans.labels_)
        return counts

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
                logging.debug("evaluating chunk background: " +
                    str(i * batch_size) +":"+ str(min((i + 1) * batch_size, len(self.im_bg)))
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
                logging.debug("evaluating chunk signal: " +
                    str(i * batch_size) +":"+ str(min((i + 1) * batch_size, len(self.im_bg)))
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

    def count_bin(
        self,
        mjjmin,
        mjjmax,
        allowed,
        bootstrap_bg,
        idealised=False,
        fractions_idealised=None,
    ):
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
                logging.debug("originaly in the window : " + str(len(self.bg_lab[indexing_bg[0] : indexing_bg[-1]])))
                logging.debug("after bootstrap resample: " + str(np.sum(bootstrap_bg[indexing_bg[0] : indexing_bg[-1]])))
                bg = np.repeat(
                    self.bg_lab[indexing_bg[0] : indexing_bg[-1]],
                    bootstrap_bg[indexing_bg[0] : indexing_bg[-1]],
                    axis=0,
                )
        else:
            bg = np.array([])
            logger.warning("no background events in this window")

        if idealised:
            bg = random.choices(
                range(self.cfg.k),
                weights=fractions_idealised,
                k=len(bg) * 2,
            )
            bg = np.array(bg)
 
        if allowed is not None:
            sg = np.repeat(
                self.sg_lab[indexing_sg[0] : indexing_sg[-1]],
                allowed[indexing_sg[0] : indexing_sg[-1]],
                axis=0,
            )

            return [
                np.array([np.sum(bg == j) for j in range(self.cfg.k)]),
                np.array([np.sum(sg == j) for j in range(self.cfg.k)]),
            ]
        else:
            return [
                np.array([np.sum(bg == j) for j in range(self.cfg.k)]),
                np.zeros(self.cfg.k),
            ]

    def perform_binning(self):
        counts_windows = []
        for i in range(self.cfg.steps):
            if hasattr(self.cfg, "idealised"):
                if self.cfg.idealised and i > 0:
                    if hasattr(counts_windows[0][0], '__iter__'):
                        fractions_idealised = counts_windows[0][0] / np.sum(counts_windows[0][0])
                    else:
                        fractions_idealised = counts_windows[0] / np.sum(counts_windows[0])
                    counts_windows.append(
                        self.count_bin(
                            self.Mjjmin_arr[i],
                            self.Mjjmax_arr[i],
                            self.allowed,
                            self.bootstrap_bg,
                            idealised=True,
                            fractions_idealised=fractions_idealised,
                        )
                    )
                else:
                    counts_windows.append(
                        self.count_bin(
                            self.Mjjmin_arr[i],
                            self.Mjjmax_arr[i],
                            self.allowed,
                            self.bootstrap_bg,
                        )
                    )
            else:
                counts_windows.append(
                    self.count_bin(
                        self.Mjjmin_arr[i],
                        self.Mjjmax_arr[i],
                        self.allowed,
                        self.bootstrap_bg,
                    )
                )

        self.counts_windows_bg = np.stack([x[0] for x in counts_windows])
        self.counts_windows_sg = np.stack([x[1] for x in counts_windows])
        self.counts_windows = [self.counts_windows_bg, self.counts_windows_sg]
        self.counts_windows_sum = sum(self.counts_windows)

        return self.counts_windows

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
        if self.cfg.k < 256:
            dtype = np.uint8
            self.bg_lab = self.bg_lab.astype(dtype)
            self.sg_lab = self.sg_lab.astype(dtype)
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

        if hasattr(self.cfg, "sig_eq_boot_IDs"):
            sig_eq_boot_IDs = self.cfg.sig_eq_boot_IDs
        else:
            sig_eq_boot_IDs = False

        if self.cfg.bootstrap:
            IDb_arr = [i for i in range(self.cfg.IDb_start, self.cfg.IDb_finish)]
        else:
            IDb_arr = [self.def_IDb]

        if self.cfg.resample_signal and (not sig_eq_boot_IDs):
            IDs_arr = [i for i in range(self.cfg.IDs_start, self.cfg.IDs_finish)]
        else:
            IDs_arr = [self.def_IDs]

        if self.cfg.restart:
            IDi_arr = [i for i in range(self.cfg.IDi_start, self.cfg.IDi_finish)]
        else:
            IDi_arr = [self.def_IDi]

        if sig_eq_boot_IDs:
            for IDb in IDb_arr:
                for IDi in IDi_arr:
                    if not os.path.exists(
                        self.save_path + f"lab{self.IDstr(IDb, IDb, IDi)}.pickle"
                    ):
                        ID_tuple_list.append([IDb, IDb, IDi])
        else:
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

    def get_IDstr(self):
        return self.IDstr(self.__bsID, self.__sigID, self.__ID)

    def counts_windows_path(self, directory=False, IDstr=None, fake=False):
        pathh = (
            self.save_path
            + f"binnedW{self.cfg.W}s{self.cfg.steps}ei{self.cfg.eval_interval[0]}{self.cfg.eval_interval[1]}"
        )
        if fake:
            pathh += "fake"
        else:
            if hasattr(self.cfg, "idealised"):
                if self.cfg.idealised:
                    pathh += "ideal"

        pathh += "/"
        if not directory:
            if IDstr is None:
                pathh += f"bres{self.get_IDstr()}.pickle"
            else:
                pathh += f"bres{IDstr}.pickle"
        return pathh

    def save_counts_windows(self):
        os.makedirs(self.counts_windows_path(directory=True), exist_ok=True)
        res = {}
        res["counts_windows"] = self.counts_windows
        res["inertia"] = self.kmeans.inertia_
        with open(self.counts_windows_path(), "wb") as file:
            pickle.dump(res, file)
            logger.info("bin counts saved to " + self.counts_windows_path())

    def load_counts_windows(self, IDstr=None):
        if IDstr is None:
            IDstr = self.get_IDstr()
        with open(self.counts_windows_path(IDstr=IDstr), "rb") as file:
            res = pickle.load(file)
        self.counts_windows = res["counts_windows"]
        if isinstance(self.counts_windows, list):
            self.counts_windows_sum = sum(self.counts_windows)
            self.counts_windows_bg = self.counts_windows[0]
            if len(self.counts_windows) > 1:
                self.counts_windows_sg = self.counts_windows[1]
            else:
                self.counts_windows_sg = np.zeros_like(self.counts_windows_bg)
        self.kmeans.inertia_ = res["inertia"]

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

    def plot_clusters(self):
        plots_path = self.save_path + f"plots{self.get_IDstr()}/"
        os.makedirs(plots_path, exist_ok=True)
        plt.figure()
        plt.grid()
        for j in range(self.cfg.k):
            plt.plot(self.mjj_bg, self.bg_lab[:, j], ".", alpha=0.1)
        plt.xlabel("$m_{jj}$")
        plt.ylabel("cluster label")
        plt.savefig(plots_path + "kmeans_ni_mjj_total.pdf")

    def make_plots(self):
        # Some plotting

        plots_path = self.save_path + f"plots{self.get_IDstr()}/"
        os.makedirs(plots_path, exist_ok=True)

        self.plot_cluster_spectra(plots_path, "kmeans_ni_mjj_total.pdf")
        self.plot_cluster_spectra(
            plots_path, "kmeans_ni_mjj_total_statAllowed.pdf", plot_stat_allowed=True
        )
        self.plot_cluster_spectra(
            plots_path, "kmeans_ni_mjj_maxnuorm.pdf", normalize="max"
        )
        self.plot_cluster_spectra(
            plots_path, "kmeans_ni_mjj_sumnorm.pdf", normalize="sum"
        )
        self.plot_cluster_spectra(
            plots_path, "kmeans_ni_mjj_perbin.pdf", normalize="per_bin"
        )
        self.plot_global_stats(plots_path, "total_stats.pdf")
        self.plot_global_stats(plots_path, "total_stats_sigsort.pdf", sort="sig")
        self.plot_global_stats(plots_path, "total_stats_bgsort.pdf", sort="bg")
        self.plot_global_stats(plots_path, "total_stats_totsort.pdf", sort="tot")
        if hasattr(self, "counts_windows_sg"):
            self.plot_cluster_images(plots_path)
            self.plot_cluster_images(plots_path, sort="bg", display_info="bg")
            self.plot_cluster_images(plots_path, sort="tot", display_info="tot")
            self.plot_cluster_images(plots_path, sort="sig", display_info="sig")
            self.plot_cluster_images(plots_path, sort="SFI", display_info="SFI_SI")
            self.plot_cluster_images(plots_path, sort="SI", display_info="SFI_SI")

    def plot_global_stats(self, plots_path, plot_name, sort=None):
        if hasattr(self, "counts_windows_sg"):
            spectra = self.counts_windows_sum
            tot_counts = np.sum(spectra, axis=0)
            bg_counts = np.sum(self.counts_windows_bg, axis=0)
            sg_counts = np.sum(self.counts_windows_sg, axis=0)

            if sort is not None:
                if sort == "sig":
                    index = np.argsort(sg_counts)
                elif sort == "bg":
                    index = np.argsort(bg_counts)
                elif sort == "tot":
                    index = np.argsort(tot_counts)
                tot_counts = tot_counts[index]
                bg_counts = bg_counts[index]
                sg_counts = sg_counts[index]
            plt.figure()
            plt.grid()
            plt.plot(tot_counts, label="total")
            plt.plot(bg_counts, label="background")
            plt.plot(sg_counts, label="signal")
            plt.legend()
            plt.savefig(plots_path + plot_name)
        else:
            print("skip global stats as no signal is present")

    def plot_cluster_images(self, plots_path, sort=None, display_info=None):
        spectra = self.counts_windows_sum
        tot_counts = np.sum(spectra, axis=0)
        bg_counts = np.sum(self.counts_windows_bg, axis=0)
        if hasattr(self, "counts_windows_sg"):
            sg_counts = np.sum(self.counts_windows_sg, axis=0)
            sfi = sg_counts/np.sum(sg_counts) / (bg_counts/np.sum(bg_counts))
            si = sg_counts/np.sum(sg_counts) / np.sqrt((bg_counts)/np.sum(bg_counts)) 
        if sort is not None:
            if sort == "sig":
                index = np.argsort(sg_counts)[::-1]
            elif sort == "bg":
                index = np.argsort(bg_counts)[::-1]
            elif sort == "tot":
                index = np.argsort(tot_counts)[::-1]
            elif sort == "SFI":
                index = np.argsort(sfi)[::-1]
            elif sort == "SI":
                index = np.argsort(si)[::-1]
            tot_counts = tot_counts[index]
            bg_counts = bg_counts[index]
            sg_counts = sg_counts[index]
            images = self.kmeans.cluster_centers_[index]
            sfi = sfi[index]
            si = si[index]
        else:
            images = self.kmeans.cluster_centers_
            index = np.arange(self.cfg.k)

        plt.figure(figsize=(20, 10))
        plt.grid()
        for j in range(self.cfg.k):
            plt.subplot(5, 10, j + 1)
            plt.tick_params(
                left=False,
                right=False,
                labelleft=False,
                labelbottom=False,
                bottom=False,
            )
            plt.imshow(
                images[j].reshape(
                    (self.cfg.image_size, self.cfg.image_size)
                ),
                cmap="turbo",
            )
            if display_info=="SI":
                plt.title(f"SI {si[j]:.2e}")
            elif display_info=="SFI":
                plt.title(f"SFI {sfi[j]:.2e}")
            elif display_info=="SFI_SI":
                plt.title(f"SFI {sfi[j]:.1e} SI {si[j]:.1e}")
            elif display_info=="sig":
                plt.title(f"signal {sg_counts[j]:.2e}")
            elif display_info=="bg":
                plt.title(f"background {bg_counts[j]:.2e}")
            elif display_info=="tot":
                plt.title(f"total {tot_counts[j]:.2e}")
            else:
                plt.title(f"cluster {index[j]}")
        plt.savefig(plots_path + "kmeans_images"+str(sort)+str(display_info)+".pdf", bbox_inches="tight")

    def plot_cluster_spectra(
        self,
        plots_path,
        plot_name,
        plot_stat_allowed=False,
        min_allowed_count=100,
        min_min_allowed_count=10,
        normalize=None,
        stairs=False,
    ):
        plt.figure()
        window_centers = (self.Mjjmin_arr + self.Mjjmax_arr) / 2

        if normalize is None:
            spectra = self.counts_windows_sum
            plt.ylabel("n points from window")
        elif normalize == "max":
            spectra = np.zeros(self.counts_windows_sum.shape)
            for i in range(self.cfg.k):
                spectra[:, i] = self.counts_windows_sum[:, i] / np.max(
                    self.counts_windows_sum[:, i]
                )
            plt.ylabel("n points from window/max(...)")
        elif normalize == "sum":
            spectra = np.zeros(self.counts_windows_sum.shape)
            for i in range(self.cfg.k):
                spectra[:, i] = self.counts_windows_sum[:, i] / np.sum(
                    self.counts_windows_sum[:, i]
                )
            plt.ylabel("n points from window/sum(...)")
        elif normalize == "per_bin":
            spectra = (
                self.counts_windows_sum
                / self.counts_windows_sum.sum(axis=1)[:, np.newaxis]
            )
            plt.ylabel("ratio points from window")

        plt.grid()
        for j in range(self.cfg.k):
            if stairs:
                plt.step(window_centers, spectra[:, j], where="mid")
            else:
                plt.plot(window_centers, spectra[:, j])
        plt.xlabel("$m_{jj}$")
        if plot_stat_allowed:
            smallest_cluster_count_window = np.min(self.counts_windows_sum, axis=1)
            for i in range(len(window_centers)):
                if smallest_cluster_count_window[i] < min_allowed_count:
                    if smallest_cluster_count_window[i] < min_min_allowed_count:
                        plt.axvline(window_centers[i], color="black", alpha=0.6)
                    else:
                        plt.axvline(window_centers[i], color="black", alpha=0.3)
        plt.savefig(plots_path + plot_name, bbox_inches="tight")

    def bin_mjj_inc(self):
        return self.counts_windows_sum.sum(axis=1)

    def generate_fake_pseudoexperiments(self, err_dist="correct", err_par=1, n=1000):
        counts_windows_inc = self.bin_mjj_inc()
        fr = self.get_counts_train()
        fr = fr / np.sum(fr)
        fake_clusters = counts_windows_inc * fr.reshape((-1, 1))
        err = np.sqrt(fake_clusters)
        # now we need to distort the clusters using the error
        for i in range(n):
            if err_dist == "normal":
                deviation = np.random.normal(err_par[0], err * err_par[1])
            if err_dist == "student-t":
                deviation = (
                    (np.random.standard_t(err_par[0], size=err.shape) + err_par[1])
                    * err
                    * err_par[2]
                )
            if err_dist == "multinomial":
                # Split each window randomly but correctly into clusters
                samples = []
                for n in counts_windows_inc:
                    samples.append(np.random.multinomial(n, fr))

                fake_clusters_sampled = np.stack(samples).T
                # Take deviation as difference between expected and actual counts
                deviation = fake_clusters_sampled - fake_clusters
                deviation *= err_par
            # deviation = deviation - np.mean(deviation, axis=0)
            fake_counts_distorted = fake_clusters + deviation
            # Now we just have to store them in the same format as the real ones
            os.makedirs(
                self.counts_windows_path(directory=True, fake=True), exist_ok=True
            )
            res = {}
            res["counts_windows"] = fake_counts_distorted.T
            res["inertia"] = self.kmeans.inertia_
            with open(
                self.counts_windows_path(
                    fake=True, IDstr=self.IDstr(i, self.__sigID, self.__ID)
                ),
                "wb",
            ) as file:
                pickle.dump(res, file)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        config_file_path = [
            "config/v4/s0_0.5_1_MB_i1.yaml",
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
