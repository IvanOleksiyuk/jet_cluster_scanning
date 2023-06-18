import os
import sys
from cluster_scanning import ClusterScanning
import logging

logging.basicConfig(level=logging.INFO)


def perform_binning_ID(config, ID, override_config=None):
    """Performs binning for the given ID in the cluster_scanning object and saves count_windows

    Parameters
    ----------
    config : path to yaml file containing the configuration parameters (path to data, binning parameters etc.)
    ID : ID to perform the binning for
    override_config : path to yaml file containing the configuration parameters (path to data, binning parameters etc.)
    HINT: if you want to override some of the parameters in the config file, you can pass a list of config files each overriding the previous one.
    """
    cs = ClusterScanning(config)
    cs.load_mjj()
    cs.ID = ID
    cs.load_results(ID)
    cs.sample_signal_events()
    cs.bootstrap_resample()
    cs.perform_binning()
    cs.save_counts_windows()


def perform_binning_all(config):
    """Performs binning for all available IDs in the cluster_scanning object and saves count_windows

    Parameters
    ----------
    config : path to yaml file containing the configuration parameters (path to data, binning parameters etc.)
    HINT: if you want to override some of the parameters in the config file, you can pass a list of config files each overriding the previous one.
    """
    cs = ClusterScanning(config)
    cs.load_mjj()

    for i, IDstr in enumerate(cs.available_IDstr()):
        if i == 0:
            cs.save_binning_array()
        if cs.check_if_binning_exist(IDstr=IDstr):
            # print("checked_exist: ", cs.counts_windows_path())
            continue
        cs.load_results(IDstr=IDstr)
        cs.sample_signal_events()
        cs.bootstrap_resample()
        cs.perform_binning()
        cs.save_counts_windows()
        # print("done: ", cs.counts_windows_path())
    # TODO: make into verobose mode


def perform_binning_directory(directory, override_config=None):
    """For all subfolders in the directory Performs binning for all available IDs in the cluster_scanning object and saves count_windows

    Parameters
    ----------
    override_config : path to yaml file containing the configuration parameters (path to data, binning parameters etc.)
    HINT: if you want to override some of the parameters in the config file, you can pass a list of config files each overriding the previous one.
    """
    for subfolder in os.listdir(directory):
        logging.info(f"doing binning in {subfolder}")
        if os.path.isdir(os.path.join(directory, subfolder)):
            if os.path.isfile(os.path.join(directory, subfolder, "confsum.yaml")):
                if override_config is None:
                    perform_binning_all(
                        os.path.join(directory, subfolder, "confsum.yaml")
                    )
                else:
                    perform_binning_all(
                        [
                            os.path.join(directory, subfolder, "confsum.yaml"),
                        ]
                        + override_config
                    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        perform_binning_directory("char/v4/")
        perform_binning_directory("char/v4/", ["config/binning/CURTAINS.yaml"])

    else:
        if sys.argv[1] == "-d":
            if len(sys.argv) >= 4:
                perform_binning_directory(sys.argv[2], sys.argv[3:])
            else:
                perform_binning_directory(sys.argv[2])
        else:
            perform_binning_all(sys.argv[1:])
