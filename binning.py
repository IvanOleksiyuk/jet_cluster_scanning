import os 
import sys
from cluster_scanning import ClusterScanning

def perform_binning_all(config):
    """ Performs binning for all available IDs in the cluster_scanning object and saves count_windows 
    
    Parameters
    ----------
    config : path to yaml file containing the configuration parameters (path to data, binning parameters etc.)
    HINT: if you want to override some of the parameters in the config file, you can pass a list of config files each overriding the previous one.
    """
    cs = ClusterScanning(config)
    cs.load_mjj()
    for jj in cs.available_IDs():
        cs.ID = jj
        cs.load_results(jj)
        cs.sample_signal_events()
        cs.perform_binning()
        cs.save_counts_windows()

def perform_binning_directory(dir, override_config=None):
    """ For all subfolders in the directory Performs binning for all available IDs in the cluster_scanning object and saves count_windows 
    
    Parameters
    ----------
    override_config : path to yaml file containing the configuration parameters (path to data, binning parameters etc.)
    HINT: if you want to override some of the parameters in the config file, you can pass a list of config files each overriding the previous one.
    """
    for subfolder in os.listdir(dir):
        if os.path.isdir(os.path.join(dir, subfolder)):
            if override_config is None:
                perform_binning_all(os.path.join(dir, subfolder, "config.yaml"))
            else:
                perform_binning_all([os.path.join(dir, subfolder, "config.yaml"), override_config])

def export_binnings(path):
    """ Exports the count_windows (bin data) from each analyssis folder and saves them in a duplicate folder for easyer extraction from cluster
    (count_winows usually take much less space than the full array of labels)
    """
    for subfolder in os.listdir(path):
        if os.path.isdir(os.path.join(path, subfolder)):
            

if __name__ == "__main__":
    if len(sys.argv) == 1:
        config_file_path = (
            "char/0kmeans_scan/k2Trueret0con0W2000_2100ste4_w0.5s1Nboot/"
        )
        +"config.yaml"
    else:
        config_file_path = sys.argv[1]
    perform_binning_all(config_file_path)

