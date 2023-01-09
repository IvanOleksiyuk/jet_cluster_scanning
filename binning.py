import sys
from cluster_scanning import ClusterScanning


def perform_binning_all(path):
    cs = ClusterScanning(path)
    cs.load_mjj()
    for jj in cs.available_IDs():
        cs.ID = jj
        cs.load_results(jj)
        cs.sample_signal_events()
        cs.perform_binning()
        cs.save_counts_windows()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        config_file_path = (
            "char/0kmeans_scan/k2Trueret0con0W2000_2100ste4_w0.5s1Nboot/"
        )
        +"config.yaml"
    else:
        config_file_path = sys.argv[1]
    perform_binning_all(config_file_path)
