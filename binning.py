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
    perform_binning_all(
        "char/0kmeans_scan/k2Trueret0con0W2000_2100ste4_w0.5s1Nboot/"
        + "config.yaml"
    )
