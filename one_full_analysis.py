from cluster_scanning import ClusterScanning
from binning import perform_binning
from cs_performance_evaluation import perform_performance_evaluation


def one_full_analysis():
    cs = ClusterScanning("config/s0_0.5_1_MB.yaml config/")
    cs.run()
    perform_binning(cs.config_file_path, cs.ID)
    cs.perform_binning()
    perform_performance_evaluation(
        cs.counts_windows,
        save=False,
        filterr="med",
        plotting=True,
        labeling=">5sigma",
        verbous=True,
    )


if __name__ == "__main__":
    one_full_analysis()
