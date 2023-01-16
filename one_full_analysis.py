from cluster_scanning import ClusterScanning
from cs_performance_evaluation import cs_performance_evaluation
from binning_utils import default_binning


def one_full_analysis():
    cs = ClusterScanning(
        [
            "config/s0_0.5_1_MB.yaml",
            "config/sig_frac/0.05.yaml",
            "config/restart/0.yaml",
            "config/binning/CURTAINS.yaml",
            "config/v3.yaml",
        ]
    )
    cs.run()
    cs.perform_binning()
    cs_performance_evaluation(
        cs.counts_windows,
        save=False,
        filterr="med",
        plotting=True,
        labeling=">5sigma",
        verbous=True,
        binning=default_binning(100, 3000, 4600, 16),
        save_path=cs.save_path,
        ID=0,
    )


if __name__ == "__main__":
    one_full_analysis()
