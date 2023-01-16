from cluster_scanning import ClusterScanning
from cs_performance_evaluation import cs_performance_evaluation


def one_full_analysis():
    cs = ClusterScanning(
        [
            "config/s0_0.5_1_MB.yaml",
            "config/sig_frac/0.05.yaml",
            "config/restart/1.yaml",
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
        save_path=cs.save_path,
        ID=1,
    )


if __name__ == "__main__":
    one_full_analysis()
