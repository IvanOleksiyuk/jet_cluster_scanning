from cluster_scanning import ClusterScanning
from cs_performance_evaluation import cs_performance_evaluation
from binning_utils import default_binning
import pickle


def one_cs_run():
    # Cluster scanning part
    # cs = ClusterScanning(
    #     [
    #         "config/s0_1_0_MB.yaml",
    #         "config/sig_frac/0.05.yaml",
    #         "config/restart/0.yaml",
    #         "config/binning/CURTAINS.yaml",
    #         "config/one_run_experiments.yaml",
    #     ]
    # )
    # cs.run()
    # cs.perform_binning()
    # cs.save_binning_array()
    # cs.save_counts_windows()

    # Evaluation part
    config_path = ["config/cs_eval/maxdev5.yaml", "config/cs_eval/plotting.yaml"]
    path1 = "char/one_run_experiments/k50MB100_3iret0con0.05W2600_2700_w1s0Nrest/binnedW100s16ei30004600/"
    jj = 0
    counts_windows = pickle.load(open(path1 + f"bres{jj}.pickle", "rb"))
    binning = pickle.load(open(path1 + "binning.pickle", "rb"))
    cs_performance_evaluation(
        counts_windows=counts_windows,
        binning=binning,
        path=path1,
        ID=jj,
        config_file_path=config_path,
    )
    cs_performance_evaluation(
        cs.counts_windows,
        filterr="med",
        plotting=True,
        labeling=">5sigma",
        verbous=True,
        binning=default_binning(100, 3000, 4600, 16),
        save_path=cs.save_path,
        ID=0,
    )


if __name__ == "__main__":
    one_cs_run()
