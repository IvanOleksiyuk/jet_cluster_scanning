from cluster_scanning import ClusterScanning
from cs_performance_evaluation import cs_performance_evaluation
from utils.binning_utils import default_binning
import pickle
import time


def one_cs_run():
    # Cluster scanning part
    cs = ClusterScanning(
        [
            "config/v4/s0_0.5_1_MB_i1.yaml",
            "config/sig_frac/0.05.yaml",
            "config/multirun/0_0_0.yaml",
            # "config/multirun/i2.yaml",
            "config/binning/CURTAINS.yaml",
            "config/tra_reg/3000_3100.yaml",
            "config/one_run_experiments.yaml",
            # "config/idealised.yaml",
        ]
    )

    start_time = time.time()
    cs.run()
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time:", execution_time, "seconds")
    cs.load_results()
    cs.load_mjj()
    cs.sample_signal_events()
    cs.bootstrap_resample()

    start_time = time.time()
    cs.perform_binning()
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time:", execution_time, "seconds")

    cs.save_binning_array()
    cs.save_counts_windows()
    cs.load_counts_windows()
    cs.make_plots()

    cs.generate_fake_pseudoexperiments(
        err_dist="multinomial",
        err_par=1.7,
        n=40000,
    )

    # # Evaluation part
    config_path = [
        "config/cs_eval/maxdev3_msde_4par.yaml",
        "config/cs_eval/plotting.yaml",
    ]
    path1 = cs.counts_windows_path(directory=True)
    counts_windows = pickle.load(open(path1 + f"bres{cs.get_IDstr()}.pickle", "rb"))[
        "counts_windows"
    ]
    binning = pickle.load(open(path1 + "binning.pickle", "rb"))
    cs_performance_evaluation(
        counts_windows=counts_windows,
        binning=binning,
        path=path1,
        ID=cs.get_IDstr(),
        config_file_path=config_path,
    )
    # cs_performance_evaluation(
    #     cs.counts_windows,
    #     filterr="med",
    #     plotting=True,
    #     labeling=">5sigma",
    #     verbous=True,
    #     binning=default_binning(100, 3000, 4600, 16),
    #     save_path=cs.save_path,
    #     ID=0,
    # )


if __name__ == "__main__":
    one_cs_run()
