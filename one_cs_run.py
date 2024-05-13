from cluster_scanning import ClusterScanning
from cs_performance_evaluation import cs_performance_evaluation
from utils.binning_utils import default_binning
import pickle
import time
import sys

def one_cs_run(config_cs, config_ev, skip_clustering=False):
    # Cluster scanning part
    cs = ClusterScanning(config_cs)

    if not skip_clustering:
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

    # cs.generate_fake_pseudoexperiments(
    #     err_dist="multinomial",
    #     err_par=1.7,
    #     n=40000,
    # )

    # # Evaluation part
    path1 = cs.counts_windows_path(directory=True)
    counts_windows = pickle.load(open(path1 + f"bres{cs.get_IDstr()}.pickle", "rb"))[
        "counts_windows"
    ]
    binning = pickle.load(open(path1 + "binning.pickle", "rb"))
    res = cs_performance_evaluation(
        counts_windows=counts_windows,
        binning=binning,
        path=path1,
        ID=cs.get_IDstr(),
        config_file_path=config_ev,
    )
    print("result:", res)
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


def named_settings(setting):
    if setting == "default":
        ###""""""""""""""""""""""""""""""""""""""""""""
        ### Default setup DO NOT CHANGE, needed for pub_plot
        config_cs = [
                "config/path.yaml",
                "config/v4/s0_0.5_1_MB_i1.yaml",
                "config/sig_frac/0.05.yaml",
                "config/multirun/0_0_0.yaml",
                "config/binning/CURTAINS.yaml",
                "config/tra_reg/3000_3100.yaml",
                "config/one_run_experiments.yaml"
            ]
        config_ev = [
            "config/cs_eval/maxdev3_msders.yaml",
            "config/cs_eval/plotting.yaml"]
    elif setting == "default+fit":
        # ###""""""""""""""""""""""""""""""""""""""""""""
        # ### No signal inclusion in training region despite using in evaluation
        config_cs = [
                "config/path.yaml",
                "config/v4/s0_0.5_1_MB_i1.yaml",
                "config/sig_frac/0.05.yaml",
                "config/multirun/0_0_0.yaml",
                "config/binning/CURTAINS.yaml",
                "config/tra_reg/3000_3100.yaml",
                "config/one_run_experiments.yaml"]
        config_ev = [
        "config/cs_eval/maxdev3_msders_3fit.yaml",
        "config/cs_eval/plotting.yaml"]
    elif setting == "default_ignore_sig":
        # ###""""""""""""""""""""""""""""""""""""""""""""
        # ### No signal inclusion in training region despite using in evaluation
        config_cs = [
                "config/path.yaml",
                "config/v4/s0_0.5_1_MB_i1.yaml",
                "config/sig_frac/0.05.yaml",
                "config/multirun/0_0_0.yaml",
                "config/binning/CURTAINS.yaml",
                "config/tra_reg/3000_3100.yaml",
                "config/one_run_experiments.yaml",
                "config/ignore_signal.yaml"]
        config_ev = [
        "config/cs_eval/maxdev3_msders.yaml",
        "config/cs_eval/plotting.yaml"]
    elif setting == "sig_reg":
        # ###""""""""""""""""""""""""""""""""""""""""""""
        # ### Training in the signal rich region
        config_cs = [
                "config/path.yaml",
                "config/v4/s0_0.5_1_MB_i1.yaml",
                "config/sig_frac/0.05.yaml",
                "config/multirun/0_0_0.yaml",
                "config/binning/CURTAINS.yaml",
                "config/tra_reg/sig_reg.yaml",
                "config/one_run_experiments.yaml"]
        config_ev = [
        "config/cs_eval/maxdev3_msders.yaml",
        "config/cs_eval/plotting.yaml"]
    elif setting == "default+fit+nosig":
        # ###""""""""""""""""""""""""""""""""""""""""""""
        # ### No signal inclusion in training region despite using in evaluation
        config_cs = [
                "config/path.yaml",
                "config/v4/s0_0.5_1_MB_i1.yaml",
                #"config/sig_frac/0.05.yaml",
                "config/multirun/0_0_0.yaml",
                "config/binning/CURTAINS.yaml",
                "config/tra_reg/3000_3100.yaml",
                "config/one_run_experiments.yaml"]
        config_ev = [
        "config/cs_eval/maxdev3_msders_3fit.yaml",
        "config/cs_eval/plotting.yaml"]
    return config_cs, config_ev

def main():
    if len(sys.argv) < 2:
        setting = "default+fit+nosig"
        #print("Usage: one_cs_run.py <setting>")
        #sys.exit(1)
    else:
        setting = sys.argv[1]
    config_cs, config_ev = named_settings(setting)
    one_cs_run(config_cs, config_ev)

if __name__ == "__main__":
    main()
