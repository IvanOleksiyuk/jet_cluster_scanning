from cs_performance_evaluation import cs_performance_evaluation_single
import sys

def batch_eval_t_stat(experiment_path, binning_folder, config_path, tstat_name, b_start, b_end, s_start, s_end, i_start, i_end):
    for b in range(b_start, b_end):
        for s in range(s_start, s_end):
            for i in range(i_start, i_end):
                bres_id = f"b{b}_s{s}_i{i}"
                print(f"bres_id: {bres_id}")
                cs_performance_evaluation_single(experiment_path, binning_folder, bres_id, config_path, tstat_name)
                
if __name__ == "__main__":
	experiment_path = sys.argv[1]
	binning_folder = sys.argv[2]
	config_path = sys.argv[3]
	tstat_name = sys.argv[4]
 
	b_start = int(sys.argv[5])
	b_end = int(sys.argv[6])
	s_start = int(sys.argv[7])
	s_end = int(sys.argv[8])
	i_start = int(sys.argv[9])
	i_end = int(sys.argv[10])
 
	batch_eval_t_stat(experiment_path, binning_folder, config_path, tstat_name, b_start, b_end, s_start, s_end, i_start, i_end)
"""
Example usage:

python batch_eval_tstat.py "char/one_run_experiments/k50MB2048_1iret0con0.05W3000_3100_w0.5s1Nboot__/" "binnedW100s16ei30004600/" "config/cs_eval/maxdev5.yaml" "maxdev5" 0 1 0 1 0 1

"""