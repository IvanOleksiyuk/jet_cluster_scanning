import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import utils.set_matplotlib_default
folder="/srv/beegfs/scratch/users/o/oleksiyu/CS/v4/k50MB2048_1iret0con0W3000_3100_w0.5s1Nboot/binnedW100s16ei30004600/"
files_list = os.listdir(folder)
bres_files = [file for file in files_list if file.startswith("bres")]

def load_counts_windows(path):
    res = pickle.load(open(path, "rb"))
    if isinstance(res, list) or isinstance(res, np.ndarray):
        return res
    else:
        return res["counts_windows"]


count_window_list = [load_counts_windows(folder + file)[0] for file in bres_files[:1000]]
min_count = [np.min(count_window) for count_window in count_window_list]
min_count = np.array(min_count)
# print(count_window_list[0])
# print(np.sum(count_window_list[0])/2)
# print(np.min(count_window_list[0]))
# print(min_count)
print(min(min_count))
print(np.sum(min_count<20))
plt.hist(min_count)
plt.axvline(x=np.median(min_count), color="red", label=f"Median: {np.median(min_count)}")
plt.legend()
plt.ylabel("Pseudo-experiments")
plt.xlabel(r"$min_{i,b}(N_{i,b})$")
plt.savefig("plots/misc/min_count.pdf",  interpolation='nearest')
