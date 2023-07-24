import matplotlib.pyplot as plt
import pickle
from cs_performance_evaluation import cs_performance_evaluation
import numpy as np
import random
from matplotlib.ticker import MaxNLocator
import cluster_scanning

import matplotlib as mpl

mpl.rcParams.update(mpl.rcParamsDefault)  # TODO: DETELE THIS

jj = 1
arr = []
ps = []
c = 0.5
path = "config/s0.05_0.5_1_MB.yml"
color = "green"
cs = cluster_scanning.ClusterScanning(path)
cs.load_mjj()
cs.ID = jj
cs.load_results(jj)
cs.sample_signal_events()
counts_windows = cs.perform_binning()

np.save("test/test_materials/count_windows.npy", counts_windows)
