import os
import sys
#Make sure the path to the root directory of the project is in the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
import h5py
import time
from utils.config_utils import Config

start_time = time.time()
cfg = Config("config/path.yaml")

data_file_path_old = cfg.get("data_directory") + "jet_images_v2.h5"
data_file_path_new = cfg.get("data_directory") + cfg.get("jet_images_file")
data_dataset_name = 'data'

mjj_file_path = cfg.get("data_directory")+cfg.get("clustered_jets_file")
mjj_dataset_name = 'm_jj'

# Load data (e.g. images)
input_f_old = h5py.File(data_file_path_old, 'r')
images_old = input_f_old[data_dataset_name]

input_f_new = h5py.File(data_file_path_new, 'r')
images_new = input_f_new[data_dataset_name]


print(np.sum(images_new[0][1] == images_old[0][1]))

