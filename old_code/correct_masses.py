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

data_file_path = cfg.get("data_directory")+"jet_images_v2.h5"#cfg.get("jet_images_file")
data_dataset_name = 'data'
mjj_file_path = cfg.get("data_directory")+cfg.get("clustered_jets_file")
mjj_dataset_name = 'm_jj'
lables_file_path = cfg.get("data_directory")+cfg.get("clustered_jets_file")
lables_dataset_name = 'labels'

output_images_bkg = cfg.get("data_directory") + cfg.get("sorted_bkg_file")
output_images_sig = cfg.get("data_directory") + cfg.get("sorted_sig_file")
output_mass_bkg = cfg.get("data_directory") + cfg.get("sorted_bkg_mass_file")
output_mass_sig = cfg.get("data_directory") + cfg.get("sorted_sig_mass_file")

# Load mjj
mjj_f = h5py.File(mjj_file_path, 'r+')
mjj = mjj_f[mjj_dataset_name][:]
mjj_f[mjj_dataset_name][:] = mjj**0.5
mjj_f.close()