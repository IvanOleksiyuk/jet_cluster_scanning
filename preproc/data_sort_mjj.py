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


# Load data (e.g. images)
input_f = h5py.File(data_file_path, 'r')
images = input_f[data_dataset_name]

# Load mjj
mjj_f = h5py.File(mjj_file_path, 'r')
mjj = mjj_f[mjj_dataset_name]

# load lables
lables_f = h5py.File(lables_file_path, 'r')
labels = lables_f[lables_dataset_name][:]

# Get QCD masses
mjj_bkg = mjj[labels==0]

# Get signal masses
mjj_sig = mjj[labels==1]

# Sort by mjj
ind_bkg=np.argsort(mjj_bkg)
ind_sig=np.argsort(mjj_sig)

np.save(output_mass_bkg, mjj_bkg[ind_bkg])
np.save(output_mass_sig, mjj_sig[ind_sig])

print("#####bkg_data")
images_bkg = images[labels==0]
print("loaded", len(images_bkg))
f = h5py.File(output_images_bkg, 'w')
f.create_dataset('data', (len(images_bkg),2,40,40))
f['data'][:] = images_bkg[ind_bkg]
print("done")
print("--- %s seconds ---" % (time.time() - start_time))
f.close()

print("#####sig_data")
images_sig = images[labels==1]
print("loaded", len(images_sig))
f = h5py.File(output_images_sig, 'w')
f.create_dataset('data', (len(images_sig),2,40,40))
f['data'][:] = images_sig[ind_sig]
print("done")
print("--- %s seconds ---" % (time.time() - start_time))
f.close()
