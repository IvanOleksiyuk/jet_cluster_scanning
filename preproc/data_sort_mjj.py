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

# Load data (e.g. images)
input_f = h5py.File(data_file_path, 'r')
images = input_f[data_dataset_name]

# Load mjj
mjj_f = h5py.File(mjj_file_path, 'r')
mjj = mjj_f[mjj_dataset_name]

plt.figure()
plt.imshow(images[0][1])
plt.colorbar()
plt.savefig("plots/data1/data_test1.png")

plt.figure()
plt.imshow(images[0][0])
plt.colorbar()
plt.savefig("plots/data1/data_test2.png")

plt.figure()
plt.imshow(images[1][1])
plt.colorbar()
plt.savefig("plots/data1/data_test3.png")


# mjj_bkg=np.load("../../../hpcwork/rwth0934/LHCO_dataset/processed_mh/anomaly_detection_v2_bkg.npy")
# mjj_bkg=mjj_bkg[:, 0]
# mjj_sig=np.load("../../../hpcwork/rwth0934/LHCO_dataset/processed_mh/anomaly_detection_v2_sig.npy")
# mjj_sig=mjj_sig[:, 0]

# ind_bkg=np.argsort(mjj_bkg)
# ind_sig=np.argsort(mjj_sig)



# print("#####bg_data")
# images_bg = images[0:1000000]
# print("loaded")
# f = h5py.File('../../../hpcwork/rwth0934/LHCO_dataset/processed_io/v2JetImSort_bkg.h5', 'a')
# f.create_dataset('data', (1000000,2,40,40))
# f['data'][:] = images_bg[ind_bkg]
# print("done")
# print("--- %s seconds ---" % (time.time() - start_time))
# f.close()

# print("#####signal_data")
# images_sg = images[1000000:]
# print("loaded")
# f = h5py.File('../../../hpcwork/rwth0934/LHCO_dataset/processed_io/v2JetImSort_sig.h5', 'a')
# f.create_dataset('data', (100000,2,40,40))
# imges_sg = images_sg[ind_sig]
# print("sorted")
# print(type(images_sg))
# f['data'][:] = imges_sg
# print("done")
# print("--- %s seconds ---" % (time.time() - start_time))
# f.close()

# input_f.close()

# np.save("../../../hpcwork/rwth0934/LHCO_dataset/processed_io/mjj_bkg_sort.npy", mjj_bkg[ind_bkg])
# np.save("../../../hpcwork/rwth0934/LHCO_dataset/processed_io/mjj_sig_sort.npy", mjj_sig[ind_sig])
