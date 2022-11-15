import matplotlib.pyplot as plt
from pyjet import cluster,DTYPE_PTEPM
import preprocessing
import numpy as np
import pickle
from subjettiness import subjettinesses
import h5py
import time
start_time = time.time()

CS=1000 #chunk size

events_data_path ="../../../hpcwork/rwth0934/LHCO_dataset/processed_io/jet_images_v2.h5"
# Option 3: Use generator to loop over the whole file
input_f = h5py.File(events_data_path, 'r')
images = input_f['data']

plt.figure()
plt.imshow(images[0][1])
plt.colorbar()
plt.savefig("plots/test/data_test1.png")

plt.figure()
plt.imshow(images[0][0])
plt.colorbar()
plt.savefig("plots/test/data_test2.png")

plt.figure()
plt.imshow(images[1][1])
plt.colorbar()
plt.savefig("plots/test/data_test3.png")


mjj_bkg=np.load("../../../hpcwork/rwth0934/LHCO_dataset/processed_mh/anomaly_detection_v2_bkg.npy")
mjj_bkg=mjj_bkg[:, 0]
mjj_sig=np.load("../../../hpcwork/rwth0934/LHCO_dataset/processed_mh/anomaly_detection_v2_sig.npy")
mjj_sig=mjj_sig[:, 0]

ind_bkg=np.argsort(mjj_bkg)
ind_sig=np.argsort(mjj_sig)



print("#####bg_data")
images_bg = images[0:1000000]
print("loaded")
f = h5py.File('../../../hpcwork/rwth0934/LHCO_dataset/processed_io/v2JetImSort_bkg.h5', 'a')
f.create_dataset('data', (1000000,2,40,40))
f['data'][:] = images_bg[ind_bkg]
print("done")
print("--- %s seconds ---" % (time.time() - start_time))
f.close()

print("#####signal_data")
images_sg = images[1000000:]
print("loaded")
f = h5py.File('../../../hpcwork/rwth0934/LHCO_dataset/processed_io/v2JetImSort_sig.h5', 'a')
f.create_dataset('data', (100000,2,40,40))
imges_sg = images_sg[ind_sig]
print("sorted")
print(type(images_sg))
f['data'][:] = imges_sg
print("done")
print("--- %s seconds ---" % (time.time() - start_time))
f.close()

input_f.close()

np.save("../../../hpcwork/rwth0934/LHCO_dataset/processed_io/mjj_bkg_sort.npy", mjj_bkg[ind_bkg])
np.save("../../../hpcwork/rwth0934/LHCO_dataset/processed_io/mjj_sig_sort.npy", mjj_sig[ind_sig])
