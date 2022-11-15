import matplotlib.pyplot as plt
from pyjet import cluster,DTYPE_PTEPM
import preprocessing
import numpy as np
import pickle
from subjettiness import subjettinesses
import h5py
import time
start_time = time.time()

events_data_path ="../../../hpcwork/rwth0934/LHCO_dataset/processed_io/jet_images_v2_gzip.h5"
# Option 3: Use generator to loop over the whole file
input_f = h5py.File(events_data_path, 'r')
images = input_f['data']

f = h5py.File('../../../hpcwork/rwth0934/LHCO_dataset/processed_io/jet_images_v2.h5', 'a')
CS=1000 #chunk size

f.create_dataset('data', (1100000,2,40,40))

for i in range(1100):
    f['data'][-CS:] = images[CS*i:CS*(i+1)]

  
f.close()
input_f.close()


