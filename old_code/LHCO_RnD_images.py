import matplotlib.pyplot as plt
from pyjet import cluster,DTYPE_PTEPM
import preprocessing
import numpy as np
import pickle
from subjettiness import subjettinesses
import h5py
import time
start_time = time.time()


events_data_path ="../../../hpcwork/rwth0934/LHCO_dataset/processed_tf/lhco_events_v2.h5"
input_f = h5py.File(events_data_path, 'r')
events = input_f['events']

obj_num=300
R=1

f = h5py.File('../../../hpcwork/rwth0934/LHCO_dataset/processed_io/jet_images_v2.h5', 'a')
CS=1000 #chunk size
NC=1100
f.create_dataset('data', (NC*CS,2,40,40))

for i in range(NC):
    events_chunk = events[CS*i:CS*(i+1)]
    images=np.zeros((CS, 2, 40, 40))
    for j in range(CS):
        for k in range(2):
            images[j, k]=preprocessing.calorimeter_image(events_chunk[j, k].flatten())
    
    f['data'][i*CS:(i+1)*CS] = images
    print("chunk {:} done:".format(i))
    print("--- %s seconds ---" % (time.time() - start_time))
  
f.close()
input_f.close()


