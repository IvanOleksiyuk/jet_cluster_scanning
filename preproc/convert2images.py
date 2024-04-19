import os
import sys
#Make sure the path to the root directory of the project is in the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import preprocessing
import numpy as np
import h5py
import time
from utils.config_utils import Config

cfg = Config("config/path.yaml")

start_time = time.time()

infilepath = cfg.get("data_directory")+cfg.get("clustered_jets_file")
outfilepath = cfg.get("data_directory")+cfg.get("jet_images_file")
input_f = h5py.File(infilepath, 'r')
events = input_f['events']

f = h5py.File(outfilepath, 'w')
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


