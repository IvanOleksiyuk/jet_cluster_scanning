import matplotlib.pyplot as plt
from pyjet import cluster,DTYPE_PTEPM
import pandas as pd
import preprocessing
import numpy as np
import pickle
from subjettiness import subjettinesses
import h5py

events_data_path ="../../../hpcwork/rwth0934/LHCO_dataset/processed_tf/lhco_events_v2.h5"
input_f = h5py.File(events_data_path, 'r')
events = input_f['events']

Mjj=[]

obj_num=300
R=1
pixils=80

for i in range(100):
    event = events[i]
    e=np.sum(event[:, :, 0])
    px=np.sum(event[:, :, 1])
    py=np.sum(event[:, :, 2])
    pz=np.sum(event[:, :, 3])
    dijet_mass=e**2-px**2-py**2-pz**2
    dijet_mass=dijet_mass**0.5
    Mjj.append(dijet_mass)

print(Mjj)

mjj_bkg=np.load("../../../hpcwork/rwth0934/LHCO_dataset/processed_mh/anomaly_detection_v2_bkg.npy")
mjj_bkg=mjj_bkg[:, 0]

print(mjj_bkg[:100])