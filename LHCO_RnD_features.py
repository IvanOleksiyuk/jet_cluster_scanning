import matplotlib.pyplot as plt
from pyjet import cluster,DTYPE_PTEPM
import pandas as pd
import preprocessing
import numpy as np
import pickle


features=pd.read_hdf("C:/datasets/events_anomalydetection_v2.features.h5")