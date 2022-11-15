import matplotlib.pyplot as plt
from pyjet import cluster,DTYPE_PTEPM
import pandas as pd
import preprocessing
import numpy as np
import pickle
events_data_path ="C://datasets/events_anomalydetection.h5"
# Option 3: Use generator to loop over the whole file
class generator:
    def __init__(self, filename, chunksize=512, total_size=1100000):
        self.i=0
        self.j=0
        self.chunksize=chunksize
        self.filename=filename
        self.fnew = pd.read_hdf(self.filename,start=self.i*self.chunksize, stop=(self.i+1)*self.chunksize)
        self.fnew = self.fnew.T
        
    def next(self):
        if self.j>=self.chunksize*(self.i+1):
            self.i+=1
            self.fnew = pd.read_hdf(self.filename,start=self.i*self.chunksize, stop=(self.i+1)*self.chunksize)
            self.fnew = self.fnew.T
            self.j+=1
            return self.fnew[self.j-1]
        else:
            self.j+=1
            return self.fnew[self.j-1]

gen = generator(events_data_path)

needed_bg=100000
needed_sg=2000
images_bg=[]
images_sg=[]
images_bg_2l=[]
images_sg_2l=[]

obj_num=300

while needed_bg>len(images_bg) or needed_sg>len(images_sg):
    event = gen.next()
    issignal = event[2100]
    
    if len(images_sg)%100==0:
        print(len(images_sg))
    
    if (needed_bg<=len(images_bg) and issignal==0):
        continue
    elif (needed_sg<=len(images_sg) and issignal==1):
         continue
    pseudojets_input = np.zeros(len([x for x in event[::3] if x > 0]), dtype=DTYPE_PTEPM)
    for j in range(700):
        if (event[j*3]>0):
            pseudojets_input[j]['pT'] = event[j*3]
            pseudojets_input[j]['eta'] = event[j*3+1]
            pseudojets_input[j]['phi'] = event[j*3+2]
            pass
        pass
    sequence = cluster(pseudojets_input, R=1.0, p=-1)
    jets = sequence.inclusive_jets(ptmin=20)
    
    #first leading jet
    i=0
    constituents = jets[i].constituents_array()
    const_array = [constituents["pT"], constituents["eta"], constituents["phi"]]
    const_array = np.array(const_array)
    const_array = const_array.T
    const_array = const_array.reshape((3*len(constituents["pT"])))
    const_array = np.pad(const_array, (0, obj_num*3-3*len(constituents["pT"])))
    
    if issignal:
        images_sg.append(preprocessing.calorimeter_image_ptethaphi(const_array, IMG_SIZE=40,
                                                                       phi_bonds=(-1, 1),
                                                                       eta_bonds=(-1, 1),
                                                                       obj_num=obj_num))
    else:
        images_bg.append(preprocessing.calorimeter_image_ptethaphi(const_array, IMG_SIZE=40,
                                                                   phi_bonds=(-1, 1),
                                                                   eta_bonds=(-1, 1),
                                                                   obj_num=obj_num))
    #second leading jet
    i=1
    constituents = jets[i].constituents_array()
    const_array = [constituents["pT"], constituents["eta"], constituents["phi"]]
    const_array = np.array(const_array)
    const_array = const_array.T
    const_array = const_array.reshape((3*len(constituents["pT"])))
    const_array = np.pad(const_array, (0, obj_num*3-3*len(constituents["pT"])))
    
    if issignal:
        images_sg_2l.append(preprocessing.calorimeter_image_ptethaphi(const_array, IMG_SIZE=40,
                                                                       phi_bonds=(-1, 1),
                                                                       eta_bonds=(-1, 1),
                                                                       obj_num=obj_num))
    else:
        images_bg_2l.append(preprocessing.calorimeter_image_ptethaphi(const_array, IMG_SIZE=40,
                                                                   phi_bonds=(-1, 1),
                                                                   eta_bonds=(-1, 1),
                                                                   obj_num=obj_num))    
   
images_bg=np.array(images_bg)
images_sg=np.array(images_sg)
images_bg_2l=np.array(images_bg_2l)
images_sg_2l=np.array(images_sg_2l)

pickle_out = open("C://datasets/=100K_BG1l.pickle", "wb")
pickle.dump(images_bg, pickle_out)
pickle_out.close()

pickle_out = open("C://datasets/2K_SG1l.pickle", "wb")
pickle.dump(images_sg, pickle_out)
pickle_out.close()

pickle_out = open("C://datasets/=100K_BG2l.pickle", "wb")
pickle.dump(images_bg_2l, pickle_out)
pickle_out.close()
        
pickle_out = open("C://datasets/2K_SG2l.pickle", "wb")
pickle.dump(images_sg_2l, pickle_out)
pickle_out.close()