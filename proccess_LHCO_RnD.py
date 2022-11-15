import matplotlib.pyplot as plt
from pyjet import cluster,DTYPE_PTEPM
import pandas as pd
import preprocessing
import numpy as np
import pickle
#from subjettiness import subjettinesses
events_data_path ="/media/ivan/Windows/datasets/events_anomalydetection.h5"
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

needed_bg=2000
needed_sg=0
bg_was=0
sg_was=0
images_bg=[]
images_sg=[]
jet_sjts=[]
jet_mass=[]
obj_num=300
R=1
pixils=40

while needed_bg>len(images_bg) or needed_sg>len(images_sg):
    event = gen.next()
    issignal = event[2100]
    
    if len(images_bg)%100==0 and len(images_bg)!=bg_was:
        print("BG", len(images_bg))
        bg_was=len(images_bg)
        
    if len(images_sg)%100==0 and len(images_sg)!=sg_was:
        print("SG", len(images_sg))
        sg_was=len(images_sg)
    
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
    for i in range(2):
        jet=jets[i]
        #jet_mass.append(jet.mass)
        #sjts=subjettinesses(jet)
        constituents = jet.constituents_array()
        const_array = [constituents["pT"], constituents["eta"], constituents["phi"]]
        const_array = np.array(const_array)
        const_array = const_array.T
        #preprocessing.jet_boost_to_Em((jet.pt, jet.eta, jet.phi, jet.mass), const_array)
        const_array = const_array.reshape((3*len(constituents["pT"])))
        const_array = np.pad(const_array, (0, obj_num*3-3*len(constituents["pT"])))
        
        if issignal:
            images_sg.append(preprocessing.calorimeter_image_ptethaphi(const_array, IMG_SIZE=pixils,
                                                                           phi_bonds=(-R, R),
                                                                           eta_bonds=(-R, R),
                                                                           obj_num=obj_num))
            #plt.figure()
            #plt.imshow(images_sg[-1])
            #plt.title("t21 {:.3f} t32 {:.3f} m {:.1f} l {}".format(sjts[1]/sjts[0], sjts[2]/sjts[1], jet_mass[-1], issignal))
            #plt.show()
            #plt.show()
        else:
            images_bg.append(preprocessing.calorimeter_image_ptethaphi(const_array, IMG_SIZE=pixils,
                                                                       phi_bonds=(-R, R),
                                                                       eta_bonds=(-R, R),
                                                                       obj_num=obj_num))
            #plt.figure()
            #plt.imshow(images_bg[-1])
            #plt.title("t21 {:.3f} t32 {:.3f} m {:.1f} l {}".format(sjts[1]/sjts[0], sjts[2]/sjts[1], jet_mass[-1], issignal))
            #plt.show()
            
images_bg=np.array(images_bg)
images_sg=np.array(images_sg)

pickle_out = open("/home/ivan/datasets/2K_BG40.pickle", "wb")
pickle.dump(images_bg, pickle_out)
pickle_out.close()
        
#pickle_out = open("C://datasets/2K_SGp.pickle", "wb")
#pickle.dump(images_sg, pickle_out)
#pickle_out.close()