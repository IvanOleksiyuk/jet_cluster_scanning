import matplotlib.pyplot as plt
from pyjet import cluster,DTYPE_PTEPM
import pandas as pd
import preprocessing
import numpy as np
import pickle
from subjettiness import subjettinesses
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
bg_was=0
sg_was=0
feat={}
feat["JJmass"]=[]
feat["mass"]=[]
feat["tau1"]=[]
feat["tau2"]=[]
feat["tau3"]=[]
feat["label"]=[]


obj_num=300
R=1
pixils=80

while needed_bg>len(feat["label"])-np.sum(feat["label"]) or needed_sg>np.sum(feat["label"]):
    event = gen.next()
    issignal = event[2100]
    
    num_sg=np.sum(feat["label"])
    num_bg=len(feat["label"])-np.sum(feat["label"])
    
    if num_bg%100==0 and num_bg!=bg_was:
        print("BG", num_bg)
        bg_was=num_bg
        
    if num_sg%100==0 and num_sg!=sg_was:
        print("SG", num_sg)
        sg_was=num_sg
    
    if (needed_bg<=num_bg and issignal==0):
        continue
    elif (needed_sg<=num_sg and issignal==1):
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
    dijet_mass=(jets[0].e+jets[1].e)**2-(jets[0].px+jets[1].px)**2-(jets[0].py+jets[1].py)**2-(jets[0].pz+jets[1].pz)**2
    dijet_mass=dijet_mass**0.5
    for i in range(2):
        jet=jets[i]
        sjts=subjettinesses(jet)
        feat["JJmass"].append(dijet_mass)
        feat["mass"].append(jet.mass)
        feat["tau1"].append(sjts[0])
        feat["tau2"].append(sjts[1])
        feat["tau3"].append(sjts[2])
        feat["label"].append(issignal)
        
feat["JJmass"]=np.array(feat["JJmass"])
feat["mass"]=np.array(feat["mass"])
feat["tau1"]=np.array(feat["tau1"])
feat["tau2"]=np.array(feat["tau2"])
feat["tau3"]=np.array(feat["tau3"])
feat["label"]=np.array(feat["label"])



#%%
pickle_out = open("C://datasets/100Kbg+@Ksg_feastures.pickle", "wb")
pickle.dump(feat, pickle_out)
pickle_out.close()

#%%
with open("C://datasets/100Kbg+@Ksg_feastures.pickle", "rb") as f:
    feat=pickle.load(f)

#%%
feat_sg={}
for key in feat.keys():
    feat_sg[key]=feat[key][feat["label"]==1]
    
    
feat_bg={}
for key in feat.keys():
    feat_bg[key]=feat[key][feat["label"]==0]
    
#%%
feat["primary"]=[]
for i in range(len(feat["label"])//2):
    j1=feat["mass"][i*2]
    j2=feat["mass"][i*2+1]
    if j1<j2:
        feat["primary"].append(0)
        feat["primary"].append(1)
    else:
        feat["primary"].append(1)
        feat["primary"].append(0)    
        
feat["primary"]=np.array(feat["primary"])
        
#%%

plt.hist(feat_sg["mass"][feat_sg["primary"]==1], bins=20, density=True, histtype='step')
plt.hist(feat_sg["mass"][feat_sg["primary"]==0], bins=20, density=True, histtype='step')

plt.hist(feat_bg["mass"][feat_bg["primary"]==1], bins=100, density=True, histtype='step')
plt.hist(feat_bg["mass"][feat_bg["primary"]==0], bins=100, density=True, histtype='step')


#%%
plt.figure()
a=feat_sg["tau2"]/feat_sg["tau1"]
b=feat_bg["tau2"]/feat_bg["tau1"]
plt.hist(b[feat_bg["primary"]==1], bins=100, density=True, histtype='step')
plt.hist(a[feat_sg["primary"]==1], bins=20, density=True, histtype='step')
plt.figure()
plt.hist(b[feat_bg["primary"]==0], bins=100, density=True, histtype='step')
plt.hist(a[feat_sg["primary"]==0], bins=20, density=True, histtype='step')
