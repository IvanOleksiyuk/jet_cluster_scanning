import numpy as np
import matplotlib.pyplot as plt
import pickle 

def ft(fromm, too):
    return str(fromm//1000)+"k-"+str(too//1000)+"k-"

def muller(dim, N):
    X=np.random.normal(0, 1, (N, dim))
    l=np.sum(X**2, axis=1)**0.5
    return X/l.reshape((-1, 1))
    
np.random.seed(10)

gpath={}
gpath["images"]="C:/bachelor work/Spyder/image_data_sets/"
num_tra=100000
num_val=20000
num_tes=20000
num_tot=num_tra+num_val+num_tes
DATASET=8.1

#two gaussians with background one being much broader bu the signal one has a mean dispalaced from the centre
if DATASET==5:
    bg_name="2d10s"
    sg_name="2d1s6m"
    SAVE=True
    X=np.random.normal(loc=(0, 0), scale=(10, 1), size=(num_tot, 2))
    Y=np.random.normal(loc=(0, 6), scale=(1, 1), size=(num_tot, 2))
# A uniform distribution from 0 to 1 and as signal from 0 to 2
if DATASET==5.1:
    bg_name="1d1Us"
    sg_name="1d2Us"
    SAVE=True
    X=np.random.uniform(low=-0.5, high=0.5, size=(num_tot, 1))
    Y=np.random.uniform(low=-1, high=1, size=(num_tot, 1))
    
if DATASET==5.2:
    bg_name="2d1Us"
    sg_name="2d2Us"
    SAVE=True
    X=np.random.uniform(low=-0.5, high=0.5, size=(num_tot, 2))
    Y=np.random.uniform(low=-1, high=1, size=(num_tot, 2))

if DATASET==5.3:
    bg_name="5d1Us"
    sg_name="5d2Us"
    SAVE=True
    X=np.random.uniform(low=-0.5, high=0.5, size=(num_tot, 5))
    Y=np.random.uniform(low=-1, high=1, size=(num_tot, 5))
    
if DATASET==5.4:
    bg_name="tor10u2s"
    sg_name="2d7s"
    SAVE=True
    phi=np.random.uniform(low=-np.pi, high=np.pi, size=(num_tot))
    R=10
    r=np.random.normal(loc=(0), scale=(2), size=(num_tot))
    X=np.empty(shape=(num_tot, 2))
    X[:, 0]=(R+r)*np.sin(phi)
    X[:, 1]=(R+r)*np.cos(phi)
    Y=np.random.normal(loc=(0, 0), scale=(7, 7), size=(num_tot, 2))
    
if DATASET==5.5:
    bg_name="2d10s"
    sg_name="2d1s4m"
    SAVE=True
    X=np.random.normal(loc=(0, 0), scale=(10, 1), size=(num_tot, 2))
    Y=np.random.normal(loc=(0, 4), scale=(1, 1), size=(num_tot, 2))
    
if DATASET==6:
    bg_name="2dslope"
    sg_name="2dus"
    SAVE=True
    X=np.zeros(shape=(num_tot, 2))
    X[:, 0]=1-np.sqrt(np.random.uniform(low=-0, high=1, size=(num_tot)))
    X[:, 1]=np.random.uniform(low=0, high=1, size=(num_tot))
    Y=np.random.uniform(low=-0.5, high=1.5, size=(num_tot, 2))
    
if DATASET==6.1:
    dim=5
    bg_name="{:}dslope".format(dim)
    sg_name="{:}dus".format(dim)
    SAVE=True
    X=np.zeros(shape=(num_tot, dim))
    X[:, 0]=1-np.sqrt(np.random.uniform(low=-0, high=1, size=(num_tot)))
    X[:, 1:]=np.random.uniform(low=0, high=1, size=(num_tot, dim-1))
    Y=np.random.uniform(low=-0.5, high=1.5, size=(num_tot, dim))
    
if DATASET==7:
    bg_name="2dexp1"
    sg_name="2du2-1.5"
    SAVE=True
    X=np.zeros(shape=(num_tot, 2))
    X[:, 0]=np.random.exponential(size=(num_tot))
    X[:, 1]=np.random.uniform(low=0, high=1, size=(num_tot))
    Y=np.random.uniform(low=0, high=1, size=(num_tot, 2))
    Y[:, 1]*=2
    Y[:, 1]-=0.5
    Y[:, 0]*=12
    Y[:, 0]-=1
    
if DATASET==8:
    bg_name="2d1s"
    sg_name="tor4u1s" 
    SAVE=True
    phi=np.random.uniform(low=-np.pi, high=np.pi, size=(num_tot))
    R=4
    r=np.random.normal(loc=(0), scale=(1), size=(num_tot))
    Y=np.empty(shape=(num_tot, 2))
    Y[:, 0]=(R+r)*np.sin(phi)
    Y[:, 1]=(R+r)*np.cos(phi)
    X=np.random.normal(loc=(0, 0), scale=(1, 1), size=(num_tot, 2))
    
if DATASET==8.1:
    bg_name="5d1s"
    sg_name="tor5d4u1s" 
    SAVE=True
    phi=np.random.uniform(low=-np.pi, high=np.pi, size=(num_tot))
    R=4
    r=np.random.normal(loc=(0), scale=(1), size=(num_tot))
    Y=muller(5, num_tot)*(R+r.reshape((-1, 1)))
    X=np.random.normal(loc=0, scale=1, size=(num_tot, 5))
    

bg_tr=X[:num_tra]
sg_tr=Y[:num_tra]
bg_val=X[num_tra:num_tra+num_val]
sg_val=Y[num_tra:num_tra+num_val]
bg_tes=X[num_tra+num_val:]
sg_tes=Y[num_tra+num_val:]
print(len(bg_tr))
print(len(bg_val))
print(len(bg_tes))

if SAVE:
    pickle_out = open(gpath["images"]+"Xtra_"+bg_name+ft(0, num_tra)+".pickle", "wb")
    pickle.dump(bg_tr, pickle_out)
    pickle_out.close()
    
    pickle_out = open(gpath["images"]+"Xval_"+bg_name+ft(num_tra, num_tra+num_val)+".pickle", "wb")
    pickle.dump(bg_val, pickle_out)
    pickle_out.close()
    
    pickle_out = open(gpath["images"]+"Xtes_"+bg_name+ft(num_tra+num_val, num_tra+num_val+num_tes)+".pickle", "wb")
    pickle.dump(bg_tes, pickle_out)
    pickle_out.close()

    pickle_out = open(gpath["images"]+"Xtra_"+sg_name+ft(0, num_tra)+".pickle", "wb")
    pickle.dump(sg_tr, pickle_out)
    pickle_out.close()
    
    pickle_out = open(gpath["images"]+"Xval_"+sg_name+ft(num_tra, num_tra+num_val)+".pickle", "wb")
    pickle.dump(sg_val, pickle_out)
    pickle_out.close()
    
    pickle_out = open(gpath["images"]+"Xtes_"+sg_name+ft(num_tra+num_val, num_tra+num_val+num_tes)+".pickle", "wb")
    pickle.dump(sg_tes, pickle_out)
    pickle_out.close()
    
plt.scatter(bg_tr[:, 0], bg_tr[:, 1], s=0.1)
plt.scatter(sg_tr[:, 0], sg_tr[:, 1], s=0.1)
plt.xlim(-40, 40)
plt.ylim(-40, 40)