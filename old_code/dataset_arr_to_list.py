import pickle 
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 10})
cmap="gnuplot2_r"

with open("/media/ivan/Windows/datasets/100Kbg+@Ksg_feastures.pickle", "rb") as f:
    feat=pickle.load(f)
    
feat_sg={}
for key in feat.keys():
    feat_sg[key]=feat[key][feat["label"]==1]
    
    
feat_bg={}
for key in feat.keys():
    feat_bg[key]=feat[key][feat["label"]==0]

images_sg=pickle.load(open("/media/ivan/Windows/datasets/2K_SG.pickle", "rb"))

#plt.figure()
#images_sg=np.array(images_sg)
a=8
fig, axs = plt.subplots(nrows=a, ncols=a)

for i in range(a//2):
    for j in range(a):
        plt.sca(axs[i, j])
        axs[i, j].imshow(images_sg[8*i+j], cmap=cmap)
        plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
        plt.title("m{:.0f} t{:.2f}".format(feat_sg["mass"][8*i+j], feat_sg["tau2"][8*i+j]/feat_sg["tau1"][8*i+j]))
#plt.imshow(np.mean(images_sg, axis=0), cmap="turbo")

#pickle_out = open("C://datasets/2K_SGp.pickle", "wb")
#pickle.dump(images_sg, pickle_out)
#pickle_out.close()

images_bg=pickle.load(open("/media/ivan/Windows/datasets/100K_BG.pickle", "rb"))

for i in range(a//2, a):
    for j in range(a):
        plt.sca(axs[i, j])
        axs[i, j].imshow(images_bg[8*i+j], cmap=cmap)
        plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
        plt.title("m{:.0f} t{:.2f}".format(feat_bg["mass"][8*i+j], feat_bg["tau2"][8*i+j]/feat_bg["tau1"][8*i+j]))
#plt.figure()
#images_bg=np.array(images_bg)
#plt.imshow(np.mean(images_bg, axis=0), cmap="turbo")

#pickle_out = open("C://datasets/100K_BGp.pickle", "wb")
#pickle.dump(images_bg, pickle_out)
#pickle_out.close()
