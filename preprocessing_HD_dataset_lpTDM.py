import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import preprocessing
import pickle

IMG_SIZE=40
num_tra=0
num_val=0
num_tes=20000
num_tot=num_tra+num_val+num_tes
gpath={}
gpath["images"]="C:/bachelor work/Spyder/image_data_sets/"

events_data_path="C:/bachelor work/Spyder/data_set/DM_HB_lowpt/jets_q50m10lpt.npy"
output_name="jets_q50m10lpt"
signal=False

events=np.load(events_data_path)
print("found", len(events), "events")
images=[]
k=0
for event in events[:num_tot]:
    if k % 1000 == 0:
        print(k)
    images.append(preprocessing.calorimeter_image_ptethaphi(event, IMG_SIZE=IMG_SIZE))
    k+=1

images=np.array(images).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

if signal:
    y = np.ones((len(images)))
else:
    y = np.zeros((len(images)))

def ft(fromm, too):
    return str(fromm//1000)+"k-"+str(too//1000)+"k-"

pickle_out = open(gpath["images"]+"Xtra_"+output_name+ft(0, num_tra)+".pickle", "wb")
pickle.dump(images[:num_tra], pickle_out)
pickle_out.close()

pickle_out = open(gpath["images"]+"ytra_"+output_name+ft(0, num_tra)+".pickle", "wb")
pickle.dump(y[:num_tra], pickle_out)
pickle_out.close()

pickle_out = open(gpath["images"]+"Xval_"+output_name+ft(num_tra, num_tra+num_val)+".pickle", "wb")
pickle.dump(images[num_tra:(num_tra+num_val)], pickle_out)
pickle_out.close()

pickle_out = open(gpath["images"]+"yval_"+output_name+ft(num_tra, num_tra+num_val)+".pickle", "wb")
pickle.dump(y[num_tra:(num_tra+num_val)], pickle_out)
pickle_out.close()

pickle_out = open(gpath["images"]+"Xtes_"+output_name+ft(num_tra+num_val, num_tra+num_val+num_tes)+".pickle", "wb")
pickle.dump(images[(num_tra+num_val):], pickle_out)
pickle_out.close()

pickle_out = open(gpath["images"]+"ytes_"+output_name+ft(num_tra+num_val, num_tra+num_val+num_tes)+".pickle", "wb")
pickle.dump(y[(num_tra+num_val):], pickle_out)
pickle_out.close()

#plotting a single image for example:
eta_bonds=[-0.8, 0.8]
phi_bonds=[-0.8, 0.8]

image=images[1]
matplotlib.rcParams.update({'font.size': 18})
fig, ax = plt.subplots(figsize=(8, 6))
plt.xlabel("$\eta$")
plt.ylabel("$\phi$")
ax.imshow(image, cmap=cm.gnuplot2, interpolation='nearest', extent=eta_bonds+phi_bonds, origin='lower')
plt.savefig("example.png")

