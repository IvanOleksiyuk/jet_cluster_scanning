import os
import sys
#Make sure the path to the root directory of the project is in the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import colors

import numpy as np
import h5py
import time
from utils.config_utils import Config
from utils import set_matplotlib_default
from preproc.reprocessing import reweighting, gaussian_smearing, sum_1_norm

start_time = time.time()
cfg = Config("config/path.yaml")

data_file_path = cfg.get("data_directory")+"jet_images_v2.h5"#cfg.get("jet_images_file")
data_dataset_name = 'data'
mjj_file_path = cfg.get("data_directory")+cfg.get("clustered_jets_file")
mjj_dataset_name = 'm_jj'
lables_file_path = cfg.get("data_directory")+cfg.get("clustered_jets_file")
lables_dataset_name = 'labels'

output_images_bkg = cfg.get("data_directory") + cfg.get("sorted_bkg_file")
output_images_sig = cfg.get("data_directory") + cfg.get("sorted_sig_file")
output_mass_bkg = cfg.get("data_directory") + cfg.get("sorted_bkg_mass_file")
output_mass_sig = cfg.get("data_directory") + cfg.get("sorted_sig_mass_file")


# Load data (e.g. images)
input_f = h5py.File(data_file_path, 'r')
images = input_f[data_dataset_name]

# Load mjj
mjj_f = h5py.File(mjj_file_path, 'r')
mjj = mjj_f[mjj_dataset_name]
inf = mjj_f['jet_info']

# load lables
lables_f = h5py.File(lables_file_path, 'r')
labels = lables_f[lables_dataset_name][:]

# Get QCD masses
mjj_bkg = mjj[labels==0]

# Get signal masses
mjj_sig = mjj[labels==1]

inf_bkg = inf[labels==0]
inf_sig = inf[labels==1]


# Sort by mjj
ind_bkg=np.argsort(mjj_bkg)
ind_sig=np.argsort(mjj_sig)

# Visualise effect of reprocessing on a single image
image = images[-1, 1]
sigmas = [0, 1, 3]
ns = [1, 0.5, 0.25]
reprocessed_images = []
for sigma in sigmas:
	for n in ns:
		reprocessed_images.append(sum_1_norm(gaussian_smearing(reweighting(image, n), sigma, batch=False), batch=False))
#plot the images
fig, axs = plt.subplots(len(sigmas), len(ns), figsize=(6, 6))
for i in range(len(sigmas)):
	for j in range(len(ns)):
		axs[i, j].imshow(reprocessed_images[i*len(ns)+j], cmap=cm.turbo)
		axs[i, j].set_title("$\sigma$ = "+str(sigmas[i])+", n = "+str(ns[j]))
		axs[i, j].xaxis.set_ticklabels([])
		axs[i, j].xaxis.set_ticks([])
		axs[i, j].yaxis.set_ticklabels([])
		axs[i, j].yaxis.set_ticks([])
plt.tight_layout()
plt.savefig("plots/data/reprocessing.png", bbox_inches='tight')


# Plot a distriution of jet_masses
plt.figure()
plt.hist(inf_bkg[:, :, 0].flatten(), bins=100, label="QCD", histtype="step")
plt.hist(inf_sig[:, 0, 0].flatten(), bins=100, label="Signal_lead", histtype="step")
plt.hist(inf_sig[:, 1, 0].flatten(), bins=100, label="Signal_sublead", histtype="step")
plt.savefig("plots/data/jet_masses.png", bbox_inches='tight')
#plt.show()

print("#####bkg_data")
images_bkg = images[labels==0]


print("#####sig_data")
images_sig = images[labels==1]

def images_signal_order_by_mass(images_sig, inf_sig):
    for k in range(len(images_sig)):
        images_sig[k] = images_sig[k][np.argsort(inf_sig[k, :, 0])]
    return images_sig

images_sig = images_signal_order_by_mass(images_sig, inf_sig)
# Plot average backgroound image, average leading singal and average second leading jet
av_bkg = np.mean(images_bkg, axis=(0, 1))
av_sig_lead = np.mean(images_sig[:, 0], axis=0)
av_sig_subl = np.mean(images_sig[:, 1], axis=0)
vmax = max(np.max(av_bkg), np.max(av_sig_lead), np.max(av_sig_subl))
vmin = min(np.min(av_bkg[av_bkg!=0]), np.min(av_sig_lead[av_sig_lead!=0]), np.min(av_sig_subl[av_sig_subl!=0]))

fig = plt.figure(figsize=(9, 3))
gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.05])

# Plot images on subplots
ax1 = fig.add_subplot(gs[0])
ax1.imshow(np.mean(images_bkg, axis=(0, 1)), norm=colors.LogNorm(vmax=vmax, vmin=vmin), cmap=cm.turbo)
ax1.set_title("Average background image")
ax1.xaxis.set_ticklabels([])
ax1.xaxis.set_ticks([])
ax1.yaxis.set_ticklabels([])
ax1.yaxis.set_ticks([])

ax2 = fig.add_subplot(gs[1])
ax2.imshow(np.mean(images_sig[:, 0], axis=0), norm=colors.LogNorm(vmax=vmax, vmin=vmin), cmap=cm.turbo)
ax2.set_title("Average lighter signal jet")
ax2.xaxis.set_ticklabels([])
ax2.xaxis.set_ticks([])
ax2.yaxis.set_ticklabels([])
ax2.yaxis.set_ticks([])

ax3 = fig.add_subplot(gs[2])
im = ax3.imshow(np.mean(images_sig[:, 1], axis=0), norm=colors.LogNorm(vmax=vmax, vmin=vmin), cmap=cm.turbo)
ax3.set_title("Average heavier signal jet")
ax3.xaxis.set_ticklabels([])
ax3.xaxis.set_ticks([])
ax3.yaxis.set_ticklabels([])
ax3.yaxis.set_ticks([])

# Add colorbar
cax = fig.add_subplot(gs[3])
fig.colorbar(im, cax=cax)

# Adjust padding between subplots
plt.tight_layout()

def cout_nonzero_pixels(images):
	n_non_zero = []
	for image in images:
		n_non_zero.append(len(image[image!=0]))
	return n_non_zero

n_non_zero_bkg=cout_nonzero_pixels(images_bkg[:, 0])+cout_nonzero_pixels(images_bkg[:, 1])
n_non_zero_sig=cout_nonzero_pixels(images_sig[:, 0])+cout_nonzero_pixels(images_sig[:, 1])

# Show the plot
plt.savefig("plots/data/avarage_images.png", bbox_inches='tight')

plt.figure()
plt.hist(n_non_zero_bkg, bins=np.linspace(0, 100, 101), label="QCD", density=True, color="black", histtype="step")
plt.hist(n_non_zero_sig, bins=np.linspace(0, 100, 101), label="X, Y", density=True, color="red", histtype="step")
plt.xlabel("number of non-zero pixels")
plt.ylabel("fraction of all images")
print(max(n_non_zero_bkg))
print(max(n_non_zero_sig))
print(np.mean(n_non_zero_bkg))
print(np.mean(n_non_zero_sig))
plt.legend()
plt.savefig("plots/data/non_zero_pixels.png", bbox_inches='tight')