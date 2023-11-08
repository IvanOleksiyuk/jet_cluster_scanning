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
import os
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

start_time = time.time()
if len(sys.argv) == 1:
	cfg = Config("config/path.yaml")
else:
	cfg = Config(["config/path.yaml", sys.argv[1]])
os.makedirs(cfg.get("plots_directory")+"data/", exist_ok=True)

data_file_path = cfg.get("data_path")+"jet_images_v2.h5"#cfg.get("jet_images_file")
data_dataset_name = 'data'
mjj_file_path = cfg.get("data_path")+cfg.get("clustered_jets_file")
mjj_dataset_name = 'm_jj'
lables_file_path = cfg.get("data_path")+cfg.get("clustered_jets_file")
lables_dataset_name = 'labels'

output_images_bkg = cfg.get("data_path") + cfg.get("sorted_bkg_file")
output_images_sig = cfg.get("data_path") + cfg.get("sorted_sig_file")
output_mass_bkg = cfg.get("data_path") + cfg.get("sorted_bkg_mass_file")
output_mass_sig = cfg.get("data_path") + cfg.get("sorted_sig_mass_file")

# Plots to save 
pts = ["inv_mass", "images", "non_zero_pixels", "reprocessing"]

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

# Get masses
mjj_bkg = mjj[labels==0]
mjj_sig = mjj[labels==1]

# Get high level features
inf_bkg = inf[labels==0]
inf_sig = inf[labels==1]

# Plot invariant mass spectrum 
def plot_invariant_mass_spectrum(mjj_bkg, mjj_sig):
	plt.figure()
	region=(3000, 4600)#
	tr_region=(3000, 3100)
	n_QCD = len(mjj_bkg[(mjj_bkg>region[0]) & (mjj_bkg<region[1])])
	n_Zp = len(mjj_sig[(mjj_sig>region[0]) & (mjj_sig<region[1])])
	binning = np.linspace(1700, 7000, 53+1)
	plt.hist(mjj_bkg, bins=binning, label="QCD", histtype="step")
	plt.hist(mjj_sig, bins=binning, label="all Z'", histtype="step")
	print("resonance width: ", np.std(mjj_sig[(mjj_sig>region[0]) & (mjj_sig<region[1])]))
	plt.hist(np.concatenate([mjj_bkg, np.random.choice(mjj_sig, 5000, replace=False)]), bins=binning, label="1M QCD, 5K Z'", histtype="step")
	plt.axvline(x=3000, color="black", label=f"Selection, \n {n_QCD} QCD, \n {n_Zp} Z'")
	plt.axvline(x=4600, color="black")
	plt.xlabel("$m_{jj}$ [GeV]")
	plt.ylabel("Number of events")
	#plt.yscale("log")
	plt.grid()
	plt.legend()
	plt.savefig(cfg.get("plots_directory")+"data/mjj.png", bbox_inches='tight')
	print("QCD points in the training region: ", len(mjj_bkg[(mjj_bkg>tr_region[0]) & (mjj_bkg<tr_region[1])]))
	print("Z' points in the training region: ", len(mjj_sig[(mjj_sig>tr_region[0]) & (mjj_sig<tr_region[1])]))




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
plt.savefig(cfg.get("plots_directory")+"data/reprocessing.png", bbox_inches='tight')


# Plot a distriution of jet_masses
plt.figure()
plt.hist(inf_bkg[:, :, 0].flatten(), bins=100, label="QCD", histtype="step")
plt.hist(inf_sig[:, 0, 0].flatten(), bins=100, label="Signal_lead", histtype="step")
plt.hist(inf_sig[:, 1, 0].flatten(), bins=100, label="Signal_sublead", histtype="step")
#plt.savefig(cfg.get("plots_directory")+"data/jet_masses.png", bbox_inches='tight')
#plt.show()

### Get and separate jet images:
print("#####bkg_data")
images_bkg = images[labels==0]


print("#####sig_data")
images_sig = images[labels==1]

def images_signal_order_by_mass(images_sig, inf_sig):
    for k in range(len(images_sig)):
        images_sig[k] = images_sig[k][np.argsort(inf_sig[k, :, 0])]
    return images_sig

images_sig = images_signal_order_by_mass(images_sig, inf_sig)

def plot_averages(images_bkg, images_sig, cfg):
	logger.info("Plotting average images")
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

	# Show the plot
	plt.savefig(cfg.get("plots_directory")+"data/avarage_images.png", bbox_inches='tight')

def cout_nonzero_pixels(images):
	n_non_zero = []
	for image in images:
		n_non_zero.append(len(image[image!=0]))
	return n_non_zero

def plot_and_nonzero_pixels(images_bkg, images_sig, cfg):
	logger.info("Plotting nonzero pixel distribution")
	n_non_zero_bkg=cout_nonzero_pixels(images_bkg[:, 0])+cout_nonzero_pixels(images_bkg[:, 1])
	n_non_zero_sig=cout_nonzero_pixels(images_sig[:, 0])+cout_nonzero_pixels(images_sig[:, 1])
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
	plt.savefig(cfg.get("plots_directory")+"data/non_zero_pixels.png", bbox_inches='tight')

def plot_and_save_background_jet_images(images_bkg, cfg, num_rows=2, num_cols=3):
    logger.info("Plotting 4 background jet images")
    num_images_to_plot = num_rows * num_cols
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols*2, num_rows*2))
    
    images_to_plot = images_bkg[100:100+num_images_to_plot]
    reprocessed_images = []
    for image in images_to_plot:
        reprocessed_images.append(sum_1_norm(gaussian_smearing(reweighting(image[0], 0.5), 1, batch=False), batch=False))
    images_to_plot = reprocessed_images

    for i in range(num_images_to_plot):
        row = i // num_cols
        col = i % num_cols
        axs[row, col].imshow(images_to_plot[i], cmap=cm.turbo)
        axs[row, col].set_title(f"Jet {i + 1}")
        axs[row, col].xaxis.set_ticklabels([])
        axs[row, col].xaxis.set_ticks([])
        axs[row, col].yaxis.set_ticklabels([])
        axs[row, col].yaxis.set_ticks([])

    plt.tight_layout()
    output_path = cfg.get("plots_directory") + "data/background_jet_images.png"
    plt.savefig(output_path, bbox_inches='tight')

# Run the functions:
if "inv_mass" in pts:
	plot_invariant_mass_spectrum(mjj_bkg, mjj_sig)
#plot_averages(images_bkg, images_sig, cfg)
#plot_and_nonzero_pixels(images_bkg, images_sig, cfg)
plot_and_save_background_jet_images(images_bkg, cfg)

