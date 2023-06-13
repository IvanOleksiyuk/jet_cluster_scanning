# this file creates the central plot of the paper
# We see how much does significance change ehwn we chang the signl fraction
# %% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Imports block
import numpy as np
import scipy.stats as sts
from matplotlib import pyplot as plt
from scipy.special import gammainc
import pickle
from utils import p2Z

# %% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load the data block + define some parameters

# load the mass spectra
data_path = "../../DATA/LHCO/"
mjj_bg = np.load(data_path + "mjj_bkg_sort.npy")
mjj_sg = np.load(data_path + "mjj_sig_sort.npy")

# Choose the binning
binning = "CURTAINS"
if binning == "CURTAINS":
    n_bins = 16
    bins = np.linspace(3000, 4600, n_bins + 1)
elif binning == "BH":
    n_bins = 60
    bins = np.linspace(3000, 4600, n_bins + 1)


#%% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Py Bump Hunter block
# perform bump hunting using the pyBumpHunter package
import pyBumpHunter as BH


def get_hunter_scheme(hunter) -> np.ndarray:
    if hunter.str_scale == "lin":
        sig_str = np.arange(
            hunter.str_min,
            hunter.str_min + hunter.str_step * len(hunter.sigma_ar),
            step=hunter.str_step,
        )
    else:
        sig_str = np.array(
            [
                i % 10 * 10 ** (hunter.str_min + i // 10)
                for i in range(len(hunter.sigma_ar) + len(hunter.sigma_ar) // 10 + 1)
                if i % 10 != 0
            ]
        )
    return sig_str


hunter = BH.BumpHunter1D(
    rang=[3000, 4600],
    width_min=2,
    width_max=10,
    width_step=2,
    scan_step=1,
    npe=100000,
    nworker=6,
    seed=42,
    use_sideband=True,
    str_min=-4,
    sigma_limit=10,
    str_scale="log",
    bins=bins,
)

# Run the bump hunter without any cuts
hunter.signal_inject(mjj_sg, mjj_bg)
raw_str = get_hunter_scheme(hunter)
raw_sens = hunter.sigma_ar.copy()
min_l = min(len(raw_sens), len(raw_str))  # Sometimes the hunter lengths dont match
raw_sens = raw_sens[:min_l]
raw_str = raw_str[:min_l]
#%% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Cluster scanning block
# Load significances from the results of t_statistic_distribution.py
results = pickle.load(
    open(r"plots\test_stat\prep05_1_maxdev5CURTAINS_results.pickle", "rb")
)
print(results.keys())
#%% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Ideal fit block

# plt.plot(sig_frac, sign_ideal, color="red")
#%% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Plot
# Create the figure
fig = plt.figure(figsize=(5, 5))
plt.errorbar(
    raw_str,
    raw_sens[:, 0],
    xerr=0,
    yerr=[raw_sens[:, 1], raw_sens[:, 2]],
    marker="x",
    color="k",
    uplims=raw_sens[:, 2] == 0,
    label="pyBumpHunter" + str(n_bins),
)
results["Zs"] = np.array(results["Zs"])
print(results["Zs"])
results["Zs"][np.isinf(results["Zs"])] = p2Z(results["p_upper_bound"][0])
print(results["Zs"])
plt.plot(results["contaminations"], results["Z_mean_ps"], label="Cluster Scanning")
plt.plot(results["contaminations"], results["Z_meanres_ps"], label="Cluster Scanning")
plt.plot(
    results["contaminations"], np.mean(results["Zs"], axis=1), label="Cluster Scanning"
)
print(results["contaminations"])
print(results["Z_mean_ps"])
print(np.mean(results["Zs"], axis=1))
# plt.plot([0.005, 0.0025], [3.21, 0.92], color="red")
plt.legend()
plt.grid()
plt.xscale("log")
plt.xlabel("N signal * 10^(-5)")
plt.ylabel("significance [sigma]")

#%% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Save the plot
plt.savefig("plots/main/significances.png", bbox_inches="tight")

# %%
