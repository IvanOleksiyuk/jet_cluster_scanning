import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import utils.set_matplotlib_default
from scipy import stats
folder="char/one_run_experiments/k50MB2048_1iret0con0.05W3000_3100_w0.5s1Nboot/binnedW100s16ei30004600/"
files_list = os.listdir(folder)
bres_files = [file for file in files_list if file.startswith("bres")]

def load_counts_windows(path):
    res = pickle.load(open(path, "rb"))
    if isinstance(res, list) or isinstance(res, np.ndarray):
        return res
    else:
        return res["counts_windows"]

def test_sample(sample):
    # Shapiro-Wilk Test
    shapiro_statistic, shapiro_p_value = stats.shapiro(sample)
    print(f"Shapiro-Wilk Test - Statistic: {shapiro_statistic}, p-value: {shapiro_p_value}")

    # Kolmogorov-Smirnov Test
    ks_statistic, ks_p_value = stats.kstest(sample, 'norm')
    print(f"Kolmogorov-Smirnov Test - Statistic: {ks_statistic}, p-value: {ks_p_value}")

    # Jarque-Bera Test
    jarque_bera_statistic, jarque_bera_p_value = stats.jarque_bera(sample)
    print(f"Jarque-Bera Test - Statistic: {jarque_bera_statistic}, p-value: {jarque_bera_p_value}")
    return shapiro_p_value, ks_p_value, jarque_bera_p_value

count_window_list = [load_counts_windows(folder + file) for file in bres_files[:1000]]
def preprocess(count_window):
    count_window /= np.sum(count_window, axis=0)
    count_window = count_window.T
    count_window -= np.mean(count_window, axis=0)
    count_window /= np.std(count_window, axis=0)
    return count_window
count_windows_bg = count_window_list[0][0].astype(dtype=np.float64)
count_windows_bg = preprocess(count_windows_bg)
count_window_full = count_window_list[0][0].astype(dtype=np.float64) + count_window_list[0][1].astype(dtype=np.float64)
count_window_full = preprocess(count_window_full)
print(count_window_full.flatten())

plt.figure()
plt.grid()
for count_window in count_windows_bg:
    plt.plot(count_window, color="blue")

#plt.savefig("experiment.png")

def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))/np.sqrt(2*np.pi*sigma**2)

plt.figure()
bins = np.linspace(-4, 4, 20+1)
shapiro_p_value, ks_p_value, jarque_bera_p_value = test_sample(count_windows_bg.flatten())
plt.hist(count_windows_bg.flatten(), bins=bins, density=True, alpha=0.5, label="Background $p_{SW}$"+f"={shapiro_p_value:.2f}\n"+" $p_{KS}$"+f"={ks_p_value:.2f}"+" $p_{JB}$"+f"={jarque_bera_p_value:.2f}")
shapiro_p_value, ks_p_value, jarque_bera_p_value = test_sample(count_window_full.flatten())
plt.hist(count_window_full.flatten(), bins=bins, density=True, alpha=0.5, label="Bkg+sig $p_{SW}$"+f"={shapiro_p_value:.2f}\n"+" $p_{KS}$"+f"={ks_p_value:.2f}"+" $p_{JB}$"+f"={jarque_bera_p_value:.2f}")
x=np.linspace(-4, 4, 100)
plt.hist(np.random.normal(loc=0, scale=1, size=10_000_000), bins=bins, label='Unit Gaussian', color='red', histtype='step', linewidth=1, density=True)
plt.legend()
#plt.yscale("log")
plt.ylabel("density")
plt.xlabel("standardized normalized counts")
plt.savefig("plots/misc/gausianity_check.png", bbox_inches="tight")
# plt.yscale("log")
# plt.savefig("experiment3.png")


# Generate a sample of 800 numbers (replace this with your data)
#np.random.seed(42)]
sample = np.random.normal(loc=0, scale=1, size=80000)



