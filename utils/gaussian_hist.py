import numpy as np
import matplotlib.pyplot as plt

def get_analytic_gaussian_hist(mu, sigma, bins=np.linspace(-4, 4, 20+1), density=True, sum_bins=None):
	# Generate values for x
	x = np.linspace(-7, 7, 10000)

	# Calculate the probability density function (PDF) of the Gaussian distribution
	pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

	# Calculate the probability for each bin
	bin_width = bins[1] - bins[0]
	if density:
		probabilities = [np.mean(pdf[(x >= bins[i]) & (x < bins[i+1])]) for i in range(len(bins)-1)]
	else:
		probabilities = [np.mean(pdf[(x >= bins[i]) & (x < bins[i+1])]) * bin_width for i in range(len(bins)-1)]

	probabilities = np.array(probabilities)
	if sum_bins is not None:
		probabilities=probabilities/np.sum(probabilities)
		probabilities=probabilities*sum_bins
 
	return probabilities, bins, bin_width

def plot_analytic_gaussian_hist(bins=None, sum_bins=None, label=None):
	probabilities, bins, bin_width = get_analytic_gaussian_hist(0, 1, bins=bins, sum_bins=sum_bins)
	# Plot the histogram-like representation
	plt.step((bins[:-1]+bins[1:])/2, probabilities, where='mid', color='black', label=label)