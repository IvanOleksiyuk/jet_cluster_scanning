import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial import Voronoi, voronoi_plot_2d

# Generate random data
np.random.seed(42)
num_samples = 5000
data = np.random.randn(num_samples, 2)

n_outliers = 30
outliers_1 = np.random.uniform(low=-4, high=4, size=(n_outliers, 2))
outliers_2 = np.random.randn(n_outliers, 2)+ np.array([2, 2])
outliers_3 = np.random.randn(n_outliers, 2)/5 + np.array([1.5, -0.1])
outliers_4 = np.random.randn(n_outliers, 2)/2 + np.array([3.2, 3.2])

outliers = [outliers_1, outliers_2, outliers_4, outliers_3]
text_labels = ['Uniformly dispersed outliers', 'Shift anomaly', 'Outlier cluster',  'Overdensity in the bulk', ]
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

for i, ax in enumerate(axs.flatten()):
	outliers_i=outliers[i]
	labels_i=text_labels[i]
	# Perform k-means clustering
	num_clusters = 10  # You can adjust this value
	kmeans = KMeans(n_clusters=num_clusters)
	labels = kmeans.fit_predict(np.concatenate((data, outliers_i)))
	plt.sca(ax)
	# Display the Voronoi diagram
	vor = Voronoi(kmeans.cluster_centers_)
	voronoi_plot_2d(vor, ax=ax, show_vertices=False, show_points=False)
	plt.scatter(data[:, 0], data[:, 1], c=labels[:len(data)], cmap='winter', alpha=0.7, s=2)
	plt.scatter(outliers_i[:, 0], outliers_i[:, 1], c="red", alpha=0.7, s=2)
	plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='black', marker='x', s=100)
	plt.title(labels_i)
	plt.xlabel('$x_1$')
	plt.ylabel('$x_2$')
	#plt.colorbar(label='Cluster Labels')
	plt.xlim([-4, 4])
	plt.ylim([-4, 4])

plt.tight_layout()
plt.savefig("voronoi.png", bbox_inches='tight')
plt.show()
