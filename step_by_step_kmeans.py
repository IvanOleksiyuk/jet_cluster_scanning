import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score  # WHAT IS THIS?
from cluster_scanning import ClusterScanning
import matplotlib.pyplot as plt

class StepByStepKmeans:
    def __init__(self, n_clusters, batch_size=100, max_iter=100, verbose=False):
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.verbose = verbose

    def fit(self, X):
        self.X = X
        # Initialize K-means with 3 clusters


        # Perform k-means clustering
        centroids_history = []
        labels_history = []
        inertia_history = []
        sil_score_history = []

        for i in range(self.max_iter):  # Number of iterations
            
			kmeans = MiniBatchKMeans(
				n_clusters=self.n_clusters,
				init="k-means++",
				random_state=0,
				max_iter=1,
				)
            kmeans.fit(self.X)
            # Save centroid positions and labels at each iteration
            centroids_history.append(kmeans.cluster_centers_)
            labels_history.append(kmeans.labels_)
            inertia_history.append(kmeans.inertia_)
            sil_score_history.append(silhouette_score(self.X, kmeans.labels_))
            if self.verbose:
                print("Iteration", i + 1, "done")

        self.centroids_history = centroids_history
        self.labels_history = labels_history
        self.inertia_history = inertia_history
        self.sil_score_history = sil_score_history

    def get_centroids_history(self):
        return self.centroids_history

    def get_labels_history(self):
        return self.labels_history

    def get_inertia_history(self):
        return self.inertia_history

    def get_sil_score_history(self):
        return self.sil_score_history

    def get_best_iteration(self):
        return np.argmax(self.sil_score_history)

    def get_best_centroids(self):
        return self.centroids_history[self.get_best_iteration()]

    def get_best_labels(self):
        return self.labels_history[self.get_best_iteration()]

    def get_best_inertia(self):
        return self.inertia_history[self.get_best_iteration()]

    def get_best_sil_score(self):
        return self.sil_score_history[self.get_best_iteration()]

    def get_best_kmeans(self):
        return KMeans(
            n_clusters=self.n_clusters,
            init=self.get_best_centroids(),
            random_state=0,
            max_iter=1,
        )

    def get_best_kmeans_labels(self):
        return self.get_best_kmeans().labels_


if __name__ == "__main__":
    # Generate random data points
    cs = ClusterScanning(
        [
            "config/s0_0.5_1_new.yaml",
            "config/sig_frac/0.05.yaml",
            "config/restart/-1_0_0.yaml",
            "config/binning/CURTAINS.yaml",
            "config/tra_reg/3000_3100.yaml",
            "config/one_run_experiments.yaml",
        ]
    )
    cs.load_mjj()
    cs.load_data()
    cs.bootstrap_resample(cs.def_IDb)
    # will do nothing if self.cfg.bootstrap=False or self.def_IDb=-1
    cs.sample_signal_events(cs.def_IDs)
    # will do nothing if self.cfg.resample_signal=False
    data = cs.data_mjj_slise(cs.cfg.train_interval[0], cs.cfg.train_interval[1])
    print(len(data))
    np.random.seed(0)
    X = np.concatenate(
        (
            np.random.randn(500, 2) * 0.75 + np.array([1, 1]),
            np.random.randn(500, 2) * 0.5 + np.array([-2, -2]),
            # np.random.randn(100, 2) * 0.5 + np.array([3, -2]),
        )
    )

    sbsk = StepByStepKmeans(n_clusters=50, batch_size=100, max_iter=10, verbose=True)
    sbsk.fit(data)
    print("best iteration:", sbsk.get_best_iteration())
    print("best centroids:", sbsk.get_best_centroids())
    # print("best labels:", sbsk.get_best_labels())
    print("best inertia:", sbsk.get_best_inertia())
    print("best sil score:", sbsk.get_best_sil_score())
    print("inertia history:", sbsk.get_inertia_history())
    plt.plot(sbsk.get_inertia_history())
