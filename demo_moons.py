import matplotlib.pyplot as plt
import numpy as np
import pickle
import sklearn.datasets
moons_X, moons_y=sklearn.datasets.make_moons(noise=0.04, n_samples=10000)
data=moons_X[moons_y==1]
data=np.append(data, [[0.2, -0.6]], axis=0)
data=np.append(data, [[1, 0.2]], axis=0)
data=np.append(data, [[1.21, -0.21]], axis=0)
data=np.append(data, [[1.22, -0.23]], axis=0)
data=np.append(data, [[1.25, -0.22]], axis=0)
data=np.append(data, [[1.2, 0.5]], axis=0)
data=np.append(data, [[1.21, 0.51]], axis=0)
data=np.append(data, [[1.22, 0.49]], axis=0)
data=np.append(data, [[1.25, 0.52]], axis=0)
data=np.append(data, [[1.2, -0.2]], axis=0)
data=np.append(data, [[0.5, 0.4]], axis=0)
data=np.append(data, [[1.5, 0.2]], axis=0)
data=np.append(data, [[0.5, 0.02]], axis=0)

plt.scatter(data[:, 0], data[:, 1])

pickle_out = open("C://datasets/moon_demo.pickle", "wb")
pickle.dump(data, pickle_out)
pickle_out.close()