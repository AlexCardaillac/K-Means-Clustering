from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
from pandas import DataFrame
import numpy as np

class KMeans:

    def __init__(self, k=2):
        self.k = k
        self.centroids = []

    def train(self, inputs, max_iteration=500, seed=42, display=False):
        if display:
            import matplotlib.pyplot as plt

        min_x, min_y = np.amin(inputs, 0) - 1
        max_x, max_y = np.amax(inputs, 0) + 1

        self.centroids = np.array([[np.random.randint(min_x, max_x), np.random.randint(min_y, max_y)] for i in range(self.k)])
        prev_centroids = np.array([])
        closests = None

        while not np.array_equal(self.centroids, prev_centroids):
            prev_centroids = self.centroids
            closests = self.match_points(inputs)
            self.update_centroids(inputs, closests)

        closests = self.match_points(inputs)

        if display:
            self.display_plot(inputs, closests, (min_x, max_x), (min_y, max_y))

    def match_points(self, inputs):
        dist_table = []
        for centroid in self.centroids:
            dist_table.append(np.sqrt((inputs.T[0] - centroid[0]) ** 2 + (inputs.T[1] - centroid[1]) ** 2))
        return np.argmin(dist_table, 0)

    def update_centroids(self, inputs, cluster_map):
        for i, centroid in enumerate(self.centroids):
            centroid[0] = np.mean(inputs.T[0][cluster_map == i])
            centroid[1] = np.mean(inputs.T[1][cluster_map == i])

    def display_plot(self, inputs, cluster_map, x_lim, y_lim):
        fig = plt.figure()
        plt.scatter(inputs.T[0], inputs.T[1], c=cluster_map, cmap=plt.cm.Paired, alpha=0.6)
        plt.scatter(self.centroids.T[0], self.centroids.T[1], c=range(self.k), cmap=plt.cm.Paired, edgecolor='k')
        plt.xlim(x_lim[0], x_lim[1])
        plt.ylim(y_lim[0], y_lim[1])
        plt.show()

X, _ = make_blobs(n_samples=100, centers=3, n_features=2)

clusterer = KMeans(k=3)
clusterer.train(X, display=True)
