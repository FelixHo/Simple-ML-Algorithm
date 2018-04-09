# encoding:utf-8
import numpy as np


class KMeans(object):
    def __init__(self, k):
        self.k = k
        self.center = None
        self.error = None

    def _init_center(self, X):
        """
        1. randomly select one center point
        2. for each point, calculate the shortest distance to the center points, denoted as D
        3. find the point with the largest D as a new center point
        4. repeat 2-3 steps until k center points are found
        """
        cids = [np.random.randint(X.shape[0])]

        for _ in range(self.k - 1):
            self.center = X[cids]
            max_dist = -1
            cid = None

            for i, point in enumerate(X):
                dist = np.min(np.linalg.norm(point - self.center, axis=1))

                if dist > max_dist:
                    max_dist = dist
                    cid = i

            cids.append(cid)

        self.center = X[cids]

    def _build_center(self, cluster):
        """
        recompute the new cluster centers as the center of mass of the points assigned to the clusters
        """
        c = []

        for points in cluster:
            c.append(np.mean(points, axis=0))

        self.center = np.array(c)

    def _find_center(self, point):
        """
        find the nearest center point
        """
        return np.argmin(np.linalg.norm(point - self.center, axis=1))

    def _build_cluster(self, X):
        """
        assign each point to its nearest cluster center
        """
        cluster = [[] for _ in range(self.k)]

        for point in X:
            c = self._find_center(point)
            cluster[c].append(point)

        return cluster

    def _cal_error(self, cluster):
        """
        error = SSE(sum of square error) of the respective clusters
        """
        error = 0

        for c, points in enumerate(cluster):
            error += np.sum((points - self.center[c]) ** 2)

        return error

    def fit(self, X, epoch=5):
        cluster = None

        for i in range(epoch):
            if i == 0:
                self._init_center(X)
            else:
                self._build_center(cluster)
            cluster = self._build_cluster(X)

        self.error = self._cal_error(cluster)

        return self

    def predict(self, X):
        label = []
        for point in X:
            label.append(self._find_center(point))
        return label


if __name__ == '__main__':
    from sklearn.datasets import make_blobs
    import matplotlib.pyplot as plt

    k = 5
    X, y = make_blobs(n_samples=500, centers=k, n_features=2, random_state=42)

    best_model = None
    best_error = None

    for i in range(10):
        model = KMeans(k=k)
        model.fit(X, epoch=10)

        if i ==0:
            best_model = model
            best_error = model.error
        else:
            if best_error > model.error:
                best_error = model.error
                best_model = model

    label = best_model.predict(X)
    center = best_model.center

    ####### plot ######################
    plt.figure(1)
    plt.subplot(121)
    plt.title('my KMeans')
    plt.scatter(X[:, 0], X[:, 1], c=label)
    plt.scatter(center[:, 0], center[:, 1], c='r', marker='+')

    ####### compare to scikit ########
    from sklearn.cluster import KMeans as KM

    km = KM(n_clusters=k)
    km.fit(X)
    plt.subplot(122)
    plt.title('scikit KMeans')
    plt.scatter(X[:, 0], X[:, 1], c=km.labels_)
    plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], c='r', marker='+')
    plt.show()
