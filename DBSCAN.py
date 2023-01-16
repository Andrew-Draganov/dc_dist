# __author__ = "Christian Frey"
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import scipy
import numpy as np

class DBSCAN(object):
    '''
    This class implements the density based spatial clustering in applications with noise algorithm

    Arguments:
      eps: epsilon being used to identify the eps-neighborhood of a datapoint
      min_pts: minimal number of points s.t. a datapoint can be considered as a core object if the number
        of points in the eps-neighborhood exceeds min_pts
      dist_method: distance function being used to calculate the proximity of two datapoints
      cluster_type: 'standard' will apply the basic procedure for DBSCAN, set to 'corepoints' if the procedure
        shall just consider core points being labeled, i.e., borderpoints are not attached to a cluster

    Properties:
      eps: epsilon value for the eps-neighborhood
      min_pts: minimal number of points for identifying core objects
      dist_method: distance fnc
      labels: labels of the datapoints, i.e., the affiliation of the points to the clusters
    '''

    def __init__(self, *, eps, min_pts, dist_method='euclidean', cluster_type="standard"):
        if cluster_type not in ['standard', 'corepoints']:
            raise AssertionError("Please select 'standard' or 'corepoints' for the cluster_type parameter.")
        self.eps = eps
        self.min_pts = min_pts
        self.dist_method = dist_method
        self.labels_ = None
        self.type_ = cluster_type

    def fit(self, data):
        '''
        This method executes the DBSCAN algorithm on the attached
        dataset. First, it calculates the distances for each point
        within the dataset to each other point. By iterating the whole
        dataset, we can identify the affiliation for the points to
        clusters. If a core point is found, the point
        is used to expand the cluster, i.e., by calling the subrouting
        _expand_cluster(.), we can identfy each point being density
        reachable from a core point. If for an datapoint the conditions
        for a core point do not hold, we can regard this point as a
        Noise point (probably border point). After the cluster expansion,
        we take the next unlabeled point and continue in the same manner
        as described till every point is labeled.

        Arguments:
          data: the dataset
        '''
        self.X = data
        m, dim = data.shape
        dist_mx = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(data))
        self.labels_ = np.full(m, np.nan)
        cluster_idx = 0
        for i in range(m):
            if not np.isnan(self.labels_[i]): continue
            neighbors = np.where(dist_mx[i] < self.eps)[0]
            if len(neighbors) < self.min_pts:
                self.labels_[i] = -1  # Noise
                continue
            cluster_idx += 1
            self.labels_[i] = cluster_idx
            seed = neighbors
            seed_set = seed.tolist()

            self._expand_cluster(i, seed_set, cluster_idx, dist_mx)

    def _expand_cluster(self, point, seed_set, cluster_idx, dist_mx):
        '''
        This method is used to aggregate all points being density
        reachable by the datapoint being attached as parameter ('point').
        All points which have been marked to be NOISE may be changed, if
        they are density-reachable from some other point within the cluster.
        This happens in the standard procedure for border points of a cluster.
        If we just consider core points, the 'border points' remain as noise points.

        Arguments:
          point: the datapoint from which on we expand the cluster
          seed_set: a set containing all points being density reachable
            within the cluster being currently regarded
          cluster_idx: the id of the current cluster
          dist_mx: a distance matrix NxN containing the proximity for
            each point in the dataset to each other point
        '''
        for s in seed_set:
            if np.isnan(self.labels_[s]) or self.labels_[s] == -1:
                neighbors = np.where(dist_mx[s] < self.eps)[0]
                # label point to cluster iff borderpoints shall be considered

                if self.type_ == "standard":
                    self.labels_[s] = cluster_idx

                if len(neighbors) < self.min_pts:
                    continue

                if self.type_ == "corepoints":
                    self.labels_[s] = cluster_idx

                for j in neighbors:
                    try:
                        seed_set.index(j)
                    except ValueError:
                        seed_set.append(j)

    def plot2D(self, groundtruth=None):
        if self.X is None:
            raise AssertionError("First call fit(.) on a dataset X")
        if self.X.shape[1] != 2:
            raise AssertionError("plot2D ist just used for plotting 2 dimensional data")

        fig = plt.figure(figsize=(12, 12))
        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)

        if groundtruth is None:
            ax1.scatter(self.X[:, 0], self.X[:, 1], s=50)
        else:
            ax1.scatter(self.X[:, 0], self.X[:, 1], c=groundtruth, s=50)
        ax1.set_title("Raw dataset")
        ax1.set_xlabel("1st feature")
        ax1.set_ylabel("2nd feature")

        ax2.scatter(self.X[:, 0], self.X[:, 1], c=self.labels_, s=50, cmap='viridis')
        ax2.set_title("clustered data - $minPts:{minPts}$, $\epsilon:{eps}$".format(minPts=self.min_pts, eps=self.eps))
        ax2.scatter(self.X[:, 0][self.labels_ != -1], self.X[:, 1][self.labels_ != -1],
                    c=self.labels_[self.labels_ != -1], s=50, cmap='viridis')
        ax2.scatter(self.X[:, 0][self.labels_ == -1], self.X[:, 1][self.labels_ == -1], s=100, color="red")
        ax2.set_xlabel("1st feature")
        ax2.set_ylabel("2nd feature")

        plt.show()


if __name__ == '__main__':
    X, y_true = make_moons(n_samples=250, noise=0.05, random_state=42)
    dbscan = DBSCAN(eps=0.1, min_pts=4, cluster_type='corepoints')
    dbscan.fit(X)
    dbscan.plot2D(y_true)