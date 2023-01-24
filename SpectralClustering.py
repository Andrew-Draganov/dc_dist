# __author__ = "Christian Frey"
from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.sparse.csgraph import laplacian as csgraph_laplacian
from sklearn.utils import check_random_state
from sklearn.manifold._spectral_embedding import _graph_is_connected, _set_diag, _deterministic_vector_sign_flip
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import warnings

from density_tree import make_tree
from distance_metric import get_nearest_neighbors
from DBSCAN import DBSCAN


class SpectralClustering_own(object):

    def __init__(self, n_clusters=4, n_vecs=None):
        self.clustering = None
        self.eigenvals = None
        self.eigenvecs = None
        self.n_clusters = n_clusters
        if n_vecs is None:
            self.n_vecs = n_clusters
        else:
            self.n_vecs = n_vecs

    def _calc_kmeans(self, n_cluster, n_vecs):
        # kmeans = KMeans(n_clusters=n_cluster, n_init='auto') ### <-- Original code
        kmeans = KMeans(n_clusters=n_cluster, n_init=5) ### <-- Andrew change 
        kmeans.fit(self.eigenvecs[:, :n_vecs])
        return kmeans

    def _get_laplacian(self, normalized=True):
        if normalized:
            laplacian_ = nx.normalized_laplacian_matrix(self.G, weight="weight")
        else:
            laplacian_ = nx.laplacian_matrix(self.G, weight='weight')
        return laplacian_.todense()

    def fit(self, affinity_mx, normalized=True):
        # Call from API
        # maps = spectral_embedding(affinity, n_components=n_components,
        #                      eigen_solver=eigen_solver,
        #                      random_state=random_state,
        #                      eigen_tol=eigen_tol, drop_first=False)

        self.eigenvals, self.eigenvecs, self.eigenvalues_full, self.eigenvecors_full = \
            self.spectral_embedding(affinity_mx,
                                    n_components=self.n_vecs,
                                    drop_first=False,
                                    norm_laplacian=False)
        if affinity_mx.shape[0] > 1:
            self.clustering = self._calc_kmeans(self.n_clusters, self.n_vecs)

    def fit_graph(self, G, normalized=True):
        self.G = G
        self.laplacian = self._get_laplacian(normalized)
        # self.eigenvals, self.eigenvecs = LA.eigh(self.laplacian)
        self.eigenvals, self.eigenvecs, self.eigenvalues_full, self.eigenvecors_full = \
            self.spectral_embedding(self.laplacian,
                                    n_components=self.n_vecs,
                                    drop_first=False,
                                    norm_laplacian=False)
        self.clustering = self._calc_kmeans(self.n_clusters, self.n_vecs)

    def spectral_embedding(self, adjacency, n_components=8, eigen_solver=None,
                           random_state=None, eigen_tol=0.0,
                           norm_laplacian=True, drop_first=True):
        """Project the sample on the first eigenvectors of the graph Laplacian.

        The adjacency matrix is used to compute a normalized graph Laplacian
        whose spectrum (especially the eigenvectors associated to the
        smallest eigenvalues) has an interpretation in terms of minimal
        number of cuts necessary to split the graph into comparably sized
        components.

        This embedding can also 'work' even if the ``adjacency`` variable is
        not strictly the adjacency matrix of a graph but more generally
        an affinity or similarity matrix between samples (for instance the
        heat kernel of a euclidean distance matrix or a k-NN matrix).

        However care must taken to always make the affinity matrix symmetric
        so that the eigenvector decomposition works as expected.

        Note : Laplacian Eigenmaps is the actual algorithm implemented here.

        Read more in the :ref:`User Guide <spectral_embedding>`.

        Parameters
        ----------
        adjacency : array-like or sparse matrix, shape: (n_samples, n_samples)
            The adjacency matrix of the graph to embed.

        n_components : integer, optional, default 8
            The dimension of the projection subspace.

        eigen_solver : {None, 'arpack', 'lobpcg', or 'amg'}, default None
            The eigenvalue decomposition strategy to use. AMG requires pyamg
            to be installed. It can be faster on very large, sparse problems,
            but may also lead to instabilities.

        random_state : int, RandomState instance or None, optional, default: None
            A pseudo random number generator used for the initialization of the
            lobpcg eigenvectors decomposition.  If int, random_state is the seed
            used by the random number generator; If RandomState instance,
            random_state is the random number generator; If None, the random number
            generator is the RandomState instance used by `np.random`. Used when
            ``solver`` == 'amg'.

        eigen_tol : float, optional, default=0.0
            Stopping criterion for eigendecomposition of the Laplacian matrix
            when using arpack eigen_solver.

        norm_laplacian : bool, optional, default=True
            If True, then compute normalized Laplacian.

        drop_first : bool, optional, default=True
            Whether to drop the first eigenvector. For spectral embedding, this
            should be True as the first eigenvector should be constant vector for
            connected graph, but for spectral clustering, this should be kept as
            False to retain the first eigenvector.

        Returns
        -------
        embedding : array, shape=(n_samples, n_components)
            The reduced samples.

        Notes
        -----
        Spectral Embedding (Laplacian Eigenmaps) is most useful when the graph
        has one connected component. If there graph has many components, the first
        few eigenvectors will simply uncover the connected components of the graph.

        References
        ----------
        * https://en.wikipedia.org/wiki/LOBPCG

        * Toward the Optimal Preconditioned Eigensolver: Locally Optimal
          Block Preconditioned Conjugate Gradient Method
          Andrew V. Knyazev
          https://doi.org/10.1137%2FS1064827500366124
        """
        # adjacency = check_symmetric(adjacency)

        try:
            from pyamg import smoothed_aggregation_solver
        except ImportError:
            if eigen_solver == "amg":
                raise ValueError("The eigen_solver was set to 'amg', but pyamg is "
                                 "not available.")

        if eigen_solver is None:
            eigen_solver = 'arpack'
        elif eigen_solver not in ('arpack', 'lobpcg', 'amg'):
            raise ValueError("Unknown value for eigen_solver: '%s'."
                             "Should be 'amg', 'arpack', or 'lobpcg'"
                             % eigen_solver)

        random_state = check_random_state(random_state)

        n_nodes = adjacency.shape[0]
        # Whether to drop the first eigenvector
        if drop_first:
            n_components = n_components + 1

        if not _graph_is_connected(adjacency):
            warnings.warn("Graph is not fully connected, spectral embedding"
                          " may not work as expected.")

        laplacian, dd = csgraph_laplacian(adjacency, normed=norm_laplacian,
                                          return_diag=True)
        if (eigen_solver == 'arpack' or eigen_solver != 'lobpcg' and
                (not sparse.isspmatrix(laplacian) or n_nodes < 5 * n_components)):
            # lobpcg used with eigen_solver='amg' has bugs for low number of nodes
            # for details see the source code in scipy:
            # https://github.com/scipy/scipy/blob/v0.11.0/scipy/sparse/linalg/eigen
            # /lobpcg/lobpcg.py#L237
            # or matlab:
            # https://www.mathworks.com/matlabcentral/fileexchange/48-lobpcg-m

            laplacian = _set_diag(laplacian, 1, norm_laplacian)

            # Here we'll use shift-invert mode for fast eigenvalues
            # (see https://docs.scipy.org/doc/scipy/reference/tutorial/arpack.html
            #  for a short explanation of what this means)
            # Because the normalized Laplacian has eigenvalues between 0 and 2,
            # I - L has eigenvalues between -1 and 1.  ARPACK is most efficient
            # when finding eigenvalues of largest magnitude (keyword which='LM')
            # and when these eigenvalues are very large compared to the rest.
            # For very large, very sparse graphs, I - L can have many, many
            # eigenvalues very near 1.0.  This leads to slow convergence.  So
            # instead, we'll use ARPACK's shift-invert mode, asking for the
            # eigenvalues near 1.0.  This effectively spreads-out the spectrum
            # near 1.0 and leads to much faster convergence: potentially an
            # orders-of-magnitude speedup over simply using keyword which='LA'
            # in standard mode.
            try:
                # We are computing the opposite of the laplacian inplace so as
                # to spare a memory allocation of a possibly very large array
                # laplacian *= -1
                v0 = random_state.uniform(-1, 1, laplacian.shape[0])
                lambdas, diffusion_map = eigsh(laplacian,
                                               k=n_components,
                                               sigma=1.0,
                                               which='LM',
                                               tol=eigen_tol,
                                               v0=v0)
                # lambdas = np.real(lambdas)
                # diffusion_map = np.real(diffusion_map)
                # print("Size diffusion map:", diffusion_map.shape)
                embedding = diffusion_map.T[n_components::-1]
                if norm_laplacian:
                    embedding = embedding / dd
            except RuntimeError:
                # When submatrices are exactly singular, an LU decomposition
                # in arpack fails. We fallback to lobpcg
                eigen_solver = "lobpcg"
                # Revert the laplacian to its opposite to have lobpcg work
                laplacian *= -1

        embedding = _deterministic_vector_sign_flip(embedding)
        # if drop_first:
        #     return lambdas, embedding[1:n_components].T
        # else:
        #     return lambdas, embedding[:n_components].T
        return lambdas[:n_components], embedding[:n_components].T, lambdas, embedding.T


def get_leave_nodes(root):
    leaves = []
    collect_leave_nodes(root, leaves)
    leaves = sorted(leaves)
    return leaves


def collect_leave_nodes(root, leaves):
    if root.point_id is not None:
        leaves.append(root.point_id)
    else:
        collect_leave_nodes(root.left_tree, leaves)
        collect_leave_nodes(root.right_tree, leaves)


def run_spectral_clustering(root, sim_mx, dist_mx, *, eps=0, it=0, min_pts, n_clusters=2, type_="sc"):
    if type_ == "eps" and eps == 0:
        raise AssertionError("Please define an epsilon range")
    if type_ == "it" and it == 0:
        raise AssertionError("Please define number of iterations (=no. of lambdas to consider)")
    if type_ == "sc" and n_clusters < 1:
        raise AssertionError("Please define a positive nubmer of clusters for spectral analysis")

    sc_ = SpectralClustering_own(n_clusters=2, n_vecs=2)
    points = np.arange(dist_mx.shape[0])
    clustering = {k: -1 for k in points}

    if type_ == "eps":
        clustering = exec_eps(sc_, root, sim_mx, sim_mx, dist_mx, dist_mx, eps, points, points, clustering, min_pts)
        clustering = reindex(clustering)
        clustering = np.array(list(clustering.values()))
    elif type_ == "it":
        clustering = exec_it(sc_, root, sim_mx, sim_mx, dist_mx, dist_mx, it, points, points, clustering, min_pts)
        clustering = reindex(clustering)
        clustering = np.array(list(clustering.values()))
    elif type_ == "sc":
        sc_, clustering = exec_sc(sc_, sim_mx, n_clusters=n_clusters)

    return sc_, clustering


def reindex(clustering):
    new_clustering = {k: -1 for k in clustering.keys()}
    new_idx = 0
    for value in np.unique(list(clustering.values())):
        if value == -1:
            continue
        for p, c_label in clustering.items():
            if value == c_label:
                new_clustering[p] = new_idx
        new_idx += 1
    return new_clustering


def exec_sc(sc_, sim_mx, n_clusters=8):
    spec = SpectralClustering_own(n_clusters=n_clusters)  # , n_vecs=sim_rescaled.shape[0]-1)
    spec.fit(sim_mx)
    return spec, spec.clustering.labels_


def exec_it(sc_, root, original_sim_mx, sim_mx, original_dist_mx, dist_mx,
            nLambdas, original_point_ids, point_ids, clustering, min_pts):
    tree_list = [root]
    while nLambdas > 0 and len(tree_list) != 0:
        tree_list = sorted(tree_list, key=lambda t: t.dist, reverse=True)
        c_tree = tree_list.pop(0)
        leaves = get_leave_nodes(c_tree)
        c_sim_mx = sim_mx[leaves][:, leaves]
        sc_.fit(c_sim_mx)
        max_cluster_label = max(clustering.values()) + 1
        counter = Counter(sc_.clustering.labels_)
        for idx, p in enumerate(leaves):
            # if counter[sc_.clustering.labels_[idx]] < min_samples: ### <-- original code
            if counter[sc_.clustering.labels_[idx]] < min_pts: ### <-- Andrew change
                clustering[p] = -1
            else:
                clustering[p] = max_cluster_label + sc_.clustering.labels_[idx]

        tree_list.append(c_tree.left_tree)
        tree_list.append(c_tree.right_tree)
        nLambdas -= 1
    return clustering


def label_as_noise(clustering, nodes):
    for entry in nodes:
        clustering[entry] = -1


def exec_eps(sc_, root, original_sim_mx, sim_mx, original_dist_mx, dist_mx,
             eps, original_point_ids, point_ids, clustering, min_pts):
    if (root.dist > eps) and root.point_id is None:
        sc_.fit(sim_mx)
        max_cluster_label = max(clustering.values()) + 1
        for idx, p in enumerate(point_ids):
            clustering[p] = max_cluster_label + sc_.clustering.labels_[idx]

        left_leaves = get_leave_nodes(root.left_tree)
        right_leaves = get_leave_nodes(root.right_tree)

        if len(left_leaves) < min_samples:
            label_as_noise(clustering, left_leaves)
        else:
            exec_eps(sc_, root.left_tree,
                     original_sim_mx, original_sim_mx[left_leaves][:, left_leaves],
                     original_dist_mx, original_dist_mx[left_leaves][:, left_leaves],
                     eps, original_point_ids, left_leaves, clustering, min_pts)

        if len(right_leaves) < min_samples:
            label_as_noise(clustering, right_leaves)
        else:
            exec_eps(sc_, root.right_tree,
                     original_sim_mx, original_sim_mx[right_leaves][:, right_leaves],
                     original_dist_mx, original_dist_mx[right_leaves, right_leaves],
                     eps, original_point_ids, right_leaves, clustering, min_pts)
    return clustering


def get_sim_mx(dsnenns):
    e = 0  # 1e-15

    # get precomputed distances
    dist_dsne = dsnenns['_all_dists']
    dist_e = dist_dsne + e
    # normalized
    norm = dist_dsne / np.linalg.norm(dist_dsne)
    # get similiarity scores
    sim = 1 - norm
    # scale similarity scores to be in range ]0, 1]
    sim_rescaled = (1 - e) * (sim - np.min(sim)) / (np.max(sim) - np.min(sim)) + e

    # sim_frac = 1 / norm
    # rescaled = np.max(dist_dsne) - dist_dsne

    return sim


def plot_one_it(sc_labels):
    fig = plt.figure(figsize=(12, 12))
    ax1 = fig.add_subplot(2, 2, 1)  # row, column, figure number
    ax2 = fig.add_subplot(2, 2, 2)

    ax1.scatter(X[:, 0], X[:, 1], c=y_true, s=25, cmap='viridis')
    ax1.set_title("Raw dataset")
    ax1.set_xlabel("$x_1$")
    ax1.set_ylabel("$x_2$")

    if True:
        for idx, entry in enumerate(X):
            ax1.annotate(idx,  # this is the text
                         entry,  # these are the coordinates to position the label
                         textcoords="offset points",  # how to position the text
                         xytext=(0, 10),  # distance from text to points (x,y)
                         ha='center')  # horizontal

    ax2.scatter(X[:, 0], X[:, 1], c=sc_labels, s=10, cmap="viridis")
    ax2.scatter(X[:, 0][sc_labels == -1], X[:, 1][sc_labels == -1], s=100, c="red")

    if False:
        for idx, entry in enumerate(X[sc_labels != 0]):
            ax2.annotate(sc_labels[sc_labels != 0][idx],
                         # idx, # this is the text
                         entry,  # these are the coordinates to position the label
                         textcoords="offset points",  # how to position the text
                         xytext=(0, 10),  # distance from text to points (x,y)
                         ha='center')  # horizontal
    ax2.set_title("Clustered data")
    ax2.set_xlabel("$x_1$")
    ax2.set_ylabel("$x_2$")

    plt.show()
    # print(spec.clustering.labels_)


def get_lambdas (root, eps_dist):
    return traversal(root, eps_dist)


def traversal(root, eps_dist):
    sum_ = 0
    if root.dist > eps_dist:
        sum_ += 1
        sum_ += traversal(root.left_tree, eps_dist)
        sum_ += traversal(root.right_tree, eps_dist)
    else:
        return 0
    return sum_


if __name__ == '__main__':
    # Generate data
    noise = 0.6
    n = 75
    d = 2
    n_clusters = 2
    X, y_true = make_blobs(n_samples=n, n_features=d, centers=n_clusters, cluster_std=noise, shuffle=False,
                           random_state=666)

    # Get distance mx and tree representation
    dsnenns = get_nearest_neighbors(X, 2, min_points=2)
    dsnedist = np.reshape(dsnenns['_all_dists'], -1)
    dist_dsne = dsnenns['_all_dists']
    root_, dc_dist = make_tree(X, y_true, min_points=2, n_neighbors=2)

    # Execute DBSCAN
    eps = 0.408
    min_samples = 2
    dbscan = DBSCAN(eps=eps, min_pts=min_samples, cluster_type='standard')
    dbscan.fit(X)
    dbscan.plot2D(y_true)

    # Execute Spectral Clustering
    no_lambdas = get_lambdas(root_, eps)
    print("number of lambdas: ", no_lambdas)
    sim = get_sim_mx(dsnenns)
    sc_, sc_labels = run_spectral_clustering(root_, sim, dist_dsne, eps=eps, it=no_lambdas, min_pts=2, n_clusters=8, type_="it")
    plot_one_it(sc_labels)
