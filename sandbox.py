import numpy as np
import numba
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll

from experiment_utils.get_data import get_dataset
from GDR import GradientDR

class Component:
    def __init__(self, nodes, comp_id):
        self.nodes = set(nodes)
        self.comp_id = comp_id

def merge_components(c_i, c_j):
    merged_list = c_i.nodes.union(c_j.nodes)
    return Component(merged_list, c_i.comp_id)

def get_nearest_neighbors(points, n_neighbors, **kwargs):
    """
    We define the distance from x_i to x_j as min(max(P(x_i, x_j))), where 
        - P(x_i, x_j) is any path from x_i to x_j
        - max(P(x_i, x_j)) is the largest edge weight in the path
        - min(max(P(x_i, x_j))) is the smallest largest edge weight
    """
    @numba.njit(fastmath=True, parallel=True)
    def get_dist_matrix(points, D, dim, num_points):
        for i in numba.prange(num_points):
            x = points[i]
            for j in range(i+1, num_points):
                y = points[j]
                dist = 0
                for d in range(dim):
                    dist += (x[d] - y[d]) ** 2
                D[i, j] = dist
                D[j, i] = dist
        return D

    num_points = int(points.shape[0])
    dim = int(points.shape[1])
    density_connections = np.zeros([num_points, num_points])
    D = np.zeros([num_points, num_points])
    D = get_dist_matrix(points, D, dim, num_points)

    flat_D = np.reshape(D, [num_points * num_points])
    argsort_inds = np.argsort(flat_D)

    num_added = 0
    component_dict = {i: Component([i], i) for i in range(num_points)}
    neighbor_dists = [[] for i in range(num_points)]
    neighbor_inds = [[] for i in range(num_points)]
    max_comp_size = 1
    for index in tqdm(argsort_inds):
        i = int(index / num_points)
        j = index % num_points
        if component_dict[i].comp_id != component_dict[j].comp_id:
            epsilon = D[i, j]
            for node_i in component_dict[i].nodes:
                for node_j in component_dict[j].nodes:
                    density_connections[node_i, node_j] = epsilon
                    density_connections[node_j, node_i] = epsilon
                    # If we have space for more neighbors
                    if len(neighbor_dists[node_i]) < n_neighbors:
                        neighbor_dists[node_i].append(epsilon)
                        neighbor_inds[node_i].append(node_j)
                    if len(neighbor_dists[node_j]) < n_neighbors:
                        neighbor_dists[node_j].append(epsilon)
                        neighbor_inds[node_j].append(node_i)

            merged_component = merge_components(component_dict[i], component_dict[j])
            for node in merged_component.nodes:
                component_dict[node] = merged_component
            size_of_component = len(component_dict[i].nodes)
            if size_of_component > max_comp_size:
                max_comp_size = size_of_component
        if max_comp_size == num_points:
            break

    return {
        '_knn_indices': np.array(neighbor_inds),
        '_knn_dists': np.array(neighbor_dists),
        '_all_dists': density_connections
    }

def uniform_line_example(num_points=50):
    # Points are [1, 2, 3, 4, ...]
    points = np.expand_dims(np.arange(num_points), -1)
    labels = np.arange(num_points)
    dists = umap_plots(points, labels, s=10)
    histogram(dists)

def linear_growth_example(num_points=50):
    points = np.zeros([num_points])
    labels = np.arange(num_points)
    # Points are [1, 3, 6, 10, 15, 21, ...]
    # This is just i * (i + 1) / 2
    for i in range(num_points):
        points[i] = (i + 1) * i / 2
    points = np.expand_dims(points, -1)
    dists = umap_plots(points, labels, s=10)
    histogram(dists)

def swiss_roll_example(num_points=5000):
    points, _ = make_swiss_roll(n_samples=num_points, noise=0.01)
    labels = np.arange(num_points)
    dists = umap_plots(points, labels, s=1)
    histogram(dists)

def histogram(dists, labels=None):
    if labels is None:
        plt.hist(dists, bins=50)
        plt.savefig('histogram.pdf')
        plt.close()
    else:
        label_agreement = np.expand_dims(labels, 0) == np.expand_dims(labels, 1)
        label_agreement = np.reshape(label_agreement, -1)
        intra_class = dists[label_agreement == 1]
        inter_class = dists[label_agreement == 0]
        min_length = min(len(intra_class), len(inter_class))
        dists_by_class = np.stack([intra_class[:min_length], inter_class[:min_length]], axis=-1)
        plt.hist(dists_by_class, bins=50)
        plt.savefig('histogram.pdf')
        plt.close()

def umap_plots(points, labels, s=1):
    dr = GradientDR(nn_alg=get_nearest_neighbors)
    projections = dr.fit_transform(points)
    density_dists = dr._all_dists
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1) # row 1, col 2 index 1
    plt.scatter(projections[:, 0], projections[:, 1], c=labels, s=s, alpha=0.8)
    plt.title("Using density connected metric")

    # Using the ambient Euclidean metric
    dr = GradientDR()
    projections = dr.fit_transform(points)
    plt.subplot(1, 2, 2) # index 2
    plt.scatter(projections[:, 0], projections[:, 1], c=labels, s=s, alpha=0.8)
    plt.title("Using traditional Euclidean distance")
    plt.savefig('embedding_comparison.pdf')
    plt.close()

    return np.reshape(density_dists, -1)

if __name__ == '__main__':
    # Basic line-based examples
    # uniform_line_example()
    # linear_growth_example()
    # swiss_roll_example()

    points, labels = get_dataset('coil', num_classes=20, points_per_class=72)
    # points, labels = get_dataset('mnist', num_classes=2, points_per_class=500)

    dists = umap_plots(points, labels, s=1)

    # Histogram with labels only works for 2 classes
    histogram(dists, labels=labels)
