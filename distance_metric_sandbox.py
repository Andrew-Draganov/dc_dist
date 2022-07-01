import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from tqdm.auto import tqdm
from experiment_utils.get_data import get_dataset

from optimizers.pca_opt import PCAOptimizer
from optimizers.umap_opt import UMAPOptimizer

def distance_metric(points, stop_criteria=-1):
    """
    We define the distance from x_i to x_j as min(max(P(x_i, x_j))), where 
        - P(x_i, x_j) is any path from x_i to x_j
        - max(P(x_i, x_j)) is the largest edge weight in the path
        - min(max(P(x_i, x_j))) is the smallest largest edge weight

    We do this through the following pseudocode:
    -------------------------------------------
        Start with:
            - the pairwise Euclidean distance matrix D
            - an empty adjacency matrix A
        Returns:
            - density-distance matrix C

        for i < n^2:
            epsilon <- i-th smallest distance in D(X)
            Put epsilon into A in the same position as it is in D(X)
            for each pair of points without a density-connected distance x_i, x_j in X:
                Find the shortest path in A from x_i to x_j
                If such a path exists, the density-connected distance from x_i to x_j is epsilon
    """
    num_points = int(points.shape[0])
    density_connections = np.zeros([num_points, num_points])
    A = np.zeros([num_points, num_points])
    D = np.zeros([num_points, num_points])

    for i in range(num_points):
        x = points[i]
        for j in range(i+1, num_points):
            y = points[j]
            dist = np.sqrt(np.sum(np.square(x - y)))
            D[i, j] = dist
            D[j, i] = dist

    flat_D = np.reshape(D, [num_points * num_points])
    argsort_inds = np.argsort(flat_D)

    # FIXME -- this is slow because the same distance gets handled multiple times.
    #          Should instead do one iteration for each unique pairwise distance
    max_connections = (num_points ** 2 - num_points) / 2
    if stop_criteria < 0 or stop_criteria > max_connections:
        stop_criteria = max_connections
    total_connections = 0
    for step in tqdm(range(0, num_points * num_points, 2)):
        i_index = int(argsort_inds[step] / num_points)
        j_index = argsort_inds[step] % num_points
        epsilon = D[i_index, j_index]
        A[i_index, j_index] = epsilon

        graph = nx.from_numpy_array(A)
        paths = nx.shortest_path(graph)
        for i in range(num_points):
            for j in range(i+1, num_points):
                if density_connections[i, j] == 0:
                    has_zeros = True
                    if i in paths:
                        if j in paths[i]:
                            density_connections[i, j] = epsilon
                            density_connections[j, i] = epsilon
                            total_connections += 1
        if total_connections >= stop_criteria:
            break

    return density_connections

def subsample_points(points, labels, num_classes, points_per_class, class_list=[]):
    if not class_list:
        all_classes = np.unique(labels)
        class_list = np.random.choice(all_classes, num_classes, replace=False)

    per_class_samples = [np.where(labels == sampled_class)[0] for sampled_class in class_list]
    min_per_class = min([len(s) for s in per_class_samples])
    per_class_samples = [s[:min_per_class] for s in per_class_samples]
    sample_indices = np.squeeze(np.stack([per_class_samples], axis=-1))
    total_points_per_class = int(sample_indices.shape[-1])
    if points_per_class < total_points_per_class:
        stride_rate = float(total_points_per_class) / points_per_class
        class_subsample_indices = np.arange(0, total_points_per_class, step=stride_rate).astype(np.int32)
        sample_indices = sample_indices[:, class_subsample_indices]

    sample_indices = np.reshape(sample_indices, -1)
    points = points[sample_indices]
    labels = labels[sample_indices]
    return points, labels

def get_dists(dataset, class_list=[], num_classes=2, points_per_class=72):
    points, labels = get_dataset(dataset, num_points=-1)
    points, labels = subsample_points(
        points,
        labels,
        class_list=class_list,
        num_classes=num_classes,
        points_per_class=points_per_class
    )
    pairwise_dists = distance_metric(points)
    pairwise_dists = np.reshape(pairwise_dists, [-1])
    return points, labels, pairwise_dists

def uniform_line_example(num_points=50):
    # Points are [1, 2, 3, 4, ...]
    points = np.expand_dims(np.arange(num_points), -1)
    labels = np.arange(num_points)
    pairwise_dists = distance_metric(points)
    pairwise_dists = np.reshape(pairwise_dists, [-1])
    plt.hist(pairwise_dists)
    plt.show()
    plt.close()

def linear_growth_example(num_points=50):
    points = np.zeros([num_points])
    labels = np.arange(num_points)
    # Points are [1, 3, 6, 10, 15, 21, ...]
    # This is just i * (i + 1) / 2
    for i in range(num_points):
        points[i] = (i + 1) * i / 2
    points = np.expand_dims(points, -1)
    pairwise_dists = distance_metric(points)
    pairwise_dists = np.reshape(pairwise_dists, -1)
    plt.hist(pairwise_dists)
    plt.show()
    plt.close()

def swiss_roll_example(num_points=250):
    points, _ = make_swiss_roll(n_samples=num_points, noise=0.01)
    labels = np.arange(num_points)
    pairwise_dists = distance_metric(points)
    pairwise_dists = np.reshape(pairwise_dists, -1)
    plt.hist(pairwise_dists)
    plt.show()
    plt.close()
    umap_plots(points, labels, pairwise_dists)

def histogram(dists, labels=None):
    if labels is None:
        plt.hist(dists, bins=50)
        plt.show()
        plt.close()

    else:
        label_agreement = np.expand_dims(labels, 0) == np.expand_dims(labels, 1)
        label_agreement = np.reshape(label_agreement, -1)
        intra_class = dists[label_agreement == 1]
        inter_class = dists[label_agreement == 0]
        dists_by_class = np.stack([intra_class, inter_class], axis=-1)
        plt.hist(dists_by_class, bins=50)
        plt.show()
        plt.close()

def umap_plots(points, labels, dists):
    optimizer = UMAPOptimizer(
        x=points,
        labels=labels,
        pairwise_x_mat=np.reshape(dists, [len(labels), len(labels)]),
        show_intermediate=False,
        momentum=0.5,
        min_p_value=0.0
    )
    grad_solution = optimizer.optimize()
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1) # row 1, col 2 index 1
    plt.scatter(grad_solution[:, 0], grad_solution[:, 1], c=labels)
    plt.title("Using density connected metric")

    # Using the ambient Euclidean metric
    optimizer = UMAPOptimizer(
        x=points,
        labels=labels,
        show_intermediate=False,
        momentum=0.5,
        min_p_value=0.0
    )
    grad_solution = optimizer.optimize()
    plt.subplot(1, 2, 2) # index 2
    plt.scatter(grad_solution[:, 0], grad_solution[:, 1], c=labels)
    plt.title("Using traditional Euclidean distance")
    plt.show()
    plt.close()

if __name__ == '__main__':
    # Basic line-based examples
    # uniform_line_example()
    # linear_growth_example()
    swiss_roll_example()

    # points, labels, dists = get_dists('coil', num_classes=2, points_per_class=72)
    # points, labels, dists = get_dists('mnist', class_list=[7, 0], points_per_class=50)
    # umap_plots(points, labels, dists)
    # histogram(dists, labels=labels)
