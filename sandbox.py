import numpy as np
from distance_metric import distance_metric
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from tqdm.auto import tqdm
from experiment_utils.get_data import get_dataset

from optimizers.pca_opt import PCAOptimizer
from optimizers.umap_opt import UMAPOptimizer

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
    histogram(pairwise_dists)
    umap_plots(points, labels, pairwise_dists)

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
    histogram(pairwise_dists)
    umap_plots(points, labels, pairwise_dists)

def swiss_roll_example(num_points=250):
    points, _ = make_swiss_roll(n_samples=num_points, noise=0.01)
    labels = np.arange(num_points)
    pairwise_dists = distance_metric(points)
    pairwise_dists = np.reshape(pairwise_dists, -1)
    histogram(pairwise_dists)
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
        n_epochs=100,
        lr=0.5,
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
    # swiss_roll_example()

    points, labels, dists = get_dists('coil', num_classes=5, points_per_class=36)
    # points, labels, dists = get_dists('mnist', class_list=[7, 0], points_per_class=150)
    umap_plots(points, labels, dists)
    histogram(dists, labels=labels)
