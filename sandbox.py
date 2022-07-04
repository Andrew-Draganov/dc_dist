import numpy as np
from distance_metric import distance_metric
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from tqdm.auto import tqdm
from experiment_utils.get_data import get_dataset

from gidr_dun.gidr_dun_ import DensityDR
from umap import UMAP

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
        plt.show()
        plt.close()
    else:
        label_agreement = np.expand_dims(labels, 0) == np.expand_dims(labels, 1)
        label_agreement = np.reshape(label_agreement, -1)
        intra_class = dists[label_agreement == 1]
        inter_class = dists[label_agreement == 0]
        min_length = min(len(intra_class), len(inter_class))
        dists_by_class = np.stack([intra_class[:min_length], inter_class[:min_length]], axis=-1)
        plt.hist(dists_by_class, bins=50)
        plt.show()
        plt.close()

def umap_plots(points, labels, s=1):
    dr = DensityDR()
    projections = dr.fit_transform(points)
    density_dists = dr._all_dists
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1) # row 1, col 2 index 1
    plt.scatter(projections[:, 0], projections[:, 1], c=labels, s=s, alpha=0.8)
    plt.title("Using density connected metric")

    # Using the ambient Euclidean metric
    dr = UMAP()
    projections = dr.fit_transform(points)
    plt.subplot(1, 2, 2) # index 2
    plt.scatter(projections[:, 0], projections[:, 1], c=labels, s=s, alpha=0.8)
    plt.title("Using traditional Euclidean distance")
    plt.show()
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
