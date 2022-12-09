import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll, make_moons
from sklearn.manifold import MDS
from sklearn.cluster import DBSCAN
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.decomposition import PCA
import networkx as nx

from experiment_utils.get_data import get_dataset, make_circles
from distance_metric import get_nearest_neighbors
from density_preserving_embeddings import make_dc_embedding, make_tree, plot_embedding
from cluster_tree import dc_kmeans
from GDR import GradientDR

def uniform_line_example(num_points=50):
    # Points are [1, 2, 3, 4, ...]
    points = np.expand_dims(np.arange(num_points), -1)
    labels = np.arange(num_points)
    dists = embedding_plots(points, labels, s=10)
    histogram(dists)

def linear_growth_example(num_points=50):
    points = np.zeros([num_points])
    labels = np.arange(num_points)
    # Points are [1, 3, 6, 10, 15, 21, ...]
    # This is just i * (i + 1) / 2
    for i in range(num_points):
        points[i] = (i + 1) * i / 2
    points = np.expand_dims(points, -1)
    dists = embedding_plots(points, labels, s=10)
    histogram(dists)

def swiss_roll_example(num_points=5000):
    points, _ = make_swiss_roll(n_samples=num_points, noise=0.01)
    labels = np.arange(num_points)
    dists = embedding_plots(points, labels, s=1)
    histogram(dists)

def circles_example(num_points=1000):
    points, labels = make_circles(n_samples=num_points, noise=0.)
    dists = embedding_plots(points, labels, s=1)
    histogram(dists)

def histogram(dists, labels=None):
    if labels is None:
        plt.hist(dists, bins=50)
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

def embedding_plots(points, labels, s=5):
    plt.rcParams.update({'font.size': 10})
    fig, axes = plt.subplots(2, 2)
    fig.set_figheight(8)
    fig.set_figwidth(12)
    plt.setp(axes, xticks=[], yticks=[])

    # Run UMAP with our density-connected distance metric
    dr = GradientDR(nn_alg=get_nearest_neighbors, random_init=True)
    projections = dr.fit_transform(points)
    density_dists = dr._all_dists.copy()
    axes[0, 0].scatter(projections[:, 0], projections[:, 1], c=labels, s=s, alpha=0.8)
    axes[0, 0].set_title("Using density connected metric UMAP")

    # Run TSNE with our density-connected distance metric
    # TSNE is just the GDR algorithm with normalized set to True :)
    dr = GradientDR(nn_alg=get_nearest_neighbors, normalized=True, random_init=True)
    projections = dr.fit_transform(points)
    axes[0, 1].scatter(projections[:, 0], projections[:, 1], c=labels, s=s, alpha=0.8)
    axes[0, 1].set_title("Using density connected metric TSNE")

    # Do multidimensional scaling on the density-connected distance matrix
    #   This is basically PCA
    # Warning: this is slow for large datasets
    dr = MDS()
    projections = dr.fit_transform(density_dists)
    axes[1, 0].scatter(projections[:, 0], projections[:, 1], c=labels, s=s, alpha=0.8)
    axes[1, 0].set_title("Using MDS on density-connected distances")

    # Density-connected preserving embedding
    points, labels = make_dc_embedding(density_dists, labels, embed_dim=2)
    axes[1, 1].scatter(points[:, 0], points[:, 1], c=labels, s=s, alpha=0.8)
    axes[1, 1].set_title("Density-Connected Preserving Embedding")

    # plt.savefig('embedding_comparison.pdf')
    plt.show()
    plt.close()
    return np.reshape(density_dists, -1)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--min-pts',
        type=int,
        default=1,
        help='Min points parameter to use for density-connectedness'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=4,
        help='Number of clusters for density-connected k-means'
    )
    parser.add_argument(
        '--n-neighbors',
        type=int,
        default=15,
        help='Dummy variable for compatibility with UMAP/tSNE distance calculation'
    )
    parser.add_argument(
        '--power',
        type=int,
        default=2,
        help='Power to raise distance to when clustering'
    )
    args = parser.parse_args()


    # Trivial examples
    # uniform_line_example()
    # linear_growth_example()
    # swiss_roll_example()
    # circles_example()

    # points, labels = get_dataset('coil', class_list=np.arange(11, 20), points_per_class=36)
    # points, labels = make_circles(n_samples=400, noise=0.03, radii=[0.2, 0.5, 1.0], thicknesses=[0.1, 0.1, 0.1])
    points, labels = make_moons(n_samples=400, noise=0.15)
    # make_tree(points, labels, min_points=args.min_pts)
    root, dc_dists = make_tree(points, labels, min_points=args.min_pts, make_image=False, n_neighbors=args.n_neighbors)
    pred_labels, epsilons = dc_kmeans(root, num_points=len(labels), k=args.k, min_points=args.min_pts, power=args.power)
    embed_points = make_dc_embedding(root, dc_dists, min_points=args.min_pts, n_neighbors=args.n_neighbors)
    print('k-Means cut off epsilons:', epsilons)
    dbscan_orig = DBSCAN(eps=np.mean(epsilons), min_samples=args.min_pts).fit(points)
    dbscan_embed = DBSCAN(eps=np.mean(epsilons), min_samples=2).fit(embed_points)
    print('NMI truth vs. dbscan:', nmi(labels, dbscan_orig.labels_))
    print('NMI truth vs. us:', nmi(labels, pred_labels))
    print('NMI dbscan vs. us:', nmi(dbscan_orig.labels_, pred_labels))
    print('NMI dbscan vs. dbscan on embedding:', nmi(dbscan_orig.labels_, dbscan_embed.labels_))
    
    plot_embedding(
        points,
        [labels, pred_labels, dbscan_orig.labels_, dbscan_embed.labels_],
        ['truth', 'us', 'dbscan_original_data', 'dbscan_embedding']
    )

    # points, labels = get_dataset('mnist', num_classes=10, points_per_class=50)

    # dists = embedding_plots(points, labels)
    # histogram(dists, labels=labels)
