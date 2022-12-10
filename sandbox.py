import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll, make_moons
from sklearn.manifold import MDS
from sklearn.cluster import DBSCAN, SpectralClustering
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.decomposition import PCA
import networkx as nx

from experiment_utils.get_data import get_dataset, make_circles
from distance_metric import get_nearest_neighbors
from density_preserving_embeddings import make_dc_embedding, make_tree, plot_embedding
from cluster_tree import dc_kmeans
from GDR import GradientDR

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
        '--norm',
        type=int,
        default=2,
        help='norm to raise distance to when clustering'
    )
    parser.add_argument(
        '--plot-tree',
        action='store_true',
        help='If present, will make a plot of the tree'
    )
    args = parser.parse_args()
    if args.norm == -1:
        args.norm = np.inf

    # Trivial examples
    # uniform_line_example()
    # linear_growth_example()
    # swiss_roll_example()
    # circles_example()

    # points, labels = get_dataset('coil', class_list=np.arange(1, 10), points_per_class=12)
    points, labels = make_circles(
        n_samples=1000,
        noise=0.03,
        radii=[0.2, 0.5, 1.0],
        thicknesses=[0.1, 0.1, 0.1]
    )
    # points, labels = make_moons(n_samples=400, noise=0.15)
    # points, labels = get_dataset('mnist', num_classes=10, points_per_class=50)

    root, dc_dists = make_tree(points, labels, min_points=args.min_pts, make_image=True, n_neighbors=args.n_neighbors)
    pred_labels, epsilons = dc_kmeans(root, num_points=len(labels), k=args.k, min_points=args.min_pts, norm=args.norm)
    embed_points = make_dc_embedding(root, dc_dists, min_points=args.min_pts, n_neighbors=args.n_neighbors)

    # spectral = SpectralClustering(n_clusters=args.k).fit(points)
    dbscan_orig = DBSCAN(eps=np.mean(epsilons), min_samples=args.min_pts).fit(points)
    dbscan_orig = DBSCAN(eps=np.mean(epsilons), min_samples=args.min_pts).fit(points)
    dbscan_embed = DBSCAN(eps=np.mean(epsilons), min_samples=2).fit(embed_points)

    print('k-Means cut off epsilons:', epsilons)
    print('NMI truth vs. dbscan:', nmi(labels, dbscan_orig.labels_))
    print('NMI truth vs. us:', nmi(labels, pred_labels))
    # print('NMI spectral vs. us:', nmi(spectral.labels_, pred_labels))
    print('NMI dbscan vs. us:', nmi(dbscan_orig.labels_, pred_labels))
    print('NMI dbscan vs. dbscan on embedding:', nmi(dbscan_orig.labels_, dbscan_embed.labels_))

    # If running on high-dimensional data, want to use the 'embed_points' variable for the plotting
    plot_embedding(
        points,
        [labels, pred_labels, dbscan_orig.labels_, dbscan_embed.labels_],
        ['truth', 'us', 'dbscan_original_data', 'dbscan_embedding']
    )


    # dists = embedding_plots(points, labels)
    # histogram(dists, labels=labels)
