import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll, make_moons
from sklearn.manifold import MDS
from sklearn.cluster import DBSCAN, SpectralClustering
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.decomposition import PCA
import networkx as nx
import hdbscan

from experiment_utils.get_data import get_dataset, make_circles
from distance_metric import get_nearest_neighbors
from density_preserving_embeddings import make_dc_embedding
from density_tree import make_tree
from tree_plotting import plot_embedding
from cluster_tree import dc_clustering
from GDR import GradientDR

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
        type=float,
        default=2,
        help='Norm to raise distance to when clustering. Negative values are interpreted as inf. norm'
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

    # points, labels = get_dataset('synth', num_classes=6, points_per_class=72)
    # points, labels = get_dataset('coil', class_list=np.arange(1, 20), points_per_class=36)
    # points, labels = make_circles(
    #     n_samples=500,
    #     noise=0.01,
    #     radii=[0.5, 1.0],
    #     thicknesses=[0.1, 0.1]
    # )
    points, labels = make_moons(n_samples=400, noise=0.1)
    # points, labels = get_dataset('mnist', num_classes=10, points_per_class=50)

    # TODO
    #  - Figure out issue for k-center <=> dbscan on min_pts > 1

    root, dc_dists = make_tree(
        points,
        labels,
        min_points=args.min_pts,
        make_image=args.plot_tree,
        n_neighbors=args.n_neighbors
    )

    pred_labels, centers, epsilons = dc_clustering(
        root,
        num_points=len(labels),
        k=args.k,
        min_points=args.min_pts,
        norm=args.norm
    )

    embed_points = make_dc_embedding(
        root,
        dc_dists,
        min_points=args.min_pts,
        n_neighbors=args.n_neighbors
    )

    # FIXME -- do I do plus or minus here?
    # In either case, change the eps by a tiny amount so that we don't cut below that
    # distance in one implementation and above it in another implementation
    eps = np.max(epsilons[np.where(epsilons > 0)]) + 1e-8

    # spectral = SpectralClustering(n_clusters=args.k).fit(points)
    dbscan_orig = DBSCAN(eps=eps, min_samples=args.min_pts).fit(points)
    dbscan_embed = DBSCAN(eps=eps, min_samples=args.min_pts).fit(embed_points)
    # dbscan_orig = hdbscan.HDBSCAN(approx_min_span_tree=False, min_samples=1).fit(points)
    # dbscan_embed = hdbscan.HDBSCAN(approx_min_span_tree=False, min_samples=1).fit(embed_points)

    dbscan_core_pt_inds = np.where(dbscan_embed.labels_ > -1)
    dc_core_pt_inds = np.where(np.logical_and(pred_labels > -1, dbscan_orig.labels_ > -1))
    dc_core_pt_inds_embed = np.where(np.logical_and(pred_labels > -1, dbscan_embed.labels_ > -1))
    print('k-Means cut off epsilons:', epsilons)
    print('NMI truth vs. dbscan:', nmi(labels, dbscan_orig.labels_))
    print('NMI truth vs. us:', nmi(labels, pred_labels))
    # print('NMI spectral vs. us:', nmi(spectral.labels_, pred_labels))
    print('NMI dbscan vs. us:', nmi(dbscan_orig.labels_[dc_core_pt_inds], pred_labels[dc_core_pt_inds]))
    print('NMI dbscan vs. dbscan on embedding:', nmi(dbscan_orig.labels_[dbscan_core_pt_inds], dbscan_embed.labels_[dbscan_core_pt_inds]))
    print('NMI us vs. dbscan on embedding:', nmi(pred_labels[dc_core_pt_inds_embed], dbscan_embed.labels_[dc_core_pt_inds_embed]))

    plot_points = points
    if points.shape[1] > 2:
        plot_points = embed_points
    plot_embedding(
        plot_points,
        [labels, pred_labels, dbscan_orig.labels_, dbscan_embed.labels_],
        ['truth', 'us', 'dbscan_original_data', 'dbscan_embedding'],
        centers=centers
    )
