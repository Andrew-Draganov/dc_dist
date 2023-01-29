from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import numpy as np

from DBSCAN import DBSCAN
from distance_metric import get_nearest_neighbors
from density_tree import make_tree
from cluster_tree import dc_clustering
from SpectralClustering import get_lambdas, get_sim_mx, run_spectral_clustering

def compare_clusterings(points, labels):
    k = 4
    min_pts_list = [1, 3, 5]
    fig, axes = plt.subplots(len(min_pts_list), 4)
    plt.rcParams.update({'text.usetex': True, 'font.size': 22})
    ### FIXME -- remaining work:
    #   - Figure out issue with kcenter giving too many noise points
    #   - Add kcenter without noise
    #   - Add regular spectral clustering?
    for i, min_pts in enumerate(min_pts_list):
        root, dc_dists = make_tree(
            points,
            labels,
            min_points=min_pts,
        )

        # Calculate kcenter clustering
        kcenter_labels, centers, epsilons = dc_clustering(
            root,
            num_points=len(labels),
            k=k,
            min_points=min_pts,
            norm=np.inf
        )

        # Get epsilon given by k-center clustering
        eps = np.max(epsilons[np.where(epsilons > 0)]) + 1e-8

        # Calculate spectral clustering
        dsnenns = get_nearest_neighbors(points, 15, min_points=min_pts)
        dsnedist = np.reshape(dsnenns['_all_dists'], -1)
        dist_dsne = dsnenns['_all_dists']
        no_lambdas = get_lambdas(root, eps)
        sim = get_sim_mx(dsnenns)
        sc_, sc_labels = run_spectral_clustering(
            root,
            sim,
            dist_dsne,
            eps=eps,
            it=no_lambdas,
            min_pts=min_pts,
            n_clusters=4,
            type_="it"
        )

        # Plot dbscan with border points
        dbscan = DBSCAN(eps=eps, min_pts=min_pts, cluster_type='standard')
        dbscan.fit(points)
        axes[i, 0].scatter(points[:, 0], points[:, 1], c=dbscan.labels_, s=10, alpha=0.8)

        # Plot dbscan without border points
        dbscan = DBSCAN(eps=eps, min_pts=min_pts, cluster_type='corepoints')
        dbscan.fit(points)
        axes[i, 1].scatter(points[:, 0], points[:, 1], c=dbscan.labels_, s=10, alpha=0.8)

        # Plot k-center clustering
        axes[i, 2].scatter(points[:, 0], points[:, 1], c=kcenter_labels, s=10, alpha=0.8)

        # Plot spectral clustering
        axes[i, 3].scatter(points[:, 0], points[:, 1], c=sc_labels, s=10, alpha=0.8)

        for axis in axes[i]:
            axis.tick_params(
                axis='both',
                which='both',
                bottom=False,
                left=False,
                right=False,
                top=False,
                labelbottom=False,
                labelleft=False
            )

    for i, mu in enumerate(min_pts_list):
        axes[i, 0].set_ylabel(r'$\mu$ = {}'.format(mu))

    for i, label in enumerate(['DBSCAN', 'DBSCAN w/o\nBorderpoints', 'K-Center', 'Spectral']):
        axes[0, i].xaxis.set_label_position('top')
        axes[0, i].set_xlabel(label)

    plt.show()

if __name__ == '__main__':
    g_points, g_labels = make_moons(n_samples=500, noise=0.05)#, random_state=42)
    compare_clusterings(g_points, g_labels)
