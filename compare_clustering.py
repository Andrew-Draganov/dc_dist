import os
from sklearn.datasets import make_moons, make_blobs
from sklearn.cluster import SpectralClustering as sk_spec
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from DBSCAN import DBSCAN
from experiment_utils.get_data import get_dataset, make_circles
from distance_metric import get_dc_dist_matrix
from density_tree import make_tree
from cluster_tree import dc_clustering
from SpectralClustering import get_lambdas, get_sim_mx, run_spectral_clustering

def compare_clusterings():
    k = 4
    min_pts = 3
    datasets = ['moons', 'circles', 'blobs']
    fig, axes = plt.subplots(len(datasets), 9)
    fig.set_figheight(len(datasets) * 2 + 2)
    fig.set_figwidth(12)
    plt.rcParams.update({'text.usetex': True, 'font.size': 16})

    all_pointsets = {
        'moons': make_moons(n_samples=500, noise=0.1),
        'circles': make_circles(
            n_samples=500,
            noise=0.03,
            radii=[0.5, 1.0, 1.5, 2.0],
            thicknesses=[0.1, 0.1, 0.1, 0.1]
        ),
        'blobs': make_blobs(n_samples=500, centers=6)
    }

    for i, dataset in enumerate(datasets):
        points, labels = all_pointsets[dataset]

        root, dc_dists = make_tree(
            points,
            labels,
            min_points=min_pts,
        )
        ###########################################
        # Calculate kcenter clustering on dc_dist #
        ###########################################
        noise_labels, _, noise_epsilons = dc_clustering(
            root,
            num_points=len(labels),
            k=k,
            min_points=min_pts,
        )

        # Calculate kcenter clustering without noise
        no_noise_labels, _, no_noise_epsilons = dc_clustering(
            root,
            num_points=len(labels),
            k=k,
            min_points=min_pts,
            with_noise=False
        )
        euc_dist_matrix = get_dist_matrix(points, 'euclidean')
        kc_labels, costs = approx_kcenter(euc_dist_matrix, k)

        # Get epsilon given by k-center clustering
        noise_eps = np.max(noise_epsilons[np.where(noise_epsilons > 0)]) + 1e-8

        # Get epsilon given by k-center clustering
        no_noise_eps = np.max(no_noise_epsilons[np.where(no_noise_epsilons > 0)]) + 1e-8

        ############################################
        # Calculate spectral clustering on dc_dist #
        ############################################
        dist_dsne = get_dc_dist_matrix(points, 15, min_points=min_pts)
        no_lambdas = get_lambdas(root, noise_eps)
        sim = get_sim_mx(dist_dsne)
        dc_sc_, dc_sc_labels = run_spectral_clustering(
            root,
            sim,
            dist_dsne,
            eps=noise_eps,
            it=no_lambdas,
            min_pts=min_pts,
            n_clusters=k,
            type_="it"
        )

        euc_sc_labels = affinity_spectral(euc_dist_matrix, k)

        # Calculate spectral clustering
        no_lambdas = get_lambdas(root, noise_eps)
        sim = get_sim_mx(euc_dist_matrix)
        euc_mst_sc_, euc_mst_sc_labels = run_spectral_clustering(
            root,
            sim,
            euc_dist_matrix,
            eps=noise_eps,
            it=no_lambdas,
            min_pts=min_pts,
            n_clusters=k,
            type_="it"
        )


        # Plot k-center clustering with noise
        non_noise = np.where(noise_labels>=0)
        noise = np.where(noise_labels<0)
        axes[i, 0].scatter(points[noise, 0], points[noise, 1], c='gray', s=10, alpha=0.5)
        axes[i, 0].scatter(points[non_noise, 0], points[non_noise, 1], c=noise_labels[non_noise], s=10, alpha=0.8)

        # Plot k-center clustering without noise
        axes[i, 1].scatter(points[:, 0], points[:, 1], c=no_noise_labels[:], s=10, alpha=0.8)

        # Plot k-center clustering in Euclidean
        axes[i, 2].scatter(points[:, 0], points[:, 1], c=kc_labels, s=10, alpha=0.8)



        # Plot dbscan on dc_dist without border points
        dbscan = DBSCAN(eps=noise_eps, min_pts=min_pts, cluster_type='corepoints')
        dbscan.fit(points, dist_mx=dist_dsne)
        non_noise = np.where(dbscan.labels_>=0)
        noise = np.where(dbscan.labels_<0)
        axes[i, 3].scatter(points[noise, 0], points[noise, 1], c='gray', s=10, alpha=0.5)
        axes[i, 3].scatter(points[non_noise, 0], points[non_noise, 1], c=dbscan.labels_[non_noise], s=10, alpha=0.8)

        # Plot dbscan without border points
        dbscan = DBSCAN(eps=noise_eps, min_pts=min_pts, cluster_type='corepoints')
        dbscan.fit(points)
        non_noise = np.where(dbscan.labels_>=0)
        noise = np.where(dbscan.labels_<0)
        axes[i, 4].scatter(points[noise, 0], points[noise, 1], c='gray', s=10, alpha=0.5)
        axes[i, 4].scatter(points[non_noise, 0], points[non_noise, 1], c=dbscan.labels_[non_noise], s=10, alpha=0.8)

        # Plot dbscan with border points
        dbscan = DBSCAN(eps=noise_eps, min_pts=min_pts, cluster_type='standard')
        dbscan.fit(points)
        non_noise = np.where(dbscan.labels_>=0)
        noise = np.where(dbscan.labels_<0)
        axes[i, 5].scatter(points[noise, 0], points[noise, 1], c='gray', s=10, alpha=0.5)
        axes[i, 5].scatter(points[non_noise, 0], points[non_noise, 1], c=dbscan.labels_[non_noise], s=10, alpha=0.8)



        # Plot spectral clustering
        non_noise = np.where(dc_sc_labels>=0)
        noise = np.where(dc_sc_labels<0)
        axes[i, 6].scatter(points[noise, 0], points[noise, 1], c='gray', s=10, alpha=0.5)
        axes[i, 6].scatter(points[non_noise, 0], points[non_noise, 1], c=dc_sc_labels[non_noise], s=10, alpha=0.8)

        # Default spectral clustering on Euclidean
        axes[i, 7].scatter(points[:, 0], points[:, 1], c=euc_sc_labels, s=10, alpha=0.8)

        # Our spectral clustering on Euclidean
        axes[i, 8].scatter(points[:, 0], points[:, 1], c=euc_mst_sc_labels, s=10, alpha=0.8)

    for row in axes:
        for axis in row:
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

    for i, label in enumerate([
        r'$k$-Center with'
        '\n'
        r'$q$-Coverage'
        '\n'
        r'on $d_{dc}$',
        r'$k$-Center'
        '\n'
        r'on $d_{dc}$',
        r'$k$-Center'
        '\non Euclidean',
        'DBSCAN*'
        '\n'
        r'on $d_{dc}$',
        'DBSCAN*',
        'Original\nDBSCAN',
        'Our method'
        '\n'
        r'on $d_{dc}$',
        'RBF Spectral',
        'Our method'
        '\n'
        r'on Euclidean',
    ]):
        axes[0, i].xaxis.set_label_position('top')
        axes[0, i].set_xlabel(label)

    gs = gridspec.GridSpec(len(datasets)+1, 9)
    for i, row in enumerate(axes):
        for j, ax in enumerate(row):
            ax.set_subplotspec(gs[i+1, j])

    aux1 = fig.add_subplot(gs[0, :3])
    aux1.set_xlabel(r'$k$-Center')
    aux1.xaxis.set_label_position('top')

    aux2 = fig.add_subplot(gs[0, 3:6])
    aux2.set_xlabel('DBSCAN')
    aux2.xaxis.set_label_position('top')

    aux3 = fig.add_subplot(gs[0, 6:])
    aux3.set_xlabel('Spectral')
    aux3.xaxis.set_label_position('top')

    for i, row in enumerate(axes):
        for j, ax in enumerate(row):
            box = ax.get_position()
            box.y0 = box.y0 + 0.13
            box.y1 = box.y1 + 0.13
            ax.set_position(box)

    for ax in [aux1, aux2, aux3]:
        ax.tick_params(size=0)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_facecolor("none")
        for pos in ["right", "left", "bottom"]:
            ax.spines[pos].set_visible(False)
        ax.spines["top"].set_linewidth(3)
        ax.spines["top"].set_color("crimson")

    for row in axes:
        for index in [0, 3, 6]:
            for spine in ['bottom', 'top', 'left', 'right']:
                row[index].spines[spine].set_linestyle('dotted')
                row[index].spines[spine].set_linewidth(1.5)
                row[index].spines[spine].set_color('blue')

    image_dir = 'scratch'
    os.makedirs(image_dir, exist_ok=True)
    plt.show()
    # plt.savefig(
    #     os.path.join(image_dir, 'cluster_comparison.pdf'),
    #     bbox_inches='tight',
    #     pad_inches=0.1
    # )




def euc_dist(p1, p2):
    return np.sqrt(np.sum(np.square(p1 - p2)))

def approx_kcenter(dist_matrix, k):
    # The basic 2-approx for k-center in any metric:
    # "Always sample the farthest point from the current set of centers"
    assert k > 0
    n = len(dist_matrix)
    centers = np.zeros(k, dtype=np.int32)
    centers[0] = np.random.choice(n)
    for i in range(1, k):
        point_dists = np.zeros(n)
        for p in range(n):
            min_dist = np.inf
            for k, center in enumerate(centers[:i]):
                if dist_matrix[p, center] < min_dist:
                    min_dist = dist_matrix[p, center]
            point_dists[p] = min_dist

        centers[i] = np.argmax(point_dists)

    labels = np.zeros(n)
    costs = np.zeros(n)
    for i in range(n):
        min_dist = np.inf
        min_ind = None
        for j, center in enumerate(centers):
            if dist_matrix[i, center] < min_dist:
                min_dist = dist_matrix[i, center]
                min_ind = j
        labels[i] = min_ind
        costs[i] = min_dist

    return labels, costs

def get_dist_matrix(points, dist, min_points=3):
    n = len(points)
    if dist in ['euclidean', 'reachability']:
        dist_matrix = np.zeros((len(points), len(points)))
        for i, point_i in enumerate(points):
            for j, point_j in enumerate(points):
                dist_matrix[i, j] = euc_dist(point_i, point_j)

        if dist == 'reachability':
            # Get reachability for each point with respect to min_points parameter
            reach_dists = np.sort(dist_matrix, axis=1)
            reach_dists = reach_dists[:, min_points - 1]

            # Make into an NxN matrix
            reach_dists_i, reach_dists_j = np.meshgrid(reach_dists, reach_dists)

            # Take max of reach_i, D_ij, reach_j
            dist_matrix = np.stack([dist_matrix, reach_dists_i, reach_dists_j], axis=-1)
            dist_matrix = np.max(dist_matrix, axis=-1)

            # Zero out the diagonal so that it's a distance metric
            diag_mask = np.ones([n, n]) - np.eye(n)
            dist_matrix *= diag_mask

        return dist_matrix

    assert dist == 'connectivity'
    return get_nearest_neighbors(points, 15, min_points=1)['_all_dists']


def rbf(dist_matrix):
    return np.exp( -1 * dist_matrix )

def affinity_spectral(dist_matrix, k):
    model = sk_spec(k, affinity='precomputed')
    dist_matrix = rbf(dist_matrix)
    model.fit(dist_matrix)
    return model.labels_


def visualize_diffs(points, labels):
    k = 2
    min_pts = 3
    dists = ['euclidean', 'reachability', 'connectivity']
    fig, axes = plt.subplots(len(dists)+1, 3)
    fig.set_figheight(8)
    fig.set_figwidth(10)
    plt.rcParams.update({'text.usetex': True, 'font.size': 22})
    for i, dist in enumerate(dists):
        dist_matrix = get_dist_matrix(points, dist, min_pts)
        dbscan = DBSCAN(
            eps=np.max(kc_costs),
            min_pts=min_pts,
            cluster_type='corepoints'
        )
        dbscan.fit(points, dist_mx=dist_matrix)

        # Plot clustering results
        axes[i, 0].scatter(points[:, 0], points[:, 1], c=kc_labels, s=10, alpha=0.8)
        axes[i, 1].scatter(points[:, 0], points[:, 1], c=sp_labels, s=10, alpha=0.8)
        axes[i, 2].scatter(points[:, 0], points[:, 1], c=dbscan.labels_, s=10, alpha=0.8)

    for axis in axes:
        for j in range(len(axis)):
            axis[j].tick_params(
                axis='both',
                which='both',
                bottom=False,
                left=False,
                right=False,
                top=False,
                labelbottom=False,
                labelleft=False
            )


    root, dc_dists = make_tree(
        points,
        labels,
        min_points=min_pts,
    )
    # Calculate kcenter clustering
    noise_labels, _, noise_epsilons = dc_clustering(
        root,
        num_points=len(labels),
        k=k,
        min_points=min_pts,
    )
    for j in range(len(axes[0])):
        axes[-1, j].scatter(points[:, 0], points[:, 1], c=noise_labels, s=10, alpha=0.8)

    for j in range(len(axes[0])):
        for spine in ['bottom', 'top', 'left', 'right']:
            axes[-1, j].spines[spine].set_linestyle('dotted')
            axes[-1, j].spines[spine].set_linewidth(1.5)
            axes[-1, j].spines[spine].set_color('blue')

    dists.append('dc-distance')
    for i, dist in enumerate(dists):
        axes[i, 0].set_ylabel(dist)
        if i == len(dists) - 1:
            axes[i, 0].set_ylabel(dist, color='blue')

    for i, label in enumerate([
        r'$k$-Center',
        'Spectral',
        r'Max $\varepsilon$ DBSCAN',
    ]):
        axes[0, i].xaxis.set_label_position('top')
        axes[0, i].set_xlabel(label)

    plt.show()



if __name__ == '__main__':
    compare_clusterings()
