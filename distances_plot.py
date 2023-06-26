from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll, make_moons
from sklearn.cluster import DBSCAN

from distance_metric import get_nearest_neighbors
from density_tree import make_tree
from cluster_tree import dc_clustering

def distance_plot():
    plt.rcParams.update({'font.size': 10, 'text.usetex': True})
    fig, ax = plt.subplots(nrows=1, ncols=3)
    fig.set_figheight(6)
    fig.set_figwidth(12)
    num_points = 200
    range_length = 10
    star_pt_index = 50
    min_pts = 3
    k = 4
    cmap = np.array(['red', 'blue', 'green', 'yellow', 'orange', 'brown', 'black', 'purple'])

    points, labels = make_moons(n_samples=num_points, noise=0.1)
    points -= np.min(points, axis=0, keepdims=True)
    points /= np.max(points, axis=0, keepdims=True)
    points *= range_length

    root, _ = make_tree(
        points,
        labels,
        min_points=min_pts,
        make_image=False,
        n_neighbors=15
    )

    pred_labels, centers, epsilons = dc_clustering(
        root,
        num_points=num_points,
        k=k,
        min_points=min_pts,
    )
    sorted_eps = np.sort(epsilons)
    levels = [
        sorted_eps[0]/2,
        sorted_eps[0],
        sorted_eps[1],
        sorted_eps[2],
        sorted_eps[3],
        sorted_eps[3] * 1.5,
        sorted_eps[3] * 2,
        sorted_eps[3] * 3
    ]

    max_eps = np.max(epsilons[np.where(epsilons > 0)]) + 1e-8
    dbscan_orig = DBSCAN(eps=max_eps, min_samples=min_pts).fit(points)
    X, Y = np.meshgrid(np.arange(range_length+1), np.arange(range_length+1))

    star_dists = np.zeros((range_length+1, range_length+1))
    center_dists = np.zeros((range_length+1, range_length+1, k))

    for x in tqdm(range(range_length + 1), total=range_length+1):
        for y in range(range_length + 1):
            augmented_points = np.concatenate([points, np.array([[x, y]])], axis=0)
            dc_dists = get_nearest_neighbors(
                augmented_points,
                n_neighbors=15,
                min_points=min_pts
            )['_all_dists']

            star_dists[x, y] = dc_dists[-1, star_pt_index]
            for i, center_ind in enumerate(centers):
                center_dists[x, y, i] = dc_dists[-1, center_ind]

    cf_plot = ax[0].contourf(X, Y, star_dists)

    ax[0].scatter(points[:, 1], points[:, 0], s=5)
    ax[0].scatter(points[star_pt_index, 1], points[star_pt_index, 0], c='r', marker='*', s=25)
    plt.colorbar(cf_plot, ax=ax[0], shrink=0.8, extend='both', location='left')
    ax[0].set_title('Distance to Star as a function of location')

    cf_plot = ax[1].contourf(
        X,
        Y,
        np.min(center_dists, axis=-1),
        levels=levels,
        alpha=0.6
    )
    for i in range(k):
        ax[1].contour(X, Y, center_dists[:, :, i], levels=[epsilons[i]], colors=(cmap[pred_labels[centers[i]].astype(np.int32)],))
        ax[1].scatter(points[centers[i], 1], points[centers[i], 0], c=cmap[pred_labels[centers[i]].astype(np.int32)], marker='*')
    ax[1].scatter(points[:, 1], points[:, 0], s=5, c=cmap[pred_labels.astype(np.int32)])
    ax[1].set_title('k-center with k=4')

    cf_plot = ax[2].contourf(
        X,
        Y,
        np.min(center_dists, axis=-1),
        levels=levels,
        alpha=0.3
    )
    ax[2].set_title('DBSCAN with eps=Max(k-center epsilons)')
    ax[2].scatter(points[:, 1], points[:, 0], s=5, c=cmap[dbscan_orig.labels_])
    plt.colorbar(cf_plot, ax=ax[2], shrink=0.8, extend='both', location='right')

    plt.show()


def plot_dbscan(X, dbscan, ax, flag_plotBorderPts=False):
    if not flag_plotBorderPts:
        # plot clusters
        ax.scatter(X[:, 0][dbscan.labels_ != -1],
                   X[:, 1][dbscan.labels_ != -1],
                   c=dbscan.labels_[dbscan.labels_ != -1],
                   s=25, cmap='viridis')
        # higlight noise pts
        ax.scatter(X[:, 0][dbscan.labels_ == -1],
                   X[:, 1][dbscan.labels_ == -1],
                   s=50,
                   color="red")
    else:
        core_indices = dbscan.core_sample_indices_
        mask = np.full(len(X), False, dtype=bool)
        mask[core_indices] = 1
        # just plot core points
        ax.scatter(X[:, 0][mask],
                   X[:, 1][mask],
                   c=dbscan.labels_[mask],
                   s=25, cmap='viridis')
        # highlight border pts
        ax.scatter(X[:, 0][~mask & (dbscan.labels_ != -1)],
                   X[:, 1][~mask & (dbscan.labels_ != -1)],
                   s=50, color="gray")
        # higlight noise pts
        ax.scatter(X[:, 0][dbscan.labels_ == -1],
                   X[:, 1][dbscan.labels_ == -1],
                   s=50,
                   color="red")
    # customize plot
    ax.set_title("DBSCAN - $minPts:{minPts}$, $\epsilon:{eps}$".format(
        minPts=dbscan.min_samples,
        eps=dbscan.eps))
    ax.set_xlabel("1st feature")
    ax.set_ylabel("2nd feature")


if __name__ == '__main__':
    distance_plot()
