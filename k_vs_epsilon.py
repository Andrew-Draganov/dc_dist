import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.metrics import normalized_mutual_info_score as nmi
from tqdm import tqdm

from experiment_utils.get_data import get_dataset, make_circles
from DBSCAN import DBSCAN
from distance_metric import get_nearest_neighbors
from density_tree import make_tree
from cluster_tree import dc_clustering

def k_vs_eps(points, labels):
    min_pts_list = [1, 3, 5, 7, 9]
    k_list = [2, 6, 18, 54, 162]
    epsilons = np.zeros((len(min_pts_list), len(k_list)))
    nmi_vals = np.zeros((len(min_pts_list), len(k_list)))
    nmi_size_scalar = 2000
    sc = []
    plt.rcParams.update({'text.usetex': True, 'font.size': 14})
    colors = ['turquoise', 'cyan', 'skyblue', 'slateblue', 'navy']
    for i, min_pts in tqdm(enumerate(min_pts_list), total=len(min_pts_list)):
        root, _ = make_tree(
            points,
            labels,
            min_points=min_pts,
        )
        for j, k in enumerate(k_list):
            # Calculate kcenter clustering
            kcenter_labels, _, kcenter_epsilons = dc_clustering(
                root,
                num_points=len(labels),
                k=k,
                min_points=min_pts,
                norm=np.inf
            )

            # Get epsilon given by k-center clustering
            epsilons[i, j] = np.max(kcenter_epsilons[np.where(kcenter_epsilons > 0)]) + 1e-8
            nmi_vals[i, j] = nmi(labels, kcenter_labels)

    # Plot points
    for i in range(len(min_pts_list)):
        scattered = plt.scatter(
            k_list,
            epsilons[i, :],
            s=nmi_vals[i, :]*nmi_size_scalar,
            c=colors[i],
            alpha=0.5
        )
    plt.xscale('log')

    # Draw a line for each min_pts
    for i, min_pts in enumerate(min_pts_list):
        plt.plot(
            np.array(k_list),
            epsilons[i, :],
            c=colors[i],
            linewidth=3,
            alpha=0.7,
            linestyle='dashed',
        )

    # For the max nmi in each min_pts, draw a red circle around it
    max_inds = np.argmax(nmi_vals, axis=1)
    plt.scatter(
        np.array(k_list)[max_inds],
        [epsilons[i, max_inds[i]] for i in range(len(max_inds))],
        [nmi_vals[i, max_inds[i]] * nmi_size_scalar for i in range(len(max_inds))],
        facecolors='none',
        edgecolors='red',
        alpha=0.75,
        linewidths=3,
        linestyles='dashed'
    )
    plt.xticks(k_list, ['2', '6', '18', '54', '162'])
    axis = plt.gca()

    color_handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colors]
    color_legend = axis.legend(color_handles, [r'$\mu=$ {}'.format(str(min_pts)) for min_pts in min_pts_list], loc='upper right')
    axis.add_artist(color_legend)

    legend_info = scattered.legend_elements("sizes", num=5)
    nmi_sizes = [0.2, 0.3, 0.4, 0.5]
    plt.legend(legend_info[0], ['nmi={}'.format(nmi) for nmi in nmi_sizes], loc='upper left')
    plt.show()

if __name__ == '__main__':
    g_points, g_labels = get_dataset('coil', class_list=np.arange(11, 31), points_per_class=36)
    k_vs_eps(g_points, g_labels, 'coil')
