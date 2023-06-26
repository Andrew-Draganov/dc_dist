import numpy as np
from distance_metric import get_nearest_neighbors
from tree_plotting import plot_tree
import matplotlib.pyplot as plt

import sys
sys.setrecursionlimit(2000)

class DensityTree:
    def __init__(self, dist, orig_node=None, path='', parent=None):
        self.dist = dist
        self.children = []
        self.left_tree = None
        self.right_tree = None
        self.label = None
        self.point_id = None

        if orig_node is not None:
            self.orig_node = orig_node
        else:
            self.orig_node = self
        self.path = path
        self.parent = parent

    def set_left_tree(self, left):
        self.left_tree = left

    def set_right_tree(self, right):
        self.right_tree = right

    @property
    def has_left_tree(self):
        return self.left_tree is not None

    @property
    def has_right_tree(self):
        return self.right_tree is not None

    #@property
    def is_leaf(self):
        return not (self.has_left_tree or self.has_right_tree)

    @property
    def in_pruned_tree(self):
        return self.orig_node is not None

    def count_children(self):
        if not self.is_leaf:
            if self.left_tree is not None:
                if self.left_tree.is_leaf:
                    self.children += [self.left_tree]
                else:
                    self.children += self.left_tree.children
            if self.right_tree is not None:
                if self.right_tree.is_leaf:
                    self.children += [self.right_tree]
                else:
                    self.children += self.right_tree.children
        else:
            self.children = [self]

    def __len__(self):
        return len(self.children)

def get_inds(all_dists, largest_dist):
    """
    It will usually be the case that one subtree has cardinality 1 and the other has cardinality n-1.
    So just randomly permute them to make tree plotting look balanced.
    """
    equal_inds = np.where(all_dists[0] == largest_dist)[0]
    unequal_inds = np.where(all_dists[0] != largest_dist)[0]
    if np.random.rand() < 0.5:
        left_inds = equal_inds
        right_inds = unequal_inds
    else:
        right_inds = equal_inds
        left_inds = unequal_inds
    return left_inds, right_inds

def _make_tree(all_dists, labels, point_ids, path=''):
    # FIXME -- the right way to do this is to build the tree up while we're connecting the components
    largest_dist = np.max(all_dists)
    root = DensityTree(largest_dist)
    root.path = path
    # FIXME -- this will break if multiple copies of the same point. Need to first check for equal points
    if largest_dist == 0:
        root.label = labels[0]
        root.point_id = point_ids[0]
        # FIXME -- do we need to make the leaf node have itself as a child?
        root.children = [root]
        return root

    left_inds, right_inds = get_inds(all_dists, largest_dist)

    left_split = all_dists[left_inds][:, left_inds]
    left_labels, left_point_ids = labels[left_inds], point_ids[left_inds]
    root.set_left_tree(_make_tree(left_split, left_labels, left_point_ids, path=path+'l'))
    root.left_tree.parent = root

    right_split = all_dists[right_inds][:, right_inds]
    right_labels, right_point_ids = labels[right_inds], point_ids[right_inds]
    root.set_right_tree(_make_tree(right_split, right_labels, right_point_ids, path=path+'r'))
    root.right_tree.parent = root

    root.count_children()
    return root

def make_tree(points, labels, min_points=1, n_neighbors=15, make_image=False, point_ids=None):
    dc_dists = get_nearest_neighbors(
        points,
        n_neighbors=n_neighbors,
        min_points=min_points
    )['_all_dists']
    np.set_printoptions(threshold=np.inf)

    if point_ids is None:
        point_ids = np.arange(int(dc_dists.shape[0]))

    root = _make_tree(dc_dists, labels, point_ids)
    if make_image:
        plot_tree(root, labels)

    return root, dc_dists



if __name__ == '__main__':
    from sklearn.datasets import make_blobs, make_moons

    # --- Data settings ---
    n = 500
    d = 2
    n_clusters = 4
    noise = 0.6

    #### Anna Discord
    n = 25
    d = 2
    n_clusters = 1
    ###

    #

    # create data
#    X, y_true = make_blobs(n_samples=n, n_features=d, centers=n_clusters, cluster_std=noise, shuffle=False, random_state=563)
    X, y_true = make_blobs(n_samples=n, n_features=d, centers=n_clusters, cluster_std=noise, shuffle=False,
                           random_state=1012)

    #plt.scatter(X[:,0], X[:, 1] , c=y_true)
    #plt.show()
    # X, y_true = make_moons(n_samples=100, noise=0.05)

    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1])
    y_true = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1])
    ### Counter example
    n = 75
    d = 2
    n_clusters = 2
    X, y_true = make_blobs(n_samples=n, n_features=d, centers=n_clusters, cluster_std=noise, shuffle=False,
                           random_state=666)

    root, dc_dist = make_tree(X, y_true)
    a = 0