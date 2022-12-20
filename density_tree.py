import numpy as np
from distance_metric import get_nearest_neighbors
from tree_plotting import plot_tree
import matplotlib.pyplot as plt

class DensityTree:
    def __init__(self, dist):
        self.dist = dist
        self.children = []
        self.left_tree = None
        self.right_tree = None
        self.label = None
        self.point_id = None
        self.path = ''

    def set_left_tree(self, left):
        self.left_tree = left

    def set_right_tree(self, right):
        self.right_tree = right

    def is_leaf(self):
        return self.label is not None

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
    # FIXME -- this will break if multiple copies of the same point. Need to first check for equal points
    if largest_dist == 0:
        root.label = labels[0]
        root.point_id = point_ids[0]
        root.path = path
        return root

    left_inds, right_inds = get_inds(all_dists, largest_dist)

    left_split = all_dists[left_inds][:, left_inds]
    left_labels, left_point_ids = labels[left_inds], point_ids[left_inds]
    root.set_left_tree(_make_tree(left_split, left_labels, left_point_ids, path=path+'l'))
    if root.left_tree.is_leaf():
        root.children += [root.left_tree]
    else:
        root.children += root.left_tree.children

    right_split = all_dists[right_inds][:, right_inds]
    right_labels, right_point_ids = labels[right_inds], point_ids[right_inds]
    root.set_right_tree(_make_tree(right_split, right_labels, right_point_ids, path=path+'r'))
    if root.right_tree.is_leaf():
        root.children += [root.right_tree]
    else:
        root.children += root.right_tree.children

    return root

def make_tree(points, labels, min_points=1, n_neighbors=15, make_image=True, point_ids=None):
    dc_dists = get_nearest_neighbors(\
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

