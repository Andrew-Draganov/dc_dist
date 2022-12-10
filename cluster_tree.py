import numpy as np
from density_preserving_embeddings import make_tree

class Cluster:
    def __init__(self, center, points, cost, max_dist):
        self.center = center
        self.points = points
        self.cost = cost
        self.max_dist = max_dist

    def __len__(self):
        return len(self.points)

def prune_tree(root, min_points):
    # FIXME -- is the size correct here?
    if root.left_tree is not None:
        if len(root.left_tree) < min_points:
            root.left_tree = None
        else:
            root.left_tree = prune_tree(root.left_tree, min_points)

    if root.right_tree is not None:
        if len(root.right_tree) < min_points:
            root.right_tree = None
        else:
            root.right_tree = prune_tree(root.right_tree, min_points)

    return root

def get_dist(root, path):
    if path == '':
        return root.dist
    if path[0] == 'l':
        return get_dist(root.left_tree, path[1:])
    return get_dist(root.right_tree, path[1:])

def merge_costs(dist, c1, c2, norm):
    if norm == 2 or norm == 1:
        # FIXME -- do I want to be subtracting the cost here?
        merge_right_cost = len(c1) * dist ** norm # - c1.cost
        merge_left_cost = len(c2) * dist ** norm # - c2.cost
    else:
        assert norm == np.inf
        merge_right_cost = dist#  - c1.cost
        merge_left_cost = dist#  - c2.cost

    return merge_right_cost, merge_left_cost

def merge_clusters(root, clusters, k, norm):
    while len(clusters) > k:
        # Placeholder variables to find best merge location
        min_cost = np.inf
        min_dist = 0
        min_i = 0
        to_be_merged = None
        merge_receiver = None
        left = False

        for i in range(len(clusters) - 1):
            left = clusters[i].center
            right = clusters[i+1].center
            
            # Get cost of merging between left and right clusters
            depth = 0
            while left.path[depth] == right.path[depth]:
                depth += 1
            parent_path = left.path[:depth]
            dist = get_dist(root, parent_path)
            merge_right_cost, merge_left_cost = merge_costs(
                dist,
                clusters[i],
                clusters[i+1],
                norm
            )

            # Track all necessary optimal merge parameters
            if min(merge_right_cost, merge_left_cost) < min_cost:
                left_right = merge_right_cost < merge_left_cost
                min_i = i
                if not left_right:
                    left_right = -1 
                    min_i = i + 1
                to_be_merged = clusters[min_i]
                merge_receiver = clusters[min_i + left_right]
                min_cost = min(merge_right_cost, merge_left_cost)
                min_dist = dist

        # Merge the smaller cluster into the bigger cluster. Delete the smaller cluster.
        merge_receiver.points += to_be_merged.points
        merge_receiver.cost += len(to_be_merged) * dist ** 2
        max_dist = max([dist, merge_receiver.max_dist, to_be_merged.max_dist])
        merge_receiver.max_dist = max_dist
        clusters.pop(min_i)

    return clusters
                
def cluster_tree(root, subroot, k, norm):
    if len(subroot) <= k:
        clusters = []
        # If this is a single leaf
        if subroot.is_leaf():
            return [Cluster(subroot, [subroot], 0, 0)]
        # Else this is a subtree and we want a cluster per leaf
        for leaf in subroot.children:
            point_cluster = Cluster(leaf, [leaf], 0, 0)
            clusters.append(point_cluster)

        return clusters

    left_clusters = cluster_tree(root, subroot.left_tree, k, norm)
    right_clusters = cluster_tree(root, subroot.right_tree, k, norm)
    clusters = merge_clusters(root, left_clusters + right_clusters, k, norm)
    return clusters


def dc_kmeans(root, num_points, k=4, prune=False, min_points=1, norm=2):
    if prune:
        root = prune_tree(root, min_points)
    clusters = cluster_tree(root, root, k=k, norm=norm)

    pred_labels = np.zeros(num_points)
    for i, cluster in enumerate(clusters):
        for point in cluster.points:
            pred_labels[point.point_id] = i
    epsilons = np.array([c.max_dist for c in clusters])

    return pred_labels, epsilons

