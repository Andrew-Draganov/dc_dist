import numpy as np
from density_tree import DensityTree

class Cluster:
    def __init__(self, center, points, peak):
        self.center = center
        self.points = points
        self.peak = peak

    def __len__(self):
        return len(self.points)

def copy_tree(root, min_points, pruned_parent=None):
    if len(root) >= min_points:
        pruned_root = DensityTree(root.dist, orig_node=root, path=root.path, parent=pruned_parent)
        if root.left_tree is not None:
            pruned_root.set_left_tree(copy_tree(root.left_tree, min_points, pruned_root))
        if root.right_tree is not None:
            pruned_root.set_right_tree(copy_tree(root.right_tree, min_points, pruned_root))
        pruned_root.count_children()
        return pruned_root

    return None

def get_node(root, path):
    if path == '':
        return root
    if path[0] == 'l':
        return get_node(root.left_tree, path[1:])
    return get_node(root.right_tree, path[1:])

def get_lca_path(left, right):
    depth = 0
    while left.path[depth] == right.path[depth]:
        depth += 1
        if depth >= len(left.path) or depth >= len(right.path):
            break
    return left.path[:depth]

def merge_costs(dist, c1, c2):
    merge_right_cost = dist
    merge_left_cost = dist

    return merge_right_cost, merge_left_cost

def merge_clusters(root, clusters, k):
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
            parent_path = get_lca_path(left, right)
            dist = get_node(root, parent_path).dist
            merge_right_cost, merge_left_cost = merge_costs(dist, clusters[i], clusters[i+1])

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
        merge_receiver.peak = get_node(root, get_lca_path(merge_receiver.peak, to_be_merged.peak))
        merge_receiver.points += to_be_merged.points
        clusters.pop(min_i)

    return clusters
                
def cluster_tree(root, subroot, k):
    if len(subroot) <= k:
        clusters = []
        # If this is a single leaf
        if subroot.is_leaf:
            new_cluster = Cluster(subroot, [subroot], subroot)
            return [new_cluster]
        # Else this is a subtree and we want a cluster per leaf
        for leaf in subroot.children:
            point_cluster = Cluster(leaf, [leaf], leaf)
            clusters.append(point_cluster)

        return clusters

    clusters = []
    if subroot.has_left_tree:
        clusters += cluster_tree(root, subroot.left_tree, k)
    if subroot.has_right_tree:
        clusters += cluster_tree(root, subroot.right_tree, k)
    clusters = merge_clusters(root, clusters, k)
    return clusters

def deprune_cluster(node):
    """ Find the cluster's maximum parent and set all the children of that node to be in the cluster """
    if node.is_leaf:
        return [node.point_id]

    points = []
    if node.left_tree is not None:
        points += deprune_cluster(node.left_tree)
    if node.right_tree is not None:
        points += deprune_cluster(node.right_tree)

    return points

def finalize_clusters(clusters):
    """ We could have the setting where
          o
         / \
        o   o
       /     \
      X       X
     / \     / \
    o   o   o   o
    is the pruned set of clusters, where X marks the peak of each cluster.
    If that is the case, we actually want the following
          o
         / \
        X   X
       /     \
      o       o
     / \     / \
    o   o   o   o
    if the new X positions are still less than the maximum epsilon.
    """
    epsilons = np.array([c.peak.dist for c in clusters])
    max_eps = np.max(epsilons[np.where(epsilons > 0)]) + 1e-8
    for cluster in clusters:
        while cluster.peak.parent.dist < max_eps:
            cluster.peak = cluster.peak.parent
    return clusters

def dc_kcenter(root, num_points, k, min_points, with_noise=True):
    # k-center has the option of accounting for noise points
    if with_noise:
        pruned_root = copy_tree(root, min_points)
    else:
        pruned_root = root

    clusters = cluster_tree(pruned_root, pruned_root, k=k)
    clusters = finalize_clusters(clusters)
    for cluster in clusters:
        cluster.points = deprune_cluster(cluster.peak.orig_node)

    return clusters

def get_cluster_metadata(clusters, num_points, k):
    # Nodes that were never put into a cluster will have pred_label -1 (noise points)
    pred_labels = -1 * np.ones(num_points)
    centers = np.zeros(k, dtype=np.int32)
    for i, cluster in enumerate(clusters):
        if cluster.center.orig_node is None:
            centers[i] = cluster.center.point_id
        else:
            centers[i] = cluster.center.orig_node.children[0].point_id
        for point in cluster.points:
            pred_labels[point] = i
    epsilons = np.array([c.peak.dist for c in clusters])

    return pred_labels, centers, epsilons

def dc_clustering(root, num_points, k=4, min_points=1, with_noise=True):
    clusters = dc_kcenter(root, num_points, k, min_points, with_noise=with_noise)

    return get_cluster_metadata(clusters, num_points, k)

