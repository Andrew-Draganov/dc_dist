import numpy as np
from density_tree import DensityTree

class Cluster:
    def __init__(self, center, points, cost, peak):
        self.center = center
        self.points = points
        self.cost = cost
        self.peak = peak

    def __len__(self):
        return len(self.points)

def copy_tree(root, prune_size):
    # FIXME -- the if-statement is wrong. Under this logic, prune_size=0 will still give noise points
    if len(root) > prune_size:
        pruned_root = DensityTree(root.dist, orig_node=root, path=root.path, parent=root.parent)
        pruned_root.set_left_tree(copy_tree(root.left_tree, prune_size))
        pruned_root.set_right_tree(copy_tree(root.right_tree, prune_size))
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
    return left.path[:depth]

def merge_costs(dist, c1, c2, norm):
    if norm >= 0 and norm != np.inf:
        merge_right_cost = len(c1) * dist ** norm
        merge_left_cost = len(c2) * dist ** norm
    else:
        assert norm == np.inf
        merge_right_cost = dist
        merge_left_cost = dist

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
            parent_path = get_lca_path(left, right)
            dist = get_node(root, parent_path).dist
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
        merge_receiver.peak = get_node(root, get_lca_path(merge_receiver.center, to_be_merged.center))
        merge_receiver.points += to_be_merged.points
        merge_receiver.cost += len(to_be_merged) * dist ** 2
        clusters.pop(min_i)

    return clusters
                
def cluster_tree(root, subroot, k, norm):
    if len(subroot) <= k:
        clusters = []
        # If this is a single leaf
        if subroot.is_leaf:
            new_cluster = Cluster(subroot, [subroot], 0, subroot)
            return [new_cluster]
        # Else this is a subtree and we want a cluster per leaf
        for leaf in subroot.children:
            point_cluster = Cluster(leaf, [leaf], 0, subroot)
            clusters.append(point_cluster)

        return clusters

    clusters = []
    if subroot.has_left_tree:
        clusters += cluster_tree(root, subroot.left_tree, k, norm)
    if subroot.has_right_tree:
        clusters += cluster_tree(root, subroot.right_tree, k, norm)
    clusters = merge_clusters(root, clusters, k, norm)
    return clusters

def deprune_clusters(clusters):
    """ Find the cluster's maximum parent and set all the children of that node to be in the cluster """
    depruned_clusters = []
    for cluster in clusters:
        cluster.points = cluster.peak.orig_node.children

def dc_kmeans(root, num_points, k=4, prune_size=0, min_points=1, norm=2):
    # FIXME -- notes for later
    # 1) Shouldn't do k-means and k-median on pruned trees since the addition of new nodes would mess up the weighting
    #    It only makes sense for the pruned tree
    # 2) Using prune_size > 1 creates a ton of noise points because our distance measure is max(d(a, b), knn(a), knn(b))??
    # 2.a) Using prune_size = 0 gives some noise points still??
    # 3) k-median gives TONS of noise points for some reason. In groups that are way too large
    pruned_root = copy_tree(root, prune_size)
    clusters = cluster_tree(pruned_root, pruned_root, k=k, norm=norm)
    deprune_clusters(clusters)
    pred_labels = -1 * np.ones(num_points)
    centers = np.zeros(k, dtype=np.int32)
    # Nodes that were never put into a cluster will have pred_label -1 (noise points)
    for i, cluster in enumerate(clusters):
        centers[i] = cluster.center.orig_node.children[0].point_id
        for point in cluster.points:
            pred_labels[point.point_id] = i
    epsilons = np.array([c.peak.dist for c in clusters])

    return pred_labels, centers, epsilons

