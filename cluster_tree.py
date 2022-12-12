import numpy as np
from density_tree import DensityTree

class Cluster:
    def __init__(self, center, points, cost, max_dist):
        self.center = center
        self.points = points
        self.cost = cost
        self.max_dist = max_dist

    def __len__(self):
        return len(self.points)

class PrunedTree(DensityTree):
    def __init__(self, dist, orig_node):
        super().__init__(dist)
        self.orig_node = orig_node
        self.path = self.orig_node.path
        self.in_pruned_tree = True


def copy_tree(root, prune_size):
    assert isinstance(root, DensityTree)
    pruned_root = PrunedTree(root.dist, root)
    if not root.is_leaf():
        if len(root.left_tree) < prune_size and len(root.right_tree) < prune_size:
            pruned_root.label = -1
            pruned_root.dist = 0
        elif len(root.left_tree) < prune_size:
            pruned_root.set_right_tree(copy_tree(root.right_tree, prune_size))
            if pruned_root.right_tree.is_leaf():
                pruned_root.children += [pruned_root.right_tree]
            else:
                pruned_root.children += pruned_root.right_tree.children
        elif len(root.right_tree) < prune_size:
            pruned_root.set_left_tree(copy_tree(root.left_tree, prune_size))
            if pruned_root.left_tree.is_leaf():
                pruned_root.children += [pruned_root.left_tree]
            else:
                pruned_root.children += pruned_root.left_tree.children
        else:
            pruned_root.set_left_tree(copy_tree(root.left_tree, prune_size))
            pruned_root.set_right_tree(copy_tree(root.right_tree, prune_size))
            pruned_root.children = pruned_root.left_tree.children + pruned_root.right_tree.children
    else:
        pruned_root.label = root.label

    return pruned_root

def get_dist(root, path):
    if path == '':
        return root.dist
    if path[0] == 'l':
        return get_dist(root.left_tree, path[1:])
    return get_dist(root.right_tree, path[1:])

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
            new_cluster = Cluster(subroot, [subroot], 0, 0)
            return [new_cluster]
        # Else this is a subtree and we want a cluster per leaf
        for leaf in subroot.children:
            point_cluster = Cluster(leaf, [leaf], 0, 0)
            clusters.append(point_cluster)

        return clusters

    clusters = []
    if subroot.left_tree is not None:
        clusters += cluster_tree(root, subroot.left_tree, k, norm)
    if subroot.right_tree is not None:
        clusters += cluster_tree(root, subroot.right_tree, k, norm)
    clusters = merge_clusters(root, clusters, k, norm)
    return clusters

def dc_kmeans(root, num_points, k=4, prune_size=0, min_points=1, norm=2):
    pruned_root = copy_tree(root, prune_size)
    clusters = cluster_tree(pruned_root, pruned_root, k=k, norm=norm)
    pred_labels = -1 * np.ones(num_points)
    centers = np.zeros(k, dtype=np.int32)
    # Nodes that were never put into a cluster will have pred_label -1 (noise points)
    for i, cluster in enumerate(clusters):
        centers[i] = cluster.center.orig_node.children[0].point_id
        for point in cluster.points:
            for child in point.orig_node.children:
                pred_labels[child.point_id] = i
    epsilons = np.array([c.max_dist for c in clusters])

    return pred_labels, centers, epsilons

