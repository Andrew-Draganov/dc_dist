import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from tqdm.auto import tqdm
import itertools
import numba

class Component:
    def __init__(self, nodes, comp_id):
        self.nodes = set(nodes)
        self.comp_id = comp_id

def merge_components(c_i, c_j):
    merged_list = c_i.nodes.union(c_j.nodes)
    return Component(merged_list, c_i.comp_id)

def get_nearest_neighbors(points, n_neighbors, min_points=1, **kwargs):
    """
    We define the distance from x_i to x_j as min(max(P(x_i, x_j))), where 
        - P(x_i, x_j) is any path from x_i to x_j
        - max(P(x_i, x_j)) is the largest edge weight in the path
        - min(max(P(x_i, x_j))) is the smallest largest edge weight
    """
    @numba.njit(fastmath=True, parallel=True)
    def get_dist_matrix(points, D, dim, num_points, min_points=1, reach=None):
        for i in numba.prange(num_points):
            x = points[i]
            for j in range(i+1, num_points):
                y = points[j]
                dist = 0
                for d in range(dim):
                    dist += (x[d] - y[d]) ** 2
                dist = np.sqrt(dist)
                D[i, j] = dist
                D[j, i] = dist

        return D
        # assert reach_D is not None

        # for i in numba.prange(num_points):
        #     reach_i = 0
        #     for j in range(i+1, num_points):
        #         y = points[j]
        #         dist = 0
        #         for d in range(dim):
        #             dist += (x[d] - y[d]) ** 2
        #         D[i, j] = dist
        #         D[j, i] = dist


    num_points = int(points.shape[0])
    dim = int(points.shape[1])
    density_connections = np.zeros([num_points, num_points])
    D = np.zeros([num_points, num_points])
    D = get_dist_matrix(points, D, dim, num_points)

    flat_D = np.reshape(D, [num_points * num_points])
    argsort_inds = np.argsort(flat_D)

    num_added = 0
    component_dict = {i: Component([i], i) for i in range(num_points)}
    neighbor_dists = [[] for i in range(num_points)]
    neighbor_inds = [[] for i in range(num_points)]
    max_comp_size = 1
    for index in tqdm(argsort_inds):
        i = int(index / num_points)
        j = index % num_points
        if component_dict[i].comp_id != component_dict[j].comp_id:
            epsilon = D[i, j]
            for node_i in component_dict[i].nodes:
                for node_j in component_dict[j].nodes:
                    density_connections[node_i, node_j] = epsilon
                    density_connections[node_j, node_i] = epsilon
                    # If we have space for more neighbors
                    if len(neighbor_dists[node_i]) < n_neighbors:
                        neighbor_dists[node_i].append(epsilon)
                        neighbor_inds[node_i].append(node_j)
                    if len(neighbor_dists[node_j]) < n_neighbors:
                        neighbor_dists[node_j].append(epsilon)
                        neighbor_inds[node_j].append(node_i)

            merged_component = merge_components(component_dict[i], component_dict[j])
            for node in merged_component.nodes:
                component_dict[node] = merged_component
            size_of_component = len(component_dict[i].nodes)
            if size_of_component > max_comp_size:
                max_comp_size = size_of_component
        if max_comp_size == num_points:
            break

    return {
        '_knn_indices': np.array(neighbor_inds),
        '_knn_dists': np.array(neighbor_dists),
        '_all_dists': np.array(density_connections)
    }


class Tree:
    def __init__(self, dist):
        self.dist = dist
        self.left_tree = None
        self.right_tree = None
        self.label = None
        self.point_id = None

    def set_left_tree(self, left):
        self.left_tree = left

    def set_right_tree(self, right):
        self.right_tree = right

    def print_tree(self, stack_depth=0):
        if self.left_tree is not None:
            self.left_tree.print_tree(stack_depth + 1)
        print('  ' * stack_depth + str(self.dist))
        if self.right_tree is not None:
            self.right_tree.print_tree(stack_depth + 1)

    def __len__(self):
        total = 0
        if self.left_tree:
            total += len(self.left_tree)
        if self.right_tree:
            total += len(self.right_tree)
        return total + 1

def make_tree(all_dists, labels, point_ids):
    # FIXME -- the right way to do this is to build the tree up while we're connecting the components
    largest_dist = np.max(all_dists)
    root = Tree(largest_dist)
    if largest_dist == 0:
        root.label = labels[0]
        root.point_id = point_ids[0]
        return root

    left_inds = np.where(all_dists[0] == largest_dist)[0]
    left_split = all_dists[left_inds][:, left_inds]
    left_labels = labels[left_inds]
    left_point_ids = point_ids[left_inds]
    root.set_left_tree(make_tree(left_split, left_labels, left_point_ids))

    right_inds = np.where(all_dists[0] != largest_dist)[0]
    right_split = all_dists[right_inds][:, right_inds]
    right_labels = labels[right_inds]
    right_point_ids = point_ids[right_inds]
    root.set_right_tree(make_tree(right_split, right_labels, right_point_ids))

    return root

class Point:
    def __init__(self, coordinates, label, point_id):
        self.coordinates = coordinates
        self.label = label
        self.point_id = point_id

def make_1d_embedding(root, embedding, coordinates):
    # FIXME -- don't need to pass embedding through. Just need to maintain max coordinate
    if root.label is not None:
        return set([Point(coordinates, root.label, root.point_id)])

    left_embedding = make_1d_embedding(root.left_tree, embedding, coordinates)
    embedding = left_embedding | embedding
    coordinates = max([e.coordinates for e in embedding])

    right_embedding = make_1d_embedding(root.right_tree, embedding, coordinates+root.dist)
    embedding = embedding | right_embedding

    return embedding


### 2d-embedding helper functions###
def connect_clusters(left_embedding, right_embedding, x_or_y, dist, rotate):
    if rotate:
        if len(left_embedding) > 1:
            left_embedding = rotate_embedding(left_embedding)
        if len(right_embedding) > 1:
            right_embedding = rotate_embedding(right_embedding)
        x_or_y = 0

    left_embedding, right_embedding = _point_align_clusters(left_embedding, right_embedding, x_or_y, dist)

    # Now both the left and right embeddings have a single point at (0, 0)
    # So we separate them by the root distance in the left/right or up/down directions
    for point in right_embedding:
        point.coordinates[x_or_y] += dist

    embedding = left_embedding | right_embedding
    return embedding

def rotate_embedding(embedding):
    points = np.array([p.coordinates for p in embedding]).T
    mean = np.mean(points, axis=1)
    cov = np.cov(points - np.expand_dims(mean, axis=1))
    eig_vals, eig_vecs = np.linalg.eig(cov)
    min_var_direction = eig_vecs[np.argmin(eig_vals)]
    alpha = np.arctan(min_var_direction[1] / min_var_direction[0])
    c, s = np.cos(alpha), np.sin(alpha)
    rot_matrix = np.array([[c, -s], [s, c]]) # Rotates axis of lowest variance onto x-axis
    for point in embedding:
        point.coordinates = np.dot(rot_matrix, np.array(point.coordinates).T - mean)

    return embedding

def _point_align_clusters(left_embedding, right_embedding, x_or_y, dist):
    # Set left embedding's right-most or top-most point to be at 0
    max_pos = -np.inf
    slide = 0
    for point in left_embedding:
        if point.coordinates[x_or_y] > max_pos:
            max_pos = point.coordinates[x_or_y]
            slide = point.coordinates[1 - x_or_y]
    for point in left_embedding:
        point.coordinates[x_or_y] -= max_pos
        point.coordinates[1 - x_or_y] -= slide

    # Set right embedding's left-most or bottom-most point to be at 0
    min_pos = np.inf
    slide = 0
    for point in right_embedding:
        if point.coordinates[x_or_y] < min_pos:
            min_pos = point.coordinates[x_or_y]
            slide = point.coordinates[1 - x_or_y]
    for point in right_embedding:
        point.coordinates[x_or_y] -= min_pos
        point.coordinates[1 - x_or_y] -= slide

    return left_embedding, right_embedding

def make_2d_embedding(root, depth=0, rotate=True):
    if root.label is not None:
        return set([Point([0, 0], root.label, root.point_id)])

    left_embedding = make_2d_embedding(root.left_tree, depth=depth+1)
    right_embedding = make_2d_embedding(root.right_tree, depth=depth+1)

    # Even (odd) depth specifies that we connect clusters left-right (up-down)
    x_or_y = depth % 2
    embedding = connect_clusters(
        left_embedding,
        right_embedding,
        x_or_y,
        root.dist,
        rotate=rotate
    )
    return embedding

def make_dc_embedding(dc_dists, labels):
    num_points = int(dc_dists.shape[0])
    point_ids = np.arange(num_points)
    root = make_tree(np.copy(dc_dists), labels, point_ids)

    # embedding = set([])
    # embedding = make_1d_embedding(root, embedding, coordinates=0)
    embedding = make_2d_embedding(root)
    point_ids = [e.point_id for e in embedding]
    sort_inds = np.argsort(point_ids)
    points = np.array([e.coordinates for e in embedding])[sort_inds]
    # If 1-d, need to add a second dimension for hardcoded distance and plotting functions
    # points = np.stack([points, np.zeros_like(points)], axis=1)
    labels = np.array([e.label for e in embedding])[sort_inds]

    # Test that embedding preserves density-connectedness
    # nn_dict = get_nearest_neighbors(points, n_neighbors=15)
    # unique_orig = np.sort(np.unique(dc_dists))
    # unique_new = np.sort(np.unique(nn_dict['_all_dists']))
    # print(np.allclose(unique_orig, unique_new))
    return points, labels
