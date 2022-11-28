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

def get_nearest_neighbors(points, n_neighbors, min_points=5, **kwargs):
    """
    We define the distance from x_i to x_j as min(max(P(x_i, x_j))), where 
        - P(x_i, x_j) is any path from x_i to x_j
        - max(P(x_i, x_j)) is the largest edge weight in the path
        - min(max(P(x_i, x_j))) is the smallest largest edge weight
    """
    @numba.njit(fastmath=True, parallel=True)
    def get_dist_matrix(points, D, dim, num_points):
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

    num_points = int(points.shape[0])
    dim = int(points.shape[1])
    density_connections = np.zeros([num_points, num_points])
    D = np.zeros([num_points, num_points])
    D = get_dist_matrix(points, D, dim, num_points)
    if min_points > 1:
        if min_points > num_points:
            raise ValueError('Min points cannot exceed the size of the dataset')
        # Get reachability for each point with respect to min_points parameter
        reach_dists = np.sort(D, axis=1)
        reach_dists = reach_dists[:, min_points]

        # Make into an NxN matrix
        reach_dists_i, reach_dists_j = np.meshgrid(reach_dists, reach_dists)

        # Zero out the diagonal so that it's a distance metric
        diag_mask = np.ones([num_points, num_points]) - np.eye(num_points)
        reach_dists_i *= diag_mask
        reach_dists_j *= diag_mask

        # Take max of reach_i, D_ij, reach_j
        D = np.stack([D, reach_dists_i, reach_dists_j], axis=-1)
        D = np.max(D, axis=-1)

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
