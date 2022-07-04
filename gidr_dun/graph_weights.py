import time

import numba
import numpy as np
import scipy
from tqdm.auto import tqdm
from .numba_optimizers.numba_utils import get_dist_matrix

from . import utils

SMOOTH_K_TOLERANCE = 1e-5
MIN_K_DIST_SCALE = 1e-3
NPY_INFINITY = np.inf

class Component:
    def __init__(self, nodes, comp_id):
        self.nodes = set(nodes)
        self.comp_id = comp_id

def merge_components(c_i, c_j):
    merged_list = c_i.nodes.union(c_j.nodes)
    return Component(merged_list, c_i.comp_id)

def get_nearest_neighbors(points, n_neighbors):
    """
    We define the distance from x_i to x_j as min(max(P(x_i, x_j))), where 
        - P(x_i, x_j) is any path from x_i to x_j
        - max(P(x_i, x_j)) is the largest edge weight in the path
        - min(max(P(x_i, x_j))) is the smallest largest edge weight
    """
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
                    # If the current neighbor is equidistant to the other ones
                    #elif np.abs(epsilon - neighbor_dists[node_i][-1]) < 0.0001:
                    #    neighbor_dists[node_i].append(epsilon)
                    #    neighbor_inds[node_i].append(node_j)
                    if len(neighbor_dists[node_j]) < n_neighbors:
                        neighbor_dists[node_j].append(epsilon)
                        neighbor_inds[node_j].append(node_i)
                    #elif np.abs(epsilon - neighbor_dists[node_j][-1]) < 0.0001:
                    #    neighbor_dists[node_j].append(epsilon)
                    #    neighbor_inds[node_j].append(node_i)

            merged_component = merge_components(component_dict[i], component_dict[j])
            for node in merged_component.nodes:
                component_dict[node] = merged_component
            size_of_component = len(component_dict[i].nodes)
            if size_of_component > max_comp_size:
                max_comp_size = size_of_component
        if max_comp_size == num_points:
            break

    return neighbor_inds, neighbor_dists, density_connections


@numba.njit(
    locals={
        "psum": numba.types.float32,
        "lo": numba.types.float32,
        "mid": numba.types.float32,
        "hi": numba.types.float32,
    },
    fastmath=True,
)  # benchmarking `parallel=True` shows it to *decrease* performance
def smooth_knn_dist(
        distances,
        k,
        n_iter=64,
        local_connectivity=1.0,
        bandwidth=1.0,
        pseudo_distance=True,
):
    # Not my code -- Andrew
    rho = np.zeros(distances.shape[0], dtype=np.float32)
    sigmas = np.zeros(distances.shape[0], dtype=np.float32)

    mean_distances = np.mean(distances)

    for i in range(distances.shape[0]):
        k = np.count_nonzero(distances[i])
        target = np.log2(k) * bandwidth
        # Calculate rho values
        ith_distances = distances[i]
        non_zero_dists = ith_distances[ith_distances > 0.0]
        if non_zero_dists.shape[0] >= local_connectivity:
            index = int(np.floor(local_connectivity))
            interpolation = local_connectivity - index
            if index > 0:
                rho[i] = non_zero_dists[index - 1]
                if interpolation > SMOOTH_K_TOLERANCE:
                    rho[i] += interpolation * (
                            non_zero_dists[index] - non_zero_dists[index - 1]
                    )
            else:
                rho[i] = interpolation * non_zero_dists[0]
        elif non_zero_dists.shape[0] > 0:
            rho[i] = np.max(non_zero_dists)

        lo = 0.0
        hi = NPY_INFINITY
        mid = 1.0
        # Calculating sigma values
        for n in range(n_iter):
            psum = 0.0
            for j in range(1, distances.shape[1]):
                if pseudo_distance:
                    d = distances[i, j] - rho[i]
                else:
                    d = distances[i, j]

                if d > 0:
                    psum += np.exp(-(d / mid))
                elif d == 0:
                    psum += 1.0
                else:
                    continue

            if np.fabs(psum - target) < SMOOTH_K_TOLERANCE:
                break

            if psum > target:
                hi = mid
                mid = (lo + hi) / 2.0
            else:
                lo = mid
                if hi == NPY_INFINITY:
                    mid *= 2
                else:
                    mid = (lo + hi) / 2.0
        sigmas[i] = mid

        # TODO: This is very inefficient, but will do for now. FIXME
        if rho[i] > 0.0:
            mean_ith_distances = np.mean(ith_distances)
            if sigmas[i] < MIN_K_DIST_SCALE * mean_ith_distances:
                # ANDREW - this never gets called on mnist
                sigmas[i] = MIN_K_DIST_SCALE * mean_ith_distances
        else:
            # ANDREW - this never gets called on mnist either
            if sigmas[i] < MIN_K_DIST_SCALE * mean_distances:
                sigmas[i] = MIN_K_DIST_SCALE * mean_distances

    return sigmas, rho

def nearest_neighbors(
        X,
        n_neighbors,
        metric,
        euclidean,
        random_state,
        num_threads=-1,
        verbose=False,
):
    if verbose:
        print(utils.ts(), "Finding Nearest Neighbors")
    num_points = len(X)

    # Sample n_neighbors nearest neighbors rather than using all of the calculated ones
    knn_indices, knn_dists, D = get_nearest_neighbors(X, n_neighbors)
    # for i in range(num_points):
    #     i_knn_inds = np.array(knn_indices[i])
    #     i_knn_dists = np.array(knn_dists[i])
    #     subsample_indices = np.random.choice(
    #         np.arange(len(i_knn_inds)),
    #         size=n_neighbors,
    #         replace=False
    #     )
    #     knn_dists[i] = i_knn_dists[subsample_indices]
    #     sort_inds = np.argsort(knn_dists[i])
    #     knn_dists[i] = knn_dists[i][sort_inds]
    #     knn_indices[i] = i_knn_inds[subsample_indices][sort_inds]
    np_indices = np.array(knn_indices)
    np_dists = np.array(knn_dists)
    return np_indices, np_dists, np.sqrt(D)

    # # Use all the nearest neighbors that got calculated, even if more than n_neighbors
    # knn_indices, knn_dists = get_nearest_neighbors(X, n_neighbors)
    # max_length = max([len(i) for i in knn_indices])
    # np_indices = np.zeros([num_points, max_length]) - 1
    # np_dists = np.zeros([num_points, max_length]) - 1
    # for i in range(num_points):
    #     for j in range(len(knn_indices[i])):
    #         np_indices[i, j] = knn_indices[i][j]
    #         np_dists[i, j] = knn_dists[i][j]
    # return np_indices, np_dists


@numba.njit(
    locals={
        "knn_dists": numba.types.float32[:, ::1],
        "sigmas": numba.types.float32[::1],
        "rhos": numba.types.float32[::1],
        "val": numba.types.float32,
    },
    # FIXME FIXME
    # parallel=True,
    fastmath=True,
)
def compute_membership_strengths(
        knn_indices,
        knn_dists,
        sigmas,
        rhos,
        return_dists=False,
        bipartite=False,
        pseudo_distance=True,
):
    # Not my code -- Andrew
    n_samples = knn_indices.shape[0]
    n_neighbors = knn_indices.shape[1]

    rows = np.zeros(knn_indices.size, dtype=np.int32)
    cols = np.zeros(knn_indices.size, dtype=np.int32)
    vals = np.zeros(knn_indices.size, dtype=np.float32)
    if return_dists:
        dists = np.zeros(knn_indices.size, dtype=np.float32)
    else:
        dists = None

    for i in range(n_samples):
        for j in range(knn_indices.shape[1]):
            if knn_indices[i, j] == -1:
                continue  # We didn't get the full knn for i
            # If applied to an adjacency matrix points shouldn't be similar to themselves.
            # If applied to an incidence matrix (or bipartite) then the row and column indices are different.
            if (bipartite == False) & (knn_indices[i, j] == i):
                val = 0.0
            elif knn_dists[i, j] - rhos[i] <= 0.0 or sigmas[i] == 0.0:
                val = 1.0
            else:
                # ANDREW - this is where the rhos are subtracted for the UMAP
                # pseudo distance metric
                # The sigmas are equivalent to those found for tSNE
                if pseudo_distance:
                    val = np.exp(-((knn_dists[i, j] - rhos[i]) / (sigmas[i])))
                else:
                    val = np.exp(-((knn_dists[i, j]) / (sigmas[i])))

            rows[i * n_neighbors + j] = i
            cols[i * n_neighbors + j] = knn_indices[i, j]
            vals[i * n_neighbors + j] = val
            if return_dists:
                dists[i * n_neighbors + j] = knn_dists[i, j]

    return rows, cols, vals, dists


def fuzzy_simplicial_set(
        X,
        n_neighbors,
        random_state,
        metric,
        knn_indices=None,
        knn_dists=None,
        verbose=False,
        return_dists=True,
        pseudo_distance=True,
        euclidean=True,
        tsne_symmetrization=False,
        gpu=False,
):
    # Not my code -- Andrew
    if knn_indices is None or knn_dists is None:
        knn_indices, knn_dists = nearest_neighbors(
            X,
            n_neighbors,
            metric,
            euclidean,
            random_state,
            verbose=verbose,
        )
    knn_dists = knn_dists.astype(np.float32)
    n_neighbors = int(knn_dists.shape[1])

    sigmas, rhos = smooth_knn_dist(
        knn_dists,
        float(n_neighbors),
        pseudo_distance=pseudo_distance,
    )

    rows, cols, vals, dists = compute_membership_strengths(
        knn_indices,
        knn_dists,
        sigmas,
        rhos,
        return_dists,
        pseudo_distance=pseudo_distance,
    )

    result = scipy.sparse.coo_matrix(
        (vals, (rows, cols)), shape=(X.shape[0], X.shape[0])
    )
    result.eliminate_zeros()

    # UMAP symmetrization:
    # Symmetrized = A + A^T - pointwise_mul(A, A^T)
    # TSNE symmetrization:
    # Symmetrized = (A + A^T) / 2
    transpose = result.transpose()
    if not tsne_symmetrization:
        prod_matrix = result.multiply(transpose)
        result = result + transpose - prod_matrix
    else:
        result = (result + transpose) / 2

    result.eliminate_zeros()

    if return_dists is None:
        return result, sigmas, rhos
    if return_dists:
        dmat = scipy.sparse.coo_matrix(
            (dists, (rows, cols)), shape=(X.shape[0], X.shape[0])
        )
        dists = dmat.maximum(dmat.transpose()).todok()
    else:
        dists = None

    return result, sigmas, rhos, dists
