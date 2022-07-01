import numpy as np
from tqdm.auto import tqdm

def zero_row_cols(mat, divide_by_n=False):
    # Apply centering matrix on either side
    mat -= np.mean(mat, axis=1, keepdims=True)
    mat -= np.mean(mat, axis=0, keepdims=True)
    if divide_by_n:
        n_points = int(mat.shape[0])
        mat /= n_points
    return mat

def zero_cols(mat, divide_by_n=False):
    # Apply centering matrix on either side
    mat -= np.mean(mat, axis=0, keepdims=True)
    if divide_by_n:
        n_points = int(mat.shape[0])
        mat /= n_points
    return mat

def get_vecs(array):
    return np.expand_dims(array, 0) - np.expand_dims(array, 1)

# FIXME -- numba njit this!
def get_gram_mat(data):
    gram_mat = np.sum(
        np.expand_dims(data, 0) * np.expand_dims(data, 1),
        axis=-1
    )
    return gram_mat

def get_sq_dist_mat(data):
    dist_mat = np.sum(
        np.square(
            np.expand_dims(data, 0) - np.expand_dims(data, 1)
        ),
        axis=-1
    )

    return dist_mat

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def get_rhos(distances):
    rhos = np.zeros(distances.shape[0], dtype=np.float32)
    for i in tqdm(range(distances.shape[0]), desc='Finding rho values'):
        min_dist = -1
        for j in range(distances.shape[0]):
            if j == i:
                continue
            if distances[i, j] < min_dist or min_dist < 0:
                min_dist = distances[i, j]
        rhos[i] = min_dist
    return rhos

def get_sigmas(distances, num_samples=-1, target=5, eps=0.005, n_iter=30):
    num_distances = int(distances.shape[0])
    if num_samples <= 0:
        num_samples = num_distances
    num_samples = min(num_samples, num_distances)

    sigmas = np.ones(num_distances, dtype=np.float32)
    mean_distances = np.mean(distances)
    for i in tqdm(range(num_distances), desc='Finding sigma values'):
        lo = 0.0
        hi = np.inf
        # Assuming points are consistently distributed, initialize sigma_i to the average
        #   of previous sigmas for faster convergence
        if i > 1:
            mid = np.mean(sigmas[:i])
        else:
            mid = 1.0

        for n in range(n_iter):
            psum = 0.0
            random_subset = np.random.choice(np.arange(num_distances), num_samples)
            for j in range(num_samples):
                d = distances[i, j]
                if d > 0:
                    psum += np.exp(-(d / mid))
                else:
                    psum += 1
            psum *= float(num_distances) / num_samples

            if np.fabs(psum - target) < eps:
                break

            if psum > target:
                hi = mid
                mid = (lo + hi) / 2.0
            else:
                lo = mid
                if hi == np.inf:
                    mid *= 2
                else:
                    mid = (lo + hi) / 2.0
        sigmas[i] = mid

    return sigmas
