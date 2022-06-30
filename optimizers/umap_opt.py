import os
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
from utils.helper_functions import get_sq_dist_mat, get_vecs, get_sigmas, get_sigmas, get_rhos, zero_cols

from optimizers.base_opt_class import DROptimizer

class UMAPOptimizer(DROptimizer):
    def __init__(
        self,
        lr=0.05,
        n_epochs=500,
        use_rhos=True,
        num_sigma_samples=-1,
        min_p_value=0.0,
        *args,
        **kwargs
    ):
        self.use_rhos = use_rhos
        self.num_sigma_samples = num_sigma_samples
        self.min_p_value = min_p_value
        super(UMAPOptimizer, self).__init__(lr=lr, n_epochs=n_epochs, *args, **kwargs)

    def get_pairwise_mat(self, dataset):
        """
        UMAP's pairwise matrix is the distance matrix
        """
        return get_sq_dist_mat(dataset)

    def high_dim_kernel(self, min_p_value=0.0):
        if self.pairwise_x_mat is None:
            self.pairwise_x_mat = self.get_pairwise_mat(self.x)
        sigmas = np.expand_dims(
            get_sigmas(
                self.pairwise_x_mat,
                target=2,
                num_samples=self.num_sigma_samples
            ),
            -1
        )
        if self.use_rhos:
            # FIXME -- using rhos currently does NOT converge
            rhos = get_rhos(self.pairwise_x_mat)
        else:
            rhos = np.zeros(self.pairwise_x_mat.shape[0], dtype=np.float32)
        rhos = np.expand_dims(rhos, -1)
        p = np.exp(-1 * (self.pairwise_x_mat - rhos) / sigmas)
        sym = (p + p.T) / 2
        sym[sym < self.min_p_value] = 0
        return sym

    def low_dim_kernel(self):
        kernel = 1 / (1 + self.pairwise_y_mat)
        return kernel

    def get_grads(self):
        self.pairwise_y_mat = self.get_pairwise_mat(self.y)
        self.Q = self.low_dim_kernel()
        dQ = -0.5 * np.square(self.Q)

        deriv_of_sim = dQ @ (self.Q - self.P)

        degree_in = np.sum(deriv_of_sim, axis=0)
        degree_out = np.sum(deriv_of_sim, axis=1)
        laplacian = (np.diag(degree_in) + np.diag(degree_out)) - deriv_of_sim - deriv_of_sim.T

        grads = -4 * laplacian @ self.y
        return grads / self.n_points

    # FIXME -- figure out how to do nearest neighbor and repulsion masking in matrix formulation
    def high_dim_mask(self, n_neighbors=15):
        nearest_neighbors = np.argsort(self.pairwise_x_mat, axis=0)
        return (nearest_neighbors < n_neighbors).astype(np.int32)

    def repulsion_mask(self, rep_forces, num_repulsions=5):
        mask = np.zeros_like(rep_forces)
        points_range = np.arange(self.n_points)
        for i in range(self.n_points):
            locs = np.random.permutation(points_range)[:num_repulsions]
            mask[i, locs] = 1
        return mask

    def get_kl_grads():
        raise NotImplementedError
