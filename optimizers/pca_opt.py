import os
from tqdm import tqdm
import numpy as np
from utils.get_data import get_dataset
from utils.helper_functions import zero_row_cols, zero_cols, get_vecs, get_gram_mat

from optimizers.base_opt_class import DROptimizer

class PCAOptimizer(DROptimizer):
    def __init__(self, lr=0.01, n_epochs=150, plot_rate=50, *args, **kwargs):
        super(PCAOptimizer, self).__init__(
            lr=lr,
            n_epochs=n_epochs,
            plot_rate=plot_rate,
            *args,
            **kwargs
        )

    def get_grads(self):
        self.Q = self.low_dim_kernel()
        kernel_diff = self.Q - self.P
        grads = kernel_diff @ zero_cols(self.y)
        return -1 / self.n_points * grads

    def get_pairwise_mat(self, dataset):
        """
        PCA's pairwise matrix is the Gram matrix
        """
        return get_gram_mat(dataset)

    def high_dim_kernel(self):
        print('Getting high dim kernel matrix...')
        # Technically the kernel is row/col centered but the centering matrices
        #   cancel out when calculating the gradient
        # So we don't center the Gram matrices here to save on computations
        if self.pairwise_x_mat is None:
            return self.get_pairwise_mat(self.x)
        else:
            return self.pairwise_x_mat

    def low_dim_kernel(self):
        return self.get_pairwise_mat(self.y)

    def get_kl_grads():
        raise NotImplementedError
