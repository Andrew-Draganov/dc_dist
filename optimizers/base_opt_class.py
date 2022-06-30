import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from utils.helper_functions import zero_row_cols, get_vecs, get_gram_mat

class DROptimizer:
    def __init__(
        self,
        x,
        labels,
        pairwise_x_mat=None,
        rank=-1,
        dim=2,
        approx=False,
        show_intermediate=False,
        plot_rate=50,
        momentum=0.7,
        kl_div=False,
        lr=-1, # FIXME this should be class specific
        n_epochs=-1, # FIXME this should be class specific
        y_init_method='multivar_normal',
        **kwargs
    ):
        self.dim = dim
        self.approx = approx
        self.show_intermediate = show_intermediate
        self.plot_rate = plot_rate
        self.momentum = momentum
        self.kl_div = kl_div
        if lr < 0 or n_epochs < 0:
            raise ValueError('learning rate and num_epochs are class specific variables'
                             'and must be overridden in class init functions')
        self.lr = lr
        self.n_epochs = n_epochs

        self.pairwise_x_mat = pairwise_x_mat

        self.y_init_method = y_init_method

        self.set_data(x, labels)
        if rank < 0:
            self.rank = self.n_points
        else:
            self.rank = rank


    def set_data(self, x, labels):
        assert len(x.shape) == 2
        self.x = x
        self.n_points = int(x.shape[0])
        self.labels = labels

        self.init_y()

    def init_y(self):
        if self.y_init_method == 'multivar_normal':
            self.y = np.random.multivariate_normal(
            np.zeros([self.dim]),
            np.eye(self.dim),
            self.n_points
        )
        else:
            raise ValueError("Can only init Y with multivariate normal for now")

    def optimize(self):
        self.P = self.high_dim_kernel()
        forces = np.zeros_like(self.y)
        for i_epoch in tqdm(range(self.n_epochs), total=self.n_epochs):
            grads = self.get_grads()
            forces = self.momentum * forces + grads * self.lr
            self.y += forces
            if i_epoch % self.plot_rate == 0 and self.show_intermediate:
                self.show_plot()

        return self.y

    def show_plot(self):
        plt.scatter(self.y[:, 0], self.y[:, 1], c=self.labels)
        plt.show()
        plt.close()

    def get_pairwise_mat(self):
        raise NotImplementedError

    def get_grads(self):
        raise NotImplementedError

    def high_dim_kernel(self):
        raise NotImplementedError

    def low_dim_kernel(self):
        raise NotImplementedError
