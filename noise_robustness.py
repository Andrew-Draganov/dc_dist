import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from distance_metric import get_nearest_neighbors

def eval_noise():
    plt.rcParams.update({'text.usetex': True, 'font.size': 18})
    class_size = 300
    mu_list = [1, 3, 5, 7, 9]
    # noise_size_list = list(range(0, 100, 10))
    # FIXME
    num_iterations = 10

    noise_size_list_a = list(np.arange(1, 2.01, 0.1))
    min_inter_dists = np.zeros((num_iterations, len(mu_list), len(noise_size_list_a)))
    mean_inter_dists = np.zeros((num_iterations, len(mu_list), len(noise_size_list_a)))
    mean_intra_dists = np.zeros((num_iterations, len(mu_list), len(noise_size_list_a)))
    max_intra_dists = np.zeros((num_iterations, len(mu_list), len(noise_size_list_a)))

    for i, iteration in tqdm(enumerate(range(num_iterations)), total=num_iterations):
        for j, mu in enumerate(mu_list):
            for k, noise_size in enumerate(noise_size_list_a):
                data = np.concatenate(
                    [
                        np.random.multivariate_normal([-3, 0], [[noise_size, 0], [0, noise_size]], size=class_size),
                        np.random.multivariate_normal([3, 0], [[noise_size, 0], [0, noise_size]], size=class_size)
                    ],
                    axis=0
                )

                dists = get_nearest_neighbors(data, 15, min_points=mu)['_all_dists']
                min_inter_dists[i, j, k] = np.min(dists[:class_size, class_size:2*class_size])
                mean_intra_dists[i, j, k] = np.mean(dists[:class_size, :class_size])


    # noise_size_list_b = list(np.arange(0.6, 1.61, 0.2))
    # for i, iteration in tqdm(enumerate(range(num_iterations)), total=num_iterations):
    #     for j, mu in enumerate(mu_list):
    #         for k, noise_size in enumerate(noise_size_list_b):
    #             data = np.concatenate(
    #                 [
    #                     np.random.multivariate_normal([-3, 0], [[noise_size, 0], [0, noise_size]], size=class_size),
    #                     np.random.multivariate_normal([3, 0], [[noise_size, 0], [0, noise_size]], size=class_size)
    #                 ],
    #                 axis=0
    #             )

    #             dists = get_nearest_neighbors(data, 15, min_points=mu)['_all_dists']
    #             mean_inter_dists[i, j, k] = np.mean(dists[:class_size, class_size:2*class_size])
    #             max_intra_dists[i, j, k] = np.max(dists[:class_size, :class_size])

    min_mean = np.mean((min_inter_dists - mean_intra_dists) / mean_intra_dists, axis=0)
    # mean_max = np.mean((mean_inter_dists - max_intra_dists) / max_intra_dists, axis=0)

    axis = plt.gca()
    colors = ['skyblue', 'steelblue', 'royalblue', 'darkslateblue', 'navy']
    for i in range(len(min_mean)):
        plt.plot(noise_size_list_a, min_mean[i], color=colors[i])
    plt.plot([1.0, 2.0], [0, 0], c='r', linestyle='dashed', alpha=0.7)
    color_handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colors]
    color_legend = axis.legend(color_handles, [r'$\mu=$ {}'.format(str(mu)) for mu in mu_list], loc='upper right')
    axis.add_artist(color_legend)
    axis.set_xlabel('Diagonal variance')
    # axes[0].set_yscale('log')
    axis.set_ylabel(r'$(\min_{inter} - \mu_{intra}) / \mu_{intra}$')

    # sub_colors = ['skyblue', 'steelblue', 'royalblue', 'darkslateblue', 'navy']
    # for i in range(len(mean_max)):
    #     axes[1].plot(noise_size_list_b, mean_max[i], color=sub_colors[i])
    # axes[1].plot([0.6, 1.6], [0, 0], c='r', linestyle='dashed', alpha=0.7)
    # color_handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in sub_colors]
    # color_legend = axes[1].legend(color_handles, [r'$\mu=$ {}'.format(str(mu)) for mu in mu_list], loc='upper right')
    # axes[1].add_artist(color_legend)
    # # axes[1].set_yscale('log')
    # axes[1].set_title(r'$(\mu_{inter} - \max_{intra}) / \max_{intra}$')
    plt.show()

if __name__ == '__main__':
    eval_noise()
