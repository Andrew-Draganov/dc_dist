import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from ..optimizers.sgd_low_rank_optimize import grad_descent

def sim(x, y):
    dot = np.sum(x * y)
    x_norm, y_norm = np.sum(np.square(x)), np.sum(np.square(y))
    return dot / np.sqrt(x_norm * y_norm)

def low_rank_approx_quality(x, num_samples=10):
    n_points = int(x.shape[0])
    correct_g_x = get_gram_mat(x, low_rank=False)
    output_str = 'Rank %d gradient has mean cosine sim %f with std. deviation %f'
    cosine_sims = {}
    std_devs = {}
    for d in tqdm(range(2, 4), total=2):
        cosine_sims[d] = {}
        std_devs[d] = {}
        rank = d-1
        lra_x = get_gram_mat(x, rank=rank)
        sims = np.zeros(num_samples)
        for j in range(num_samples):
            y = np.random.multivariate_normal(np.zeros([d]), np.eye(d), n_points)
            correct_grads = get_grads(y, correct_g_x)
            approx_grads = get_grads(y, lra_x, rank=rank)
            sims[j] = sim(correct_grads, approx_grads)
        cosine_sims[d]['rank d-1'] = np.mean(sims)
        std_devs[d]['rank d-1'] = np.std(sims)

        rank = d
        lra_x = get_gram_mat(x, rank=rank)
        sims = np.zeros(num_samples)
        for j in range(num_samples):
            y = np.random.multivariate_normal(np.zeros([d]), np.eye(d), n_points)
            correct_grads = get_grads(y, correct_g_x)
            approx_grads = get_grads(y, lra_x, rank=rank)
            sims[j] = sim(correct_grads, approx_grads)
        cosine_sims[d]['rank d'] = np.mean(sims)
        std_devs[d]['rank d'] = np.std(sims)

    colors = {'rank d-1': 'red', 'rank d': 'green'}
    labels = list(colors.keys())
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label]) for label in labels]
    plt.bar(
        [1 + i * 4 for i in cosine_sims],
        [cosine_sims[i]['rank d-1'] for i in cosine_sims],
        yerr=[std_devs[i]['rank d-1'] for i in std_devs],
        width=1,
        color=colors['rank d-1']
    )

    plt.bar(
        [2 + i * 4 for i in cosine_sims],
        [cosine_sims[i]['rank d'] for i in cosine_sims],
        yerr=[std_devs[i]['rank d'] for i in std_devs],
        width=1,
        color=colors['rank d']
    )

    plt.title('Cosine similarity between low-rank gradient approx and true gradient')
    plt.xlabel('Dimensionality d of Y')
    plt.ylabel('Cosine similarity')
    plt.xticks([1.5 + i * 4 for i in cosine_sims], list(cosine_sims.keys()))
    plt.legend(handles, labels, loc='lower right')
    plt.show()


def effect_of_rank_on_grad_approx(
    x,
    y,
    labels,
    n_points,
):
    image_dir = os.path.join('scratch_images')

    correct_g_x = get_gram_mat(x, low_rank=False)
    correct_solution = grad_descent(
        y,
        correct_g_x,
        labels,
        image_dir=image_dir
    )
    plt.title("Full rank approx of pca gradient")
    plt.scatter(correct_solution[:, 0], correct_solution[:, 1], c=labels)
    plt.show()
    plt.close()

    for i_rank in [1, 2, 5, 10, 50, 200]:
        lra_x = get_gram_mat(x, rank=i_rank)
        low_rank_solution = grad_descent(
            y,
            lra_x,
            labels,
            rank=i_rank,
            image_dir=image_dir
        )

        plt.title("Rank-%d approx of pca gradient" % i_rank)
        plt.scatter(low_rank_solution[:, 0], low_rank_solution[:, 1], c=labels)
        plt.show()
        plt.close()


if __name__ == '__main__':
    # FIXME -- check this
    data_size = 1000
    x, labels = get_dataset(data_size, mnist=True)
    n_points = int(x.shape[0])

    d = 2
    y = np.random.multivariate_normal(np.zeros([d]), np.eye(d), n_points)
    low_rank_approx_quality(x)
    # effect_of_rank_on_grad_approx(x, y, labels, n_points)
