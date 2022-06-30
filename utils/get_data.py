import os
import numpy as np
from mnist import MNIST
from sklearn.datasets import make_swiss_roll, make_circles, load_iris

def load_mnist():
    mnist_data_path = os.path.join('data', 'mnist')
    if not os.path.isdir(mnist_data_path):
        import subprocess
        subprocess.call(os.path.join('utils', 'mnist_get_data.sh'))

    mndata = MNIST(mnist_data_path)
    points, labels = mndata.load_training()
    points = np.array(points)
    labels = np.array(labels)
    return points, labels

def upsample_dataset(num_points, points, labels):
    """
    If the dataset doesn't have as many points as we want, make copies of the
        dataset until it does
    Add 1 to copies of the dataset so that we don't get arbitrarily many points that
        are identical to one another

    NOTE -- this makes the data bogus
    """
    assert int(points.shape[0]) == int(labels.shape[0])
    num_samples = int(points.shape[0])
    while num_points > num_samples:
        points = np.concatenate([points, points+1], axis=0)
        labels = np.concatenate([labels, labels], axis=0)
        num_samples = int(points.shape[0])
        
    return points, labels

def get_dataset(desired_size, data_str):
    if data_str == 'iris':
        iris = load_iris()
        return iris.data, iris.target

    if data_str == 'circles':
        points, labels = make_circles(n_samples=1_000, factor=0.3, noise=0.05, random_state=0)
        return points, labels

    if data_str == 'random':
        points, _ = make_swiss_roll(n_samples=desired_size, noise=0.001)
        labels = np.ones([int(points.shape[0])])
        return points, labels

    if data_str == 'mnist':
        print('Loading MNIST dataset into memory...')
        points, labels = load_mnist()
        points = points.astype(np.float32)

        # Resize MNIST dataset to be desired number of points
        if desired_size < int(points.shape[0]):
            dataset_size = int(points.shape[0])
            downsample_stride = int(float(dataset_size) / desired_size)
            points, labels = points[::downsample_stride], labels[::downsample_stride]
            n_points = int(points.shape[0])
            points = np.reshape(points, [n_points, -1])
            points /= 255.0
            points = points.astype(np.float32)
        else:
            points, labels = upsample_dataset(desired_size, points, labels)

        return points, labels

    raise ValueError('Unsupported data type %s' % data_str)
