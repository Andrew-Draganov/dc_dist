import os, csv, glob
import numpy as np
import os
from tqdm.auto import tqdm
from sklearn.datasets import make_swiss_roll
from mnist import MNIST
from PIL import Image

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

def make_circles(n_samples, noise=0., radii=None, thicknesses=None, labels_as_radius=False):
    assert radii is not None
    if thicknesses is None:
        thicknesses = [0 for r in radii]
    n_circles = len(radii)
    points_per_circle = n_samples // n_circles

    circles = [[] for i in range(n_circles)]
    for circle in range(n_circles):
        linspace = np.linspace(0, 2 * np.pi, points_per_circle, endpoint=False)
        point_norms = (np.random.rand(points_per_circle) - 0.5) * thicknesses[circle] + radii[circle]
        x_vals = np.cos(linspace) * point_norms
        y_vals = np.sin(linspace) * point_norms
        circles[circle] = np.stack((x_vals, y_vals), axis=-1)
        circles[circle] += np.random.normal(0, scale=noise, size=circles[circle].shape)

    points = np.concatenate(circles, axis=0)
    if labels_as_radius:
        labels = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
    else:
        labels = np.hstack([np.ones([points_per_circle]) * i for i in range(n_circles)])
    return points, labels


def load_coil100_data(directory=None):
    """
    This is using the coil100 dataset available on Kaggle at https://www.kaggle.com/datasets/jessicali9530/coil100
    Using it requires manually unzipping it into a directory
    """
    if directory is None:
        directory = os.path.join('data', 'coil-100')
    pickled_path = os.path.join(directory, 'pickled_coil.npy')
    if os.path.exists(pickled_path):
        dataset = np.load(pickled_path, allow_pickle=True)[()]
        return dataset['points'], dataset['labels']

    print('Could not find pickled dataset at {}. Loading from png files and pickling...'.format(pickled_path))
    filelist = glob.glob(os.path.join(directory, '*.png'))
    if not filelist:
        raise ValueError('Coil 100 data directory {} is empty!'.format(directory))

    points = np.zeros([7200, 128, 128, 3])
    labels = np.zeros([7200])
    for i, fname in tqdm(enumerate(filelist)):
        image = np.array(Image.open(fname))
        points[i] = image

        image_name = os.path.split(fname)[-1]
        # This assumes that your images are named objXY__i.png
        #   where XY are the class label and i is the picture angle
        class_label = [int(c) for c in image_name[:6] if c.isdigit()]
        class_label = np.array(class_label[::-1])
        digit_powers = np.power(10, np.arange(len(class_label)))
        class_label = np.sum(class_label * digit_powers)
        labels[i] = class_label

    points = np.reshape(points, [7200, -1])
    np.save(pickled_path, {'points': points, 'labels': labels})
    return points, labels
    
def load_dsnesynth_data(directory=None):

    if directory is None:
        directory = os.path.join('data', 'synth')
    file_path = os.path.join(directory, 'synth_data_10400_5_2_1.npy')
    dataset = np.load(file_path)[()]
    points = dataset[:,:-1]
    labels = dataset[:,-1]
    print(np.unique(labels))
    return dataset[:,:-1], dataset[:,-1]

def resample_dim(desired_dim, points):
    """
    Up/Down sample to desired dimensionality
    """
    dim = int(points.shape[1])
    while dim < desired_dim:
        points = np.concatenate([points, points], axis=-1)
        dim = int(points.shape[1])
    random_perm = np.random.permutation(np.arange(dim))
    points = points[:, random_perm]
    points = points[:, :desired_dim]
    return points

def subsample_points(points, labels, num_classes, points_per_class, class_list=None):
    if class_list is None:
        unique_classes = np.unique(labels)
        if num_classes > len(unique_classes):
            raise ValueError('Cannot subsample to {} classes when only have {} available'.format(num_classes, unique_classes.shape[0]))
        class_list = np.random.choice(unique_classes, num_classes, replace=False)

    per_class_samples = [np.where(labels == sampled_class)[0] for sampled_class in class_list]
    min_per_class = min([len(s) for s in per_class_samples])
    per_class_samples = [s[:min_per_class] for s in per_class_samples]
    sample_indices = np.squeeze(np.stack([per_class_samples], axis=-1))
    total_points_per_class = int(sample_indices.shape[-1])
    if points_per_class < total_points_per_class:
        stride_rate = float(total_points_per_class) / points_per_class
        class_subsample_indices = np.arange(0, total_points_per_class, step=stride_rate).astype(np.int32)
        sample_indices = sample_indices[:, class_subsample_indices]

    sample_indices = np.reshape(sample_indices, -1)
    points = points[sample_indices]
    labels = labels[sample_indices]
    return points, labels


def get_dataset(
    data_name,
    normalize=True,
    desired_dim=-1,
    num_classes=2,
    points_per_class=1000,
    class_list=None
):
    if data_name == 'mnist':
        points, labels = load_mnist()
    elif data_name == 'synth':
        points, labels = load_dsnesynth_data()
    elif data_name == 'coil':
        points, labels = load_coil100_data()
    elif data_name == 'coil_20':
        pass
        # FIXME
    else:
        raise ValueError("Unsupported dataset")

    if desired_dim > 0:
        points = resample_dim(desired_dim, points)

    points, labels = subsample_points(
        points,
        labels,
        num_classes,
        points_per_class,
        class_list
    )
    num_samples = int(points.shape[0])
    points = np.reshape(points, [num_samples, -1])

    if normalize:
        points = np.array(points) / np.max(points).astype(np.float32)

    return points, labels
