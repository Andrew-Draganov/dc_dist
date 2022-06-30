import os, csv, glob
import numpy as np
import os
from tqdm import tqdm
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

def upsample_dataset(num_points, points, labels):
    # If the dataset doesn't have as many points as we want, make copies of the
    #   dataset until it does
    # Note -- this is only for timing purposes and resulting embeddings may be bogus
    assert int(points.shape[0]) == int(labels.shape[0])
    num_samples = int(points.shape[0])
    while num_points > num_samples:
        # add 1 to each dimension of the points when making copies of dataset
        #   - want to make sure that optimization doesn't get arbitrarily faster
        #     with identical copies of points
        points = np.concatenate([points, points+1], axis=0)
        labels = np.concatenate([labels, labels], axis=0)
        num_samples = int(points.shape[0])
        
    return points, labels

def resample_dim(desired_dim, points):
    dim = int(points.shape[1])
    while dim < desired_dim:
        points = np.concatenate([points, points], axis=-1)
        dim = int(points.shape[1])
    random_perm = np.random.permutation(np.arange(dim))
    points = points[:, random_perm]
    points = points[:, :desired_dim]
    return points

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
    for i, fname in enumerate(filelist):
        image = np.array(Image.open(fname))
        points[i] = image

        image_name = os.path.split(fname)[-1]
        # This assumes that your images are named objXY__i.png
        #   where XY are the class label and i is the picture angle
        class_label = [int(c) for c in image_name[:5] if c.isdigit()]
        class_label = np.array(class_label[::-1])
        digit_powers = np.power(10, np.arange(len(class_label)))
        class_label = np.sum(class_label * digit_powers)
        labels[i] = class_label

    points = np.reshape(points, [7200, -1])
    np.save(pickled_path, {'points': points, 'labels': labels})
    return points, labels

def get_dataset(data_name, num_points, normalize=True, desired_dim=-1):
    if data_name == 'mnist':
        points, labels = load_mnist()
    elif data_name == 'swiss_roll':
        points, _ = make_swiss_roll(n_samples=num_points, noise=0.01)
        labels = np.arange(num_points)
    elif data_name == 'google_news':
        # To run this dataset, download https://data.world/jaredfern/googlenews-reduced-200-d
        #   and place it into the directory 'data'
        file = open(os.path.join('data', 'gnews_mod.csv'), 'r', encoding="utf-8")
        reader = csv.reader(file)
        if num_points < 0:
            num_points = 350000
        num_points = min(num_points, 350000)
        points = np.zeros([num_points, 200])
        for i, line in tqdm(enumerate(reader), total=num_points):
            # First line is column descriptions
            if i == 0:
                continue
            if i > num_points:
                break
            for j, element in enumerate(line[1:]): # First column is string text
                points[i-1, j] = float(element)
        labels = np.ones([num_points])
    elif data_name == 'coil':
        points, labels = load_coil100_data()
    elif data_name == 'coil_20':
        pass
        # FIXME
    else:
        raise ValueError("Unsupported dataset")

    if desired_dim > 0:
        points = resample_dim(desired_dim, points)

    if num_points < 0:
        num_points = int(points.shape[0])
    points, labels = upsample_dataset(num_points, points, labels)
    num_samples = int(points.shape[0])
    downsample_stride = int(float(num_samples) / num_points)
    points, labels = points[::downsample_stride], labels[::downsample_stride]
    num_samples = int(points.shape[0])
    points = np.reshape(points, [num_samples, -1])

    # FIXME - do we need this normalization?
    if normalize:
        points = np.array(points) / np.max(points).astype(np.float32)

    return points, labels
