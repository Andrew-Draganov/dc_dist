import numpy as np
from distance_metric import get_nearest_neighbors
from tree_plotting import plot_tree
import matplotlib.pyplot as plt

class Tree:
    def __init__(self, dist):
        self.dist = dist
        self.children = []
        self.left_tree = None
        self.right_tree = None
        self.label = None
        self.point_id = None
        self.path = ''

    def set_left_tree(self, left):
        self.left_tree = left

    def set_right_tree(self, right):
        self.right_tree = right

    def print_tree(self, stack_depth=0):
        if self.left_tree is not None:
            self.left_tree.print_tree(stack_depth + 1)
        print('  ' * stack_depth + str(self.dist))
        if self.right_tree is not None:
            self.right_tree.print_tree(stack_depth + 1)

    def __len__(self):
        return len(self.children)

class Point:
    def __init__(self, coordinates, label, point_id):
        self.coordinates = np.array(coordinates, dtype=np.float32)
        self.label = label
        self.point_id = point_id

def _make_tree(all_dists, labels, point_ids, path=''):
    # FIXME -- the right way to do this is to build the tree up while we're connecting the components
    largest_dist = np.max(all_dists)
    root = Tree(largest_dist)
    # FIXME -- this will break if multiple copies of the same point. Need to first check for equal points
    if largest_dist == 0:
        root.label = labels[0]
        root.point_id = point_ids[0]
        root.path = path
        return root

    left_inds = np.where(all_dists[0] == largest_dist)[0]
    left_split = all_dists[left_inds][:, left_inds]
    left_labels, left_point_ids = labels[left_inds], point_ids[left_inds]
    root.set_left_tree(_make_tree(left_split, left_labels, left_point_ids, path=path + 'l'))
    if root.left_tree.label is not None:
        root.children += [root.left_tree]
    else:
        root.children += root.left_tree.children

    right_inds = np.where(all_dists[0] != largest_dist)[0]
    right_split = all_dists[right_inds][:, right_inds]
    right_labels, right_point_ids = labels[right_inds], point_ids[right_inds]
    root.set_right_tree(_make_tree(right_split, right_labels, right_point_ids, path=path + 'r'))
    if root.right_tree.label is not None:
        root.children += [root.right_tree]
    else:
        root.children += root.right_tree.children

    return root

def connect_clusters(left_embedding, right_embedding, dist, rotate):
    if rotate:
        if len(left_embedding) > 1:
            left_embedding = rotate_embedding(left_embedding)
        if len(right_embedding) > 1:
            right_embedding = rotate_embedding(right_embedding)

    left_embedding, right_embedding = _point_align_clusters(left_embedding, right_embedding, dist)

    # Now both the left and right embeddings have a single point at (0, 0)
    # So we separate them by the root distance in the left/right directions
    for point in right_embedding:
        point.coordinates[0] += dist

    embedding = left_embedding | right_embedding
    return embedding

def rotate_embedding(embedding):
    """ Rotate embedding so that its covariance matrix is axis-aligned """
    # Get centered covariance matrix
    points = np.array([p.coordinates for p in embedding]).T
    mean = np.mean(points, axis=1)
    cov = np.cov(points - np.expand_dims(mean, axis=1))

    # Find rotation matrix so that low-eigenvalue principal component is parallel to x-axis
    eig_vals, eig_vecs = np.linalg.eig(cov)
    min_var_direction = eig_vecs[np.argmin(eig_vals)]
    alpha = np.arctan(min_var_direction[1] / min_var_direction[0])
    c, s = np.cos(alpha), np.sin(alpha)
    rot_matrix = np.array([[c, -s], [s, c]]) # Rotates axis of lowest variance onto x-axis

    # Rotate the zero-mean embedding
    for point in embedding:
        point.coordinates = np.dot(rot_matrix, np.array(point.coordinates).T - mean)

    return embedding

def _point_align_clusters(left_embedding, right_embedding, dist):
    """
    Connect two embedding branches along a single point. Assume we are connecting along x.
    Set the left embedding's right-most point and the right embedding's left-most point to (0, 0).
    Lastly, move them apart by the necessary distance along x-axis.
    """
    # Set left embedding's right-most or top-most point to be at 0
    max_pos = np.array([-np.inf, 0])
    for point in left_embedding:
        if point.coordinates[0] > max_pos[0]:
            max_pos = np.array(point.coordinates)
    for point in left_embedding:
        point.coordinates -= max_pos

    # Set right embedding's left-most or bottom-most point to be at 0
    min_pos = np.array([np.inf, 0])
    for point in right_embedding:
        if point.coordinates[0] < min_pos[0]:
            min_pos = np.array(point.coordinates)
    for point in right_embedding:
        point.coordinates -= min_pos

    return left_embedding, right_embedding
def make_embedding(root, rotate=True):
    if root.label is not None:
        return set([Point([0, 0], root.label, root.point_id)])

    left_embedding = make_embedding(root.left_tree, rotate)
    right_embedding = make_embedding(root.right_tree, rotate)

    embedding = connect_clusters(
        left_embedding,
        right_embedding,
        root.dist,
        rotate=rotate
    )
    return embedding

def get_embedding_metadata(embedding):
    point_ids = [e.point_id for e in embedding]
    sort_inds = np.argsort(point_ids)
    points = np.array([e.coordinates for e in embedding])[sort_inds]
    if len(points.shape) == 1:
        points = np.stack([points, np.zeros_like(points)], axis=1)

    return points

def assert_correctness(dc_dists, embedding, n_neighbors=15, min_points=1):
    # Test that embedding preserves density-connectedness
    nn_dict = get_nearest_neighbors(embedding, n_neighbors=n_neighbors, min_points=1)
    np.testing.assert_allclose(dc_dists, nn_dict['_all_dists'])

def make_tree(points, labels, min_points=1, n_neighbors=15, make_image=True, point_ids=None):
    dc_dists = get_nearest_neighbors(\
        points,
        n_neighbors=n_neighbors,
        min_points=min_points
    )['_all_dists']
    np.set_printoptions(threshold=np.inf)

    if point_ids is None:
        point_ids = np.arange(int(dc_dists.shape[0]))

    root = _make_tree(dc_dists, labels, point_ids)
    if make_image:
        plot_tree(root, labels)

    return root, dc_dists

def plot_embedding(embed_points, embed_labels, titles):
    if len(embed_points.shape) == 1:
        embed_points = np.stack((embed_points, np.zeros_like(embed_points)), -1)
    if not isinstance(embed_labels, list):
        embed_labels = [embed_labels]
    if not isinstance(titles, list):
        titles = [titles]
    assert len(embed_labels) == len(titles)
    fig, axes = plt.subplots(1, len(embed_labels))
    fig.set_figwidth(3 * len(embed_labels))
    for i, labels in enumerate(embed_labels):
        axes[i].scatter(embed_points[:, 0], embed_points[:, 1], c=labels)
        axes[i].set_title(titles[i])
    plt.show()

def make_dc_embedding(root, dc_dists, min_points=1, n_neighbors=15, embed_dim=2):
    rotate = embed_dim > 1
    embedding = make_embedding(root, rotate=rotate)
    embed_points = get_embedding_metadata(embedding)

    assert_correctness(dc_dists, embed_points, n_neighbors=n_neighbors)
    return embed_points
