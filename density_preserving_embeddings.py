import numpy as np
from distance_metric import get_nearest_neighbors

class Tree:
    def __init__(self, dist):
        self.dist = dist
        self.left_tree = None
        self.right_tree = None
        self.label = None
        self.point_id = None

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
        total = 0
        if self.left_tree:
            total += len(self.left_tree)
        if self.right_tree:
            total += len(self.right_tree)
        return total + 1

class Point:
    def __init__(self, coordinates, label, point_id):
        self.coordinates = coordinates
        self.label = label
        self.point_id = point_id



def make_tree(all_dists, labels, point_ids):
    # FIXME -- the right way to do this is to build the tree up while we're connecting the components
    largest_dist = np.max(all_dists)
    root = Tree(largest_dist)
    if largest_dist == 0:
        root.label = labels[0]
        root.point_id = point_ids[0]
        return root

    left_inds = np.where(all_dists[0] == largest_dist)[0]
    left_split = all_dists[left_inds][:, left_inds]
    left_labels = labels[left_inds]
    left_point_ids = point_ids[left_inds]
    root.set_left_tree(make_tree(left_split, left_labels, left_point_ids))

    right_inds = np.where(all_dists[0] != largest_dist)[0]
    right_split = all_dists[right_inds][:, right_inds]
    right_labels = labels[right_inds]
    right_point_ids = point_ids[right_inds]
    root.set_right_tree(make_tree(right_split, right_labels, right_point_ids))

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
    max_pos = -np.inf
    slide = 0
    for point in left_embedding:
        if point.coordinates[0] > max_pos:
            max_pos = point.coordinates[0]
            slide = point.coordinates[1]
    for point in left_embedding:
        point.coordinates[0] -= max_pos
        point.coordinates[1] -= slide

    # Set right embedding's left-most or bottom-most point to be at 0
    min_pos = np.inf
    slide = 0
    for point in right_embedding:
        if point.coordinates[0] < min_pos:
            min_pos = point.coordinates[0]
            slide = point.coordinates[1]
    for point in right_embedding:
        point.coordinates[0] -= min_pos
        point.coordinates[1] -= slide

    return left_embedding, right_embedding

def make_embedding(root, rotate=True, depth=0):
    if root.label is not None:
        return set([Point([0, 0], root.label, root.point_id)])

    left_embedding = make_embedding(root.left_tree, rotate, depth=depth+1)
    right_embedding = make_embedding(root.right_tree, rotate, depth=depth+1)

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
    labels = np.array([e.label for e in embedding])[sort_inds]

    return points, labels

def assert_correctness(dc_dists, new_points):
    # Test that embedding preserves density-connectedness
    nn_dict = get_nearest_neighbors(points, n_neighbors=15)
    unique_orig = np.sort(dc_dists)
    unique_new = np.sort(nn_dict['_all_dists'])
    assert np.allclose(unique_orig, unique_new)

def make_dc_embedding(dc_dists, labels, embed_dim=2):
    if embed_dim == 1:
        rotate = False
    else:
        rotate = True

    num_points = int(dc_dists.shape[0])
    point_ids = np.arange(num_points)
    # FIXME -- do we need np.copy here?
    root = make_tree(np.copy(dc_dists), labels, point_ids)

    embedding = make_embedding(root, rotate=rotate)
    points, labels = get_embedding_metadata(embedding)

    return points, labels
