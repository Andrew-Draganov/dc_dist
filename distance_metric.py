import numpy as np
from tqdm.auto import tqdm

class Component:
    def __init__(self, nodes, comp_id):
        self.nodes = set(nodes)
        self.comp_id = comp_id

def merge_components(c_i, c_j):
    merged_list = c_i.nodes.union(c_j.nodes)
    return Component(merged_list, c_i.comp_id)

def get_node_memberships(component_dict, num_points):
    node_membership = np.zeros([num_points])
    seen_components = []
    for i, component in component_dict.items():
        if component.comp_id not in seen_components:
            seen_components += [component.comp_id]
            for node in component.nodes:
                node_membership[node] = i

    print('num unique components:', len(seen_components))
    return node_membership


def distance_metric(points):
    """
    We define the distance from x_i to x_j as min(max(P(x_i, x_j))), where 
        - P(x_i, x_j) is any path from x_i to x_j
        - max(P(x_i, x_j)) is the largest edge weight in the path
        - min(max(P(x_i, x_j))) is the smallest largest edge weight
    """
    num_points = int(points.shape[0])
    density_connections = np.zeros([num_points, num_points])
    D = np.zeros([num_points, num_points])

    for i in range(num_points):
        x = points[i]
        for j in range(i+1, num_points):
            y = points[j]
            dist = np.sqrt(np.sum(np.square(x - y)))
            D[i, j] = dist
            D[j, i] = dist

    flat_D = np.reshape(D, [num_points * num_points])
    argsort_inds = np.argsort(flat_D)

    num_added = 0
    component_dict = {i: Component([i], i) for i in range(num_points)}
    max_comp_size = 1
    for index in tqdm(argsort_inds):
        i = int(index / num_points)
        j = index % num_points
        if component_dict[i].comp_id != component_dict[j].comp_id:
            epsilon = D[i, j]
            for node_i in component_dict[i].nodes:
                for node_j in component_dict[j].nodes:
                    density_connections[node_i, node_j] = epsilon
                    density_connections[node_j, node_i] = epsilon
            merged_component = merge_components(component_dict[i], component_dict[j])
            for node in merged_component.nodes:
                component_dict[node] = merged_component
            size_of_component = len(component_dict[i].nodes)
            if size_of_component > max_comp_size:
                max_comp_size = size_of_component
        if max_comp_size == num_points:
            break

    return density_connections


