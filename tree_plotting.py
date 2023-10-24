import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def find_node_positions(root, width=1, vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None):
    if pos is None:
        pos = [[xcenter, vert_loc]]
    else:
        pos.append([xcenter, vert_loc])
    if root.left_tree is not None and root.right_tree is not None:
        dx = width / 2
        left_x = xcenter - dx / 2
        right_x = left_x + dx
        pos = find_node_positions(
            root.left_tree,
            width=dx,
            vert_gap=vert_gap, 
            vert_loc=vert_loc-vert_gap,
            xcenter=left_x,
            pos=pos,
        )
        pos = find_node_positions(
            root.right_tree,
            width=dx,
            vert_gap=vert_gap, 
            vert_loc=vert_loc-vert_gap,
            xcenter=right_x,
            pos=pos,
        )

    return pos

def make_node_lists(root, point_labels, parent_count, dist_list, edge_list, color_list, alpha_list):
    count = parent_count
    dist_list.append(root.dist)
    if root.is_leaf():
        color_list.append(point_labels[root.point_id])
        alpha_list.append(1)
    else:
        color_list.append(-1)
        alpha_list.append(0.5)

    for tree in [root.left_tree, root.right_tree]:
        if tree is not None:
            edge_list.append((parent_count, count+1))
            count = make_node_lists(
                tree,
                point_labels,
                count+1,
                dist_list,
                edge_list,
                color_list,
                alpha_list
            )

    return count

def plot_tree(root, labels):
    edge_list = []
    dist_list = []
    color_list = []
    alpha_list = []

    make_node_lists(root, labels, 1, dist_list, edge_list, color_list, alpha_list)
    G = nx.Graph()
    G.add_edges_from(edge_list)
    pos_list = find_node_positions(root, 10)

    pos_dict = {}
    dist_dict = {}
    for i, node in enumerate(G.nodes):
        pos_dict[node] = pos_list[i]
        if dist_list[i] > 0:
            dist_dict[node] = '{:.1f}'.format(dist_list[i])
        
    nx.draw_networkx_nodes(G, pos=pos_dict, node_color=color_list, alpha=alpha_list)
    nx.draw_networkx_edges(G, pos=pos_dict)
    nx.draw_networkx_labels(G, pos=pos_dict, labels=dist_dict)
    plt.savefig("tree.png")
    plt.show()


def plot_embedding(embed_points, embed_labels, titles, centers):
    if len(embed_points.shape) == 1:
        embed_points = np.stack((embed_points, np.zeros_like(embed_points)), -1)
    if not isinstance(embed_labels, list):
        embed_labels = [embed_labels]
    if not isinstance(titles, list):
        titles = [titles]
    assert len(embed_labels) == len(titles)
    fig, axes = plt.subplots(1, len(embed_labels))
    fig.set_figwidth(4 * len(embed_labels))
    for i, labels in enumerate(embed_labels):
        axes[i].scatter(embed_points[:, 0], embed_points[:, 1], c=labels)
        axes[i].set_title(titles[i])
    plt.show()

