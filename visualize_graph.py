# visualize_graph.py

import torch
import os.path as osp
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, to_networkx

from dataset import get_train_data

def visualize_neighborhood(data, node_idx, hops=2):
    """
    Visualizes the k-hop neighborhood of a given node.
    """
    # --- CHANGE: Added num_nodes to the function call ---
    # This tells the function the true size of the graph.
    subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
        node_idx=node_idx,
        num_hops=hops,
        edge_index=data.edge_index,
        relabel_nodes=True,
        num_nodes=data.num_nodes # This line fixes the error
    )

    sub_labels = data.y[subset]
    center_node_idx = 0

    subgraph_data = Data(edge_index=sub_edge_index, num_nodes=subset.size(0))
    subgraph_nx = to_networkx(subgraph_data)

    node_colors = []
    for i, label in enumerate(sub_labels):
        if i == center_node_idx:
            node_colors.append('#9400D3') # Purple for the center node
        elif label == 0:
            node_colors.append('dodgerblue') # Blue for human
        else:
            node_colors.append('red') # Red for bot

    plt.figure(figsize=(10, 10))
    nx.draw(
        subgraph_nx,
        with_labels=False,
        node_color=node_colors,
        node_size=50,
        width=0.5
    )
    center_node_label = 'Bot' if data.y[node_idx] == 1 else 'Human'
    plt.title(f'{hops}-Hop Neighborhood of a {center_node_label} (Node {node_idx})', fontsize=16)

def main():
    print("✅ Loading data...")
    data = get_train_data('Twibot-22')

    bot_indices = (data.y == 1).nonzero(as_tuple=True)[0]
    human_indices = (data.y == 0).nonzero(as_tuple=True)[0]

    random_bot_idx = bot_indices[torch.randint(len(bot_indices), (1,))].item()
    random_human_idx = human_indices[torch.randint(len(human_indices), (1,))].item()

    print(f"Visualizing neighborhood for Bot Node {random_bot_idx}...")
    visualize_neighborhood(data, random_bot_idx)
    plt.savefig('bot_neighborhood.png')
    print("✅ Saved bot_neighborhood.png")
    
    print(f"Visualizing neighborhood for Human Node {random_human_idx}...")
    visualize_neighborhood(data, random_human_idx)
    plt.savefig('human_neighborhood.png')
    print("✅ Saved human_neighborhood.png")

if __name__ == '__main__':
    main()