import torch
from torch_geometric.utils import dropout_adj

# Define the edge dropout function
def apply_edge_dropout(data, p=0.2):
    edge_index, edge_attr = dropout_adj(
        data.edge_index,
        edge_attr=None,  # If there are edge features, include them here
        p=p,
        force_undirected=True  # To ensure symmetry for undirected graphs
    )
    data.edge_index = edge_index
    return data


def apply_node_dropout(data, p):
    num_nodes = data.num_nodes
    mask = torch.rand(num_nodes) > p  # Keep nodes with probability 1-p
    node_mask = mask.nonzero(as_tuple=True)[0]

    # Adjust edge_index to keep only edges with both nodes in the mask
    edge_mask = mask[data.edge_index[0]] & mask[data.edge_index[1]]
    data.edge_index = data.edge_index[:, edge_mask]

    # Update node features if present
    data.x = data.x[node_mask] if data.x is not None else None
    data.node_mask = node_mask  # Save the mask for later usage
    return data
