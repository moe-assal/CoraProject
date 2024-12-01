import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import NeighborLoader
from torch.utils.data import WeightedRandomSampler

def weighted_sampler(batch_size, mode, pre_transform=None, transform=None):
    dataset = Planetoid(root="../data/", name="Cora", pre_transform=pre_transform, transform=transform)
    data = dataset[0]

    # class weights
    node_labels = data.y
    class_counts = torch.bincount(node_labels)
    class_weights = 1.0 / class_counts.float()  # Inverse frequency for each class
    node_weights = class_weights[node_labels]

    # Step 3: Create a WeightedRandomSampler
    if mode == "train":
        mask = data.train_mask.nonzero(as_tuple=True)[0]
    elif mode == "test":
        mask = data.test_mask.nonzero(as_tuple=True)[0]
    elif mode == "val":
        mask = data.val_mask.nonzero(as_tuple=True)[0]
    else:
        raise ValueError("mask value invalid")

    weights = node_weights[mask]
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(mask), replacement=True)

    train_loader = NeighborLoader(
        data,
        num_neighbors=[20, 20],  # Sampling neighbors for subgraphs
        batch_size=batch_size,
        input_nodes=mask,
        sampler=sampler
    )

    return train_loader



def sample(batch_size, mode, pre_transform=None, transform=None):
    dataset = Planetoid(root="../data", name="Cora", pre_transform=pre_transform, transform=transform)
    data = dataset[0]

    if mode == "train":
        mask = data.train_mask.nonzero(as_tuple=True)[0]
    elif mode == "test":
        mask = data.test_mask.nonzero(as_tuple=True)[0]
    elif mode == "val":
        mask = data.val_mask.nonzero(as_tuple=True)[0]
    else:
        raise ValueError("mask value invalid")

    loader =  NeighborLoader(
        data,
        num_neighbors=[20, 20],  # Number of neighbors sampled at each layer
        batch_size=batch_size,  # Number of target nodes in each batch
        input_nodes= mask
    )

    return loader
