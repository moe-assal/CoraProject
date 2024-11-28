from dataloader.loader import sample
from models.GCN import GCNNetwork
from models.GAT import GATNetwork
from models.JK import JumpingKnowledge
from utils.utils import set_seed
from pipelines.search import SequentialSearch
from pipelines.trainer import GNNTrainer
from utils.loss_measures import CrossEntropyLoss
import torch.nn as nn
from utils.graph_dropout import apply_edge_dropout, apply_node_dropout
from torch_geometric.transforms import Compose
from typing import List

# use a seed for reproducibility
set_seed(42)

batch_size = 16
search_depth = 70


def get_node_augmented_loaders():
    node_dropout = lambda x: apply_node_dropout(x, 0.1)
    train_loader = sample(batch_size, mode="train", transform=node_dropout)
    val_loader = sample(batch_size, mode="val", transform=node_dropout)
    test_loader = sample(batch_size, mode="test", transform=node_dropout)
    return train_loader, val_loader, test_loader


def get_edge_augmented_loaders():
    edge_dropout = lambda x: apply_edge_dropout(x, 0.2)
    train_loader = sample(batch_size, mode="train", transform=edge_dropout)
    val_loader = sample(batch_size, mode="val", transform=edge_dropout)
    test_loader = sample(batch_size, mode="test", transform=edge_dropout)
    return train_loader, val_loader, test_loader

def get_node_and_edge_augmented_loaders():
    node_dropout = lambda x: apply_node_dropout(x, 0.1)
    edge_dropout = lambda x: apply_edge_dropout(x, 0.2)
    transform = Compose([node_dropout, edge_dropout])
    train_loader = sample(batch_size, mode="train", transform=transform)
    val_loader = sample(batch_size, mode="val", transform=transform)
    test_loader = sample(batch_size, mode="test", transform=transform)
    return train_loader, val_loader, test_loader


best_params = dict()

param_options = {
    "in_channels": 1433,
    "out_channels": 7,
    "num_heads": [1, 2, 4],
    "num_layers": [2, 3, 4, 5],
    "hidden_channels": [16, 32, 64, 128],
    "dropout": [0.0, 0.2, 0.5],
    "layer_norm": [False, True],
    "activation": [nn.ReLU, nn.LeakyReLU, nn.SELU, nn.Sigmoid],
    "lr": [0.01, 0.001],
    "mlp_num_layers": [1, 2, 3]
}


for loader, loader_name in zip([get_edge_augmented_loaders, get_node_augmented_loaders, get_node_and_edge_augmented_loaders], ['edge', 'node', 'both']):
    best_params[loader_name] = dict()
    for gnn, gnn_name in zip([GATNetwork, GCNNetwork, JumpingKnowledge], ['GAT', 'GCN', 'JK']):
        # initialize starting point
        params = {key: (values[0] if isinstance(values, List) else values) for key, values in param_options.items()}
        for i in range(search_depth):
            # re-initialize to get new augmented graph
            search = SequentialSearch(
                model_class=gnn,
                trainer_class=GNNTrainer,
                loss_class=CrossEntropyLoss,
                loaders=loader(),
                param_options=param_options,
                start_param=params
            )

            # Perform parameter optimization
            best_params[loader_name][gnn_name] = search.run_search(num_times=1)


print(best_params)
