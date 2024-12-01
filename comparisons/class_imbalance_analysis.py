from models.GAT import GATNetwork
from models.JK import JumpingKnowledge
from models.GCN import GCNNetwork
from dataloader.loader import sample, weighted_sampler
from pipelines.trainer import GNNTrainer
import torch.nn as nn
from utils.utils import set_seed
from utils.loss_measures import FocalLoss, CrossEntropyLoss


config_imbalance = {
    'weighted':{
        'GCN': {'in_channels': 1433, 'out_channels': 7, 'num_heads': 4, 'num_layers': 3, 'hidden_channels': 64,
                'dropout': 0.5, 'layer_norm': True, 'activation': nn.LeakyReLU, 'lr': 0.001, 'mlp_num_layers': 2},
        'GAT': {'in_channels': 1433, 'out_channels': 7, 'num_heads': 2, 'num_layers': 2, 'hidden_channels': 128,
                'dropout': 0.5, 'layer_norm': False, 'activation': nn.SELU, 'lr': 0.001, 'mlp_num_layers': 2},
        'JK': {'in_channels': 1433, 'out_channels': 7, 'num_heads': 2, 'num_layers': 2, 'hidden_channels': 64,
               'dropout': 0.5, 'layer_norm': True, 'activation': nn.LeakyReLU, 'lr': 0.001, 'mlp_num_layers': 1}},
    'focal': {
        'GCN': {'in_channels': 1433, 'out_channels': 7, 'num_heads': 1, 'num_layers': 2, 'hidden_channels': 16,
                'dropout': 0.0, 'layer_norm': False, 'activation': nn.ReLU, 'lr': 0.001, 'mlp_num_layers': 1},
        'GAT': {'in_channels': 1433, 'out_channels': 7, 'num_heads': 1, 'num_layers': 5, 'hidden_channels': 32,
                'dropout': 0.2, 'layer_norm': True, 'activation': nn.SELU, 'lr': 0.01, 'mlp_num_layers': 3},
        'JK': {'in_channels': 1433, 'out_channels': 7, 'num_heads': 1, 'num_layers': 4, 'hidden_channels': 64,
               'dropout': 0.2, 'layer_norm': False, 'activation': nn.LeakyReLU, 'lr': 0.01, 'mlp_num_layers': 1}},
    'base': {
        'GCN': {'in_channels': 1433, 'out_channels': 7, 'num_heads': 2, 'num_layers': 2, 'hidden_channels': 128,
                'dropout': 0.5, 'layer_norm': False, 'activation': nn.SELU, 'lr': 0.001, 'mlp_num_layers': 1},
        'GAT': {'in_channels': 1433, 'out_channels': 7, 'num_heads': 2, 'num_layers': 4, 'hidden_channels': 128,
                'dropout': 0.2, 'layer_norm': True, 'activation': nn.SELU, 'lr': 0.001, 'mlp_num_layers': 1},
        'JK': {'in_channels': 1433, 'out_channels': 7, 'num_heads': 4, 'num_layers': 3, 'hidden_channels': 64,
               'dropout': 0.5, 'layer_norm': False, 'activation': nn.ReLU, 'lr': 0.001, 'mlp_num_layers': 1}
    }
}


name_to_gnn = {
    'JK': JumpingKnowledge,
    'GCN': GCNNetwork,
    'GAT': GATNetwork
}

# use a seed for reproducibility
set_seed(42)

batch_size = 16


weighted_loaders = lambda: (weighted_sampler(batch_size, mode="train"),
                            weighted_sampler(batch_size, mode="val"),
                            weighted_sampler(batch_size, mode="test"))

sample_loaders = lambda: (sample(batch_size, mode="train"),
                          sample(batch_size, mode="val"),
                          sample(batch_size, mode="test"))


results = dict()
for config in config_imbalance.keys():
    results[config] = dict()
    if config == "weighted":
        loader = weighted_loaders()
    else:
        loader = sample_loaders()

    if config == "focal":
        loss_function = FocalLoss
    else:
        loss_function = CrossEntropyLoss

    for gnn_name in config_imbalance[config].keys():
        gnn_config = config_imbalance[config][gnn_name]
        gnn = name_to_gnn[gnn_name](**gnn_config)
        trainer = GNNTrainer(gnn, *loader, loss_function(**gnn_config), **gnn_config)
        trainer.train(num_epochs=200)
        results[config][gnn_name] = trainer.compute_f1_score(trainer.test_loader).item()

print(results)