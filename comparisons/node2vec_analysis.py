from dataloader.loader import sample
from models.GCN import GCNNetwork
from models.GAT import GATNetwork
from models.JK import JumpingKnowledge
from utils.utils import set_seed
from pipelines.trainer import GNNTrainer
from utils.loss_measures import CrossEntropyLoss
import torch.nn as nn

# use a seed for reproducibility
set_seed(42)

batch_size = 16

train_loader = sample(batch_size, mode="train")
val_loader = sample(batch_size, mode="val")
test_loader = sample(batch_size, mode="test")
loaders = (train_loader, val_loader, test_loader)

config_n2v = {
    'base': {
        'JK': {'in_channels': 1433, 'out_channels': 7, 'num_heads': 4, 'num_layers': 3, 'hidden_channels': 64,
               'dropout': 0.5, 'layer_norm': False, 'activation': nn.SELU, 'lr': 0.01, 'mlp_num_layers': 1},
        'GAT': {'in_channels': 1433, 'out_channels': 7, 'num_heads': 4, 'num_layers': 2, 'hidden_channels': 16,
                'dropout': 0.5, 'layer_norm': False, 'activation': nn.LeakyReLU, 'lr': 0.001, 'mlp_num_layers': 1},
        'GCN': {'in_channels': 1433, 'out_channels': 7, 'num_heads': 2, 'num_layers': 3, 'hidden_channels': 64,
                'dropout': 0.5, 'layer_norm': True, 'activation': nn.ReLU, 'lr': 0.001, 'mlp_num_layers': 1}},
    'base+n2v': {
        'JK': {'with_n2v': True, 'in_channels': 1561, 'out_channels': 7, 'num_heads': 1, 'num_layers': 2,
               'hidden_channels': 32, 'dropout': 0.2, 'layer_norm': False, 'activation': nn.ReLU, 'lr': 0.001, 'mlp_num_layers': 1},
        'GAT': {'with_n2v': True, 'in_channels': 1561, 'out_channels': 7, 'num_heads': 2, 'num_layers': 2, 'hidden_channels': 16,
                'dropout': 0.5, 'layer_norm': True, 'activation': nn.SELU, 'lr': 0.01, 'mlp_num_layers': 1},
        'GCN': {'with_n2v': True, 'in_channels': 1561, 'out_channels': 7, 'num_heads': 1, 'num_layers': 2,
                'hidden_channels': 32, 'dropout': 0.2, 'layer_norm': True, 'activation': nn.LeakyReLU, 'lr': 0.01, 'mlp_num_layers': 1}
    }
}


name_to_gnn = {
    'JK': JumpingKnowledge,
    'GCN': GCNNetwork,
    'GAT': GATNetwork
}

results = dict()
for config_name in config_n2v.keys():
    results[config_name] = dict()
    for gnn_name in config_n2v[config_name].keys():
        gnn_config = config_n2v[config_name][gnn_name]
        gnn = name_to_gnn[gnn_name](**gnn_config)
        trainer = GNNTrainer(gnn, *loaders, CrossEntropyLoss(**gnn_config), **gnn_config)
        trainer.train(num_epochs=200)
        results[config_name][gnn_name] = trainer.compute_f1_score(trainer.test_loader).item()

print(results)