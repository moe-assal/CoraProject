from dataloader.loader import sample
from models.GCN import GCNNetwork
from models.GAT import GATNetwork
from models.JK import JumpingKnowledge
from utils.utils import set_seed
from pipelines.search import SequentialSearch
from pipelines.trainer import GNNTrainer
from utils.loss_measures import CrossEntropyLoss
import torch.nn as nn

# use a seed for reproducibility
set_seed(42)

batch_size = 16
search_depth = 70


train_loader = sample(batch_size, mode="train")
val_loader = sample(batch_size, mode="val")
test_loader = sample(batch_size, mode="test")
loaders = (train_loader, val_loader, test_loader)

best_params = dict()

param_options_without_n2v = {
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

param_options_with_n2v = {
    "with_n2v": True,
    "in_channels": 1433 + 128,
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


for param, param_option_name in zip([param_options_without_n2v, param_options_with_n2v], ['base+n2v', 'base']):
    best_params[param_option_name] = dict()
    for gnn, gnn_name in zip([GATNetwork, GCNNetwork, JumpingKnowledge], ['GAT', 'GCN', 'JK']):
        search = SequentialSearch(
            model_class=gnn,
            trainer_class=GNNTrainer,
            loss_class=CrossEntropyLoss,
            loaders=loaders,
            param_options=param
        )

        # Perform parameter optimization
        best_params[param_option_name][gnn_name] = search.run_search(num_times=1)


print(best_params)
