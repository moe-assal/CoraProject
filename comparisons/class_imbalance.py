from dataloader.loader import weighted_sampler, sample
from models.GCN import GCNNetwork
from models.GAT import GATNetwork
from models.JK import JumpingKnowledge
from utils.utils import set_seed
from pipelines.search import SequentialSearch
from pipelines.trainer import GNNTrainer
from utils.loss_measures import CrossEntropyLoss, FocalLoss
import torch.nn as nn

best_params = dict()
search_depth = 4

#### Start with weighted sampling
best_params['weighted'] = dict()

# use a seed for reproducibility
set_seed(42)

batch_size = 16

train_loader = weighted_sampler(batch_size, mode="train")
val_loader = weighted_sampler(batch_size, mode="val")
test_loader = weighted_sampler(batch_size, mode="test")
weighted_loaders = (train_loader, val_loader, test_loader)

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

### GCN optim
search = SequentialSearch(
    model_class=GCNNetwork,
    trainer_class=GNNTrainer,
    loss_class=CrossEntropyLoss,
    loaders=weighted_loaders,
    param_options=param_options
)

# Perform parameter optimization
best_params['weighted']['GCN'] = search.run_search(num_times=search_depth)

### GAT Optim
search = SequentialSearch(
    model_class=GATNetwork,
    trainer_class=GNNTrainer,
    loss_class=CrossEntropyLoss,
    loaders=weighted_loaders,
    param_options=param_options
)

# Perform parameter optimization
best_params['weighted']['GAT'] = search.run_search(num_times=search_depth)

### JK Optim
search = SequentialSearch(
    model_class=JumpingKnowledge,
    trainer_class=GNNTrainer,
    loss_class=CrossEntropyLoss,
    loaders=weighted_loaders,
    param_options=param_options
)

# Perform parameter optimization
best_params['weighted']['JK'] = search.run_search(num_times=search_depth)

#### Focal Loss
train_loader = sample(batch_size, mode="train")
val_loader = sample(batch_size, mode="val")
test_loader = sample(batch_size, mode="test")
loaders = (train_loader, val_loader, test_loader)

### GCN optim
search = SequentialSearch(
    model_class=GCNNetwork,
    trainer_class=GNNTrainer,
    loss_class=FocalLoss,
    loaders=loaders,
    param_options=param_options
)

# Perform parameter optimization
best_params['focal']['GCN'] = search.run_search(num_times=search_depth)

### GAT Optim
search = SequentialSearch(
    model_class=GATNetwork,
    trainer_class=GNNTrainer,
    loss_class=FocalLoss,
    loaders=loaders,
    param_options=param_options
)

# Perform parameter optimization
best_params['focal']['GAT'] = search.run_search(num_times=search_depth)

### JK Optim
search = SequentialSearch(
    model_class=JumpingKnowledge,
    trainer_class=GNNTrainer,
    loss_class=FocalLoss,
    loaders=loaders,
    param_options=param_options
)

# Perform parameter optimization
best_params['focal']['JK'] = search.run_search(num_times=search_depth)


#### Base
### GCN optim
search = SequentialSearch(
    model_class=GCNNetwork,
    trainer_class=GNNTrainer,
    loss_class=CrossEntropyLoss,
    loaders=loaders,
    param_options=param_options
)

# Perform parameter optimization
best_params['base']['GCN'] = search.run_search(num_times=search_depth)

### GAT Optim
search = SequentialSearch(
    model_class=GATNetwork,
    trainer_class=GNNTrainer,
    loss_class=CrossEntropyLoss,
    loaders=loaders,
    param_options=param_options
)

# Perform parameter optimization
best_params['base']['GAT'] = search.run_search(num_times=search_depth)

### JK Optim
search = SequentialSearch(
    model_class=JumpingKnowledge,
    trainer_class=GNNTrainer,
    loss_class=CrossEntropyLoss,
    loaders=loaders,
    param_options=param_options
)

# Perform parameter optimization
best_params['base']['JK'] = search.run_search(num_times=search_depth)

print(best_params)
