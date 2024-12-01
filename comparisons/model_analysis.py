from models.JK import JumpingKnowledge
from dataloader.loader import sample
from pipelines.trainer import GNNTrainer
import torch.nn as nn
from utils.utils import set_seed
from utils.loss_measures import CrossEntropyLoss
from utils.plotting import plot_metrics, plot_confusion_matrix

set_seed(47)

JK_config = {'in_channels': 1433, 'out_channels': 7, 'num_heads': 4, 'num_layers': 3, 'hidden_channels': 64,
          'dropout': 0.5, 'layer_norm': False, 'activation': nn.SELU, 'lr': 0.01, 'mlp_num_layers': 1}

batch_size = 16

sample_loaders = (sample(batch_size, mode="train"), sample(batch_size, mode="val"), sample(batch_size, mode="test"))

gnn = JumpingKnowledge(**JK_config)
trainer = GNNTrainer(gnn, *sample_loaders, CrossEntropyLoss(**JK_config), **JK_config)
data = trainer.train(num_epochs=100, track=True)
plot_metrics(data)
cm = trainer.compute_confusion_matrix(sample_loaders[-1])
plot_confusion_matrix(cm)
