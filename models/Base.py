import torch
import torch.nn as nn
from torch_geometric.nn import MLP


class GNNClassifier(nn.Module):
    def __init__(self, gnn_model, input_dim, output_dim, **kwargs):
        super(GNNClassifier, self).__init__()

        dropout = 0 if 'dropout' not in kwargs else kwargs['dropout']
        mlp_hidden_dim = input_dim if 'mlp_hidden_dim' not in kwargs else kwargs['mlp_hidden_dim']
        num_layers = 2 if 'num_layers' not in kwargs else kwargs['num_layers']
        activation = nn.ReLU() if 'act' not in kwargs else kwargs['act']

        # GNN Model
        self.gnn_model = gnn_model

        # MLP for classification
        self.mlp = MLP(
            in_channels=input_dim,
            hidden_channels=mlp_hidden_dim,
            out_channels=output_dim,
            num_layers=num_layers,
            dropout=dropout,
            act=activation
        )


    def forward(self, x, edge_index):
        gnn_out = self.gnn_model(x, edge_index)
        logits = self.mlp(gnn_out)  # [num_nodes, output_dim]
        return torch.softmax(logits, dim=1)  # Softmax distribution over classes
