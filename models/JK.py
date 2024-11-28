import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, MLP


class JumpingKnowledge(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(JumpingKnowledge, self).__init__()

        # Hyperparameters
        num_layers = kwargs.get('num_layers', 2)
        layer_norm = kwargs.get('layer_norm', False)
        dropout = kwargs.get('dropout', 0.0)
        hidden_channels = kwargs.get('hidden_channels', 16)
        activation = kwargs.get('activation', nn.ReLU)
        mlp_layers = kwargs.get('mlp_num_layers', 2)

        assert num_layers >= 2, "Number of layers must be at least 2."

        # Initialize GNN layers
        self.gnn_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList() if layer_norm else None
        self.dropout_layers = nn.ModuleList() if dropout else None
        self.act = activation()

        # Input layer
        self.gnn_layers.append(GCNConv(in_channels, hidden_channels))
        if layer_norm:
            self.norm_layers.append(nn.LayerNorm(hidden_channels))
        if dropout:
            self.dropout_layers.append(nn.Dropout(dropout))

        # Hidden layers: GCNConv + LayerNorm + Dropout
        for _ in range(num_layers - 1):
            self.gnn_layers.append(GCNConv(hidden_channels, hidden_channels))
            if layer_norm:
                self.norm_layers.append(nn.LayerNorm(hidden_channels))
            if dropout:
                self.dropout_layers.append(nn.Dropout(dropout))

        # Projection MLP for combined outputs
        self.mlp = MLP(
            in_channels=hidden_channels * num_layers,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=mlp_layers,
            dropout=dropout,
            act=self.act
        )

    def forward(self, x, edge_index):
        # Collect outputs from all GCN layers for jumping knowledge
        pre_mlp_x = [x]
        for i in range(len(self.gnn_layers)):
            x = self.gnn_layers[i](x, edge_index)
            if self.norm_layers:
                x = self.norm_layers[i](x)
            if self.dropout_layers:
                x = self.dropout_layers[i](x)
            x = self.act(x)
            pre_mlp_x.append(x)

        # Concatenate layer outputs and pass through MLP
        x_concat = torch.cat(pre_mlp_x, dim=1)
        x_out = self.mlp(x_concat)

        return torch.softmax(x_out, dim=1)  # Output probabilities
