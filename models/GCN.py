import torch.nn as nn
from torch_geometric.nn import GCNConv


class GCNNetwork(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(GCNNetwork, self).__init__()

        # hyper-params
        num_layers = 2 if 'num_layers' not in kwargs else kwargs['num_layers']
        layer_norm = False if 'layer_norm' not in kwargs else kwargs['layer_norm']
        dropout = 0 if 'dropout' not in kwargs else kwargs['dropout']
        hidden_channels = 16 if 'hidden_channels' not in kwargs else kwargs['hidden_channels']
        activation = nn.ReLU if 'activation' not in kwargs else kwargs['activation']

        self.layers = nn.ModuleList()

        # input layer
        self.layers.append(GCNConv(in_channels, hidden_channels))
        if layer_norm:
            self.layers.append(nn.LayerNorm(hidden_channels))  # Because of multi-head, output size is num_heads * hidden_channels
        if dropout:
            self.layers.append(nn.Dropout(dropout))
        self.layers.append(activation())

        # hidden layers: GCNConv + LayerNorm + Dropout
        for _ in range(num_layers - 2):
            self.layers.append(GCNConv(hidden_channels, hidden_channels))
            if layer_norm:
                self.layers.append(nn.LayerNorm(hidden_channels))
            if dropout:
                self.layers.append(nn.Dropout(dropout))
            self.layers.append(activation())

        self.layers.append(GCNConv(hidden_channels, out_channels, heads=1))

    def forward(self, x, edge_index):
        for i in range(0, len(self.layers) - 1, 3):
            x = self.layers[i](x, edge_index)  # apply GCNConv
            x = self.layers[i + 1](x)  # layer norm
            x = self.layers[i + 2](x)  # dropout
            x = self.layers[i + 3](x) # activation

        # Final output layer (no activation or dropout)
        x = self.layers[-1](x, edge_index)
        return x
