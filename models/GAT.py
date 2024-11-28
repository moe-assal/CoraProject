import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, MLP


class GATNetwork(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(GATNetwork, self).__init__()

        # hyper-params
        num_layers = kwargs.get('num_layers', 2)
        num_heads = kwargs.get('num_heads', 4)
        layer_norm = kwargs.get('layer_norm', False)
        dropout = kwargs.get('dropout', 0)
        hidden_channels = kwargs.get('hidden_channels', 16)
        activation = kwargs.get('activation', nn.ReLU)
        mlp_layers = kwargs.get('mlp_num_layers', 2)
        self.with_n2v = kwargs.get('with_n2v', False)

        assert num_layers >= 2, "Number of layers must be at least 2."

        self.mlp = MLP(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=mlp_layers,
            dropout=dropout,
            act=activation()
        )

        self.gnn_layers = nn.ModuleList()
        self.act = activation()

        self.norm_layers = nn.ModuleList() if layer_norm else None
        self.dropout_layers = nn.ModuleList() if dropout else None

        # input layer
        self.gnn_layers.append(GATConv(in_channels, hidden_channels, heads=num_heads))
        if layer_norm:
            self.norm_layers.append(nn.LayerNorm(hidden_channels * num_heads))  # Because of multi-head, output size is num_heads * hidden_channels
        if dropout:
            self.dropout_layers.append(nn.Dropout(dropout))

        # hidden layers: GATConv + LayerNorm + Dropout
        for _ in range(num_layers - 2):
            self.gnn_layers.append(GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads))
            if layer_norm:
                self.norm_layers.append(nn.LayerNorm(hidden_channels * num_heads))
            if dropout:
                self.dropout_layers.append(nn.Dropout(dropout))

        # final layer: without multiple heads
        self.gnn_layers.append(GATConv(hidden_channels * num_heads, hidden_channels, heads=1))

    def forward(self, batch):
        x, edge_index, n2v, batch_size = batch.x, batch.edge_index, batch.n2v, batch.batch_size
        if self.with_n2v:
            x = torch.cat([x, n2v], dim=1)

        for i in range(0, len(self.gnn_layers) - 1):
            x = self.gnn_layers[i](x, edge_index)
            if self.norm_layers:
                x = self.norm_layers[i](x)
            if self.dropout_layers:
                x = self.dropout_layers[i](x)
            x = self.act(x)

        # Final output layer (no activation or dropout)
        x = self.layers[-1](x, edge_index)
        x = self.mlp(x)
        x = x[:batch_size]
        return torch.softmax(x, dim=1)
