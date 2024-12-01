import torch.nn as nn
from dataloader.preprocess import node2vec_transform
from utils.ui import API


if __name__ == "__main__":
    # Load the Cora dataset (assume it has already been preprocessed)
    from torch_geometric.datasets import Planetoid
    dataset = Planetoid(root="./data", name="Cora", pre_transform=node2vec_transform)
    cora_data = dataset[0]

    # Parameters of the model (adjust based on your setup)
    in_channels = dataset.num_node_features
    out_channels = dataset.num_classes
    JK_config = {'in_channels': 1433, 'out_channels': 7, 'num_heads': 4, 'num_layers': 3, 'hidden_channels': 64,
                 'dropout': 0.5, 'layer_norm': False, 'activation': nn.SELU, 'lr': 0.01, 'mlp_num_layers': 1}

    # Create API instance
    model_path = "./best_model.pth"
    api = API(model_path, **JK_config)

    # Run Flask app
    app = api.create_app(cora_data)
    app.run(host="0.0.0.0", port=5000)
