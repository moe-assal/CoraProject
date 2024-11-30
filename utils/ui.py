import torch
from models.GAT import GATNetwork
from torch_geometric.data import Data
from flask import Flask, request, jsonify
import json


class GATModelAPI:
    def __init__(self, model_path, in_channels, out_channels, **model_kwargs):
        """
        Initialize the API with the pre-trained model.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GATNetwork(in_channels, out_channels, **model_kwargs).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def preprocess(self, feature_vector, connections, cora_dataset):
        """
        Preprocess the input data to create a PyTorch Geometric Data object.
        """
        # Convert connections to edge_index
        num_nodes = cora_dataset.x.size(0) + 1
        new_node_id = num_nodes - 1  # New node id
        edge_index = torch.cat([cora_dataset.edge_index,
                                torch.tensor([[new_node_id, conn] for conn in connections] +
                                             [[conn, new_node_id] for conn in connections], dtype=torch.long).T], dim=1)

        # Add the new node's features to the existing feature matrix
        x = torch.cat([cora_dataset.x, torch.tensor([feature_vector], dtype=torch.float)], dim=0)

        # Prepare the data batch
        data = Data(x=x, edge_index=edge_index).to(self.device)

        # TODO: if n2v is used, do an average location
        # data.n2v = torch.zeros(data.x.size(0), 1).to(self.device)  # Placeholder for n2v, if unused
        data.batch_size = 1  # Assuming one sample at a time for prediction

        return data

    def predict(self, feature_vector, connections, cora_dataset):
        """
        Make predictions for a new node using the pre-trained model.
        """
        with torch.no_grad():
            data = self.preprocess(feature_vector, connections, cora_dataset)
            prediction = self.model(data)
            return prediction.cpu().numpy().tolist()

    def create_app(self, cora_dataset):
        """
        Create a Flask application.
        """
        app = Flask(__name__)

        @app.route("/predict", methods=["POST"])
        def predict():
            """
            Handle prediction requests.
            """
            data = request.json
            feature_vector = data.get("feature_vector")
            connections = data.get("connections")

            if feature_vector is None or connections is None:
                return jsonify({"error": "Invalid input, must contain 'feature_vector' and 'connections'."}), 400

            try:
                predictions = self.predict(feature_vector, connections, cora_dataset)
                return jsonify({"predictions": predictions})
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        return app

"""
if __name__ == "__main__":
    # Load the Cora dataset (assume it has already been preprocessed)
    from torch_geometric.datasets import Planetoid
    dataset = Planetoid(root="/tmp/Cora", name="Cora")
    cora_data = dataset[0]

    # Parameters of the model (adjust based on your setup)
    in_channels = dataset.num_node_features
    out_channels = dataset.num_classes
    model_kwargs = {
        "num_layers": 2,
        "num_heads": 4,
        "layer_norm": True,
        "dropout": 0.2,
        "hidden_channels": 16,
        "activation": torch.nn.ReLU,
        "mlp_num_layers": 2,
        "with_n2v": False,
    }

    # Create API instance
    model_path = "best_model.pth"
    api = GATModelAPI(model_path, in_channels, out_channels, **model_kwargs)

    # Run Flask app
    app = api.create_app(cora_data)
    app.run(host="0.0.0.0", port=5000)
"""