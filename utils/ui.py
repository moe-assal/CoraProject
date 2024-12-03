import torch
from models.JK import JumpingKnowledge
from torch_geometric.data import Data
from flask import Flask, request, jsonify
import json


class API:
    def __init__(self, model_path, in_channels, out_channels, **model_kwargs):
        """
        Initialize the API with the pre-trained model.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = JumpingKnowledge(in_channels, out_channels, **model_kwargs).to(self.device)
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

        data.n2v = torch.zeros(data.x.size(0), 1).to(self.device)  # Placeholder for n2v, not unused
        data.batch_size = 1  # Assuming one sample at a time for prediction

        return data, new_node_id

    def predict(self, feature_vector, connections, cora_dataset):
        """
        Make predictions for a new node using the pre-trained model.
        """
        with torch.no_grad():
            data, node_id = self.preprocess(feature_vector, connections, cora_dataset)
            prediction = self.model.node_predict(data, node_id)
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
