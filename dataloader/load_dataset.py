import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data

# Paths to the files
content_path = 'cora.content'  # Replace with your actual path
cites_path = 'cora.cites'      # Replace with your actual path

# Read the content file
content = pd.read_csv(content_path, sep='\t', header=None)
# Extract features, labels, and nodes
features = content.iloc[:, 1:-1].values  # Feature vectors
labels = pd.factorize(content.iloc[:, -1])[0]  # Encode labels as integers
nodes = content.iloc[:, 0].values  # Node IDs

# Read the cites file
cites = pd.read_csv(cites_path, sep='\t', header=None, names=['target', 'source'])
# Convert to edge list (PyG expects edges as source -> target)
edges = cites[['source', 'target']].values

# Map node IDs to indices (to ensure edges match node indices)
node_index = {node_id: idx for idx, node_id in enumerate(nodes)}
edges = np.array([[node_index[src], node_index[dst]] for src, dst in edges if src in node_index and dst in node_index])

# Create the PyG Data object
data = Data(
    x=torch.tensor(features, dtype=torch.float),
    edge_index=torch.tensor(edges.T, dtype=torch.long),
    y=torch.tensor(labels, dtype=torch.long)
)

# Save the processed PyG dataset
output_path = 'processed_cora_pyg.pt'
torch.save(data, output_path)
print(f"Dataset saved to {output_path}")

# For inspection:
print(f"Feature Matrix Shape: {data.x.shape}")
print(f"Edge Index Shape: {data.edge_index.shape}")
print(f"Label Vector Shape: {data.y.shape}")
