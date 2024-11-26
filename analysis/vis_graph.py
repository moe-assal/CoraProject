import matplotlib.pyplot as plt
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx
import networkx as nx

# Load the Cora dataset
dataset = Planetoid(root='../data', name='Cora')
data = dataset[0]

# Convert the PyG graph to a NetworkX graph
G = to_networkx(data, to_undirected=True)

# Extract the largest connected component
largest_cc = max(nx.connected_components(G), key=len)
G_largest = G.subgraph(largest_cc)

# Get node colors based on labels, filtering for the largest connected component
node_colors = [data.y[node].item() for node in G_largest.nodes]

# Generate a spring layout for the graph
pos = nx.spring_layout(G_largest, seed=42)

# Plot the graph
plt.figure(figsize=(10, 10))
nx.draw_networkx_nodes(
    G_largest, pos, node_size=50, cmap=plt.cm.tab10,
    node_color=node_colors, alpha=0.8
)
nx.draw_networkx_edges(G_largest, pos, width=0.5, alpha=0.5)
plt.title('Cora Graph')
plt.axis('off')
plt.show()
