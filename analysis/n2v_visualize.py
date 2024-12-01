import torch
from sklearn.manifold import TSNE
from torch_geometric.datasets import Planetoid
import matplotlib.pyplot as plt
import numpy as np

def plot_node2vec_tsne(data, class_labels, figsize=(10, 8), perplexity=30, learning_rate=200, n_iter=1000):
    """
    Plot the Node2Vec embeddings using t-SNE.

    Args:
        data (torch_geometric.data.Data): Cora dataset with pre-computed Node2Vec embeddings in `data.n2v`.
        class_labels (list): List of class labels corresponding to unique classes in `data.y`.
        figsize (tuple, optional): Figure size for the plot. Defaults to (10, 8).
        perplexity (int, optional): Perplexity parameter for t-SNE. Defaults to 30.
        learning_rate (int, optional): Learning rate for t-SNE optimization. Defaults to 200.
        n_iter (int, optional): Number of iterations for t-SNE. Defaults to 1000.
    """
    # Extract Node2Vec embeddings and corresponding labels
    embeddings = data.n2v.detach().numpy()  # Convert to NumPy array
    labels = data.y.numpy()  # Convert to NumPy array

    # Apply t-SNE on the embeddings
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, n_iter=n_iter, random_state=42)
    tsne_results = tsne.fit_transform(embeddings)

    # Plot t-SNE results
    plt.figure(figsize=figsize)
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap="tab10", alpha=0.8, edgecolor='k')
    plt.colorbar(scatter, ticks=np.arange(len(class_labels)))
    plt.title("t-SNE Visualization of Node2Vec Embeddings")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True)

    # Add legend with class labels
    handles, _ = scatter.legend_elements()
    plt.legend(handles, class_labels, loc="best", title="Classes")

    plt.show()


# Load the Cora dataset
dataset = Planetoid(root="../data", name="Cora")
data = dataset[0]

# Assume Node2Vec embeddings are stored in `data.n2v` after pre-transform
class_labels = ["Case_Based", "Genetic_Algorithms", "Neural_Networks",
                "Probabilistic_Methods", "Reinforcement_Learning",
                "Rule_Learning", "Theory"]

# Plot the t-SNE visualization
plot_node2vec_tsne(data, class_labels)
