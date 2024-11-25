import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import Node2Vec


def node2vec_transform(data: Data, embedding_dim=128, walk_length=10, context_size=5, walks_per_node=10, num_negative_samples=1,
                      epochs=100, lr=0.01):
    print("Applying Node2Vec transformation...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    # Initialize Node2Vec
    node2vec = Node2Vec(
        edge_index=data.edge_index,
        embedding_dim=embedding_dim,
        walk_length=walk_length,
        context_size=context_size,
        walks_per_node=walks_per_node,
        num_negative_samples=num_negative_samples,
        sparse=True
    ).to(device)

    # Train Node2Vec
    optimizer = torch.optim.SparseAdam(node2vec.parameters(), lr=lr)

    def train():
        node2vec.train()
        for epoch in range(epochs):
            perm = torch.randperm(data.num_nodes, device=device)
            total_loss = 0
            for i in range(0, data.num_nodes, 128):  # Batch size = 128
                batch = perm[i: i + 128]
                pos_rw = node2vec.pos_sample(batch)
                neg_rw = node2vec.neg_sample(batch)

                optimizer.zero_grad()
                loss = node2vec.loss(pos_rw, neg_rw)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

    train()
    data.n2v = node2vec(torch.arange(data.num_nodes, device=device)).cpu()

    return data


if __name__ == '__main__':
    dataset_path = '../data/'

    dataset = Planetoid(root=dataset_path, name='Cora', pre_transform=node2vec_transform, force_reload=True)
    data = dataset[0]  # The processed data now includes Node2Vec embeddings

    # Step 3: Save the Processed Dataset (Already Done by PyG)
    print(f"Processed dataset saved at: {dataset_path}")
    print(f"Dataset features shape: {data.n2v.shape}")
