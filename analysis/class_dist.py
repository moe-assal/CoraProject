from torch_geometric.datasets import Planetoid
import matplotlib.pyplot as plt
from collections import Counter

# Load the Cora dataset
dataset = Planetoid(root='../data', name='Cora')
data = dataset[0]

# Extract the labels
labels = data.y

# Count the occurrences of each label
label_counts = Counter(labels.tolist())

# Plot the class distribution
plt.bar(label_counts.keys(), label_counts.values(), color='skyblue')
plt.xlabel('Class Label')
plt.ylabel('Frequency')
plt.title('Class Distribution in Cora Dataset')
plt.xticks(range(len(label_counts.keys())))
plt.show()
