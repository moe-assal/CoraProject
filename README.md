# CoraProject

CoraProject focuses on **graph-based machine learning** using the **Cora citation network dataset**. The project leverages **PyTorch Geometric** to design, train, and evaluate Graph Neural Networks (GNNs) for node classification tasks, specifically using architectures like **Graph Convolutional Networks (GCNs)**, **Jumping Knowledge Graphs (JKs)**, and **Graph Attention Networks (GATs)**.

## Features

- Node classification with the Cora dataset.
- Implementation of advanced GNNs.
- Input: Scientific paper abstracts and citation links.
- Output: Predicted research topics for papers.
- API hosted using Flask
- Simple UI in CORA-UI folder

## Dataset

The Cora dataset includes:
- **Nodes**: Scientific publications.
- **Edges**: Citation relationships.
- **Classes**: Research categories.

## Installation

### Prerequisites
- Python 3.8+
- PyTorch Geometric and dependencies.
- Flask
- PyQt5

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/moe-assal/CoraProject.git
   ```
2. Navigate to the directory:
   ```bash
   cd CoraProject
   ```
 
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the project: 
   ```bash
   python main.py
   ```
## Contact

For inquiries, please feel free to reach out:

- **Zeina Mershad**: [mershadzeina7@gmail.com](mailto:mershadzeina7@gmail.com)
- **Mohammad Asal**: [moritz.asal.04@gmail.com](mailto:moritz.asal.04@gmail.com)
