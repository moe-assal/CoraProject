import pandas as pd


def generate_readable_fake_name(class_name, node_id):
    """
    Generate a readable fake name in the format 'Paper_ClassName_ID'.
    """
    return f"Paper_{class_name.replace('_', '')}_{node_id}"


def process_content_file(content_file_path):
    """
    Processes the cora.content file to extract node IDs, class labels, and fake names.

    Args:
        content_file_path (str): Path to the cora.content file.

    Returns:
        pd.DataFrame: A DataFrame with columns: "Class", "Node ID", "Fake Name".
        dict: Mapping from paper IDs to numeric IDs.
    """
    # Load the cora.content file
    content_df = pd.read_csv(content_file_path, sep="\t", header=None)

    # Extract labels and assign unique numeric IDs
    labels = content_df.iloc[:, -1].values  # Last column contains the class labels
    paper_ids = content_df[0].values  # First column contains the paper IDs
    label_to_numeric = {label: idx for idx, label in enumerate(set(labels))}
    numeric_labels = [label_to_numeric[label] for label in labels]

    # Map paper IDs to node IDs and fake names
    numeric_to_label = {v: k for k, v in label_to_numeric.items()}
    nodes_with_fake_names = []
    paper_id_to_node_id = {}
    for node_id, (paper_id, label) in enumerate(zip(paper_ids, numeric_labels)):
        class_name = numeric_to_label[label]
        fake_name = generate_readable_fake_name(class_name, node_id)
        nodes_with_fake_names.append({"Class": class_name, "Node ID": node_id, "Fake Name": fake_name})
        paper_id_to_node_id[paper_id] = node_id

    return pd.DataFrame(nodes_with_fake_names), paper_id_to_node_id


def process_cites_file(cites_file_path, paper_id_to_node_id):
    """
    Processes the cora.cites file to map the edge list to numeric node IDs.

    Args:
        cites_file_path (str): Path to the cora.cites file.
        paper_id_to_node_id (dict): Mapping from paper IDs to numeric node IDs.

    Returns:
        pd.DataFrame: A DataFrame representing the mapped edge list.
    """
    # Load the cora.cites file
    cites_df = pd.read_csv(cites_file_path, sep="\t", header=None, names=["Source", "Target"])

    # Map paper IDs in the edge list to numeric node IDs
    cites_df["Source"] = cites_df["Source"].map(paper_id_to_node_id)
    cites_df["Target"] = cites_df["Target"].map(paper_id_to_node_id)

    # Return as a DataFrame with space-separated format
    return cites_df.dropna()


def main():
    # File paths
    content_file_path = r"C:\\Users\\mersh\\OneDrive\\Desktop\\CoraProject\\cora.content"  
    cites_file_path = r"C:\\Users\\mersh\\OneDrive\\Desktop\\CoraProject\\cora.cites"  

    # Step 1: Process the cora.content file
    nodes_with_fake_names_df, paper_id_to_node_id = process_content_file(content_file_path)

    # Save the nodes with fake names to a CSV file
    nodes_output_file_path = r"C:\\Users\\mersh\\OneDrive\\Desktop\\CoraProject\\CORA-UI\\cnodes_with_readable_fake_names.csv"
    nodes_with_fake_names_df.to_csv(nodes_output_file_path, index=False)
    print(f"Nodes grouped by class and mapped to readable fake names saved to {nodes_output_file_path}")

    # Step 2: Process the cora.cites file
    mapped_edges_df = process_cites_file(cites_file_path, paper_id_to_node_id)

    # Save the mapped edge list to a file in the "3 4" format
    edges_output_file_path = r"C:\\Users\\mersh\\OneDrive\\Desktop\\CoraProject\\CORA-UI\\mapped_edge_list.txt"
    mapped_edges_df.to_csv(edges_output_file_path, index=False, header=False, sep=" ")
    print(f"Mapped edge list saved to {edges_output_file_path}")


if __name__ == "__main__":
    main()

