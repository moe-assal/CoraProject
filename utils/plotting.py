import matplotlib.pyplot as plt

def plot_metrics(metrics, save_path=None):
    """
    Plots training and validation metrics over epochs.

    Args:
        metrics (dict): A dictionary containing lists of metrics for each epoch:
                        {'train_loss': [...], 'train_accuracy': [...],
                         'val_loss': [...], 'val_accuracy': [...]}
        save_path (str, optional): Path to save the plot. If None, displays the plot.
    """
    epochs = range(1, len(metrics["train_loss"]) + 1)

    # Plot loss
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, metrics["train_loss"], label="Train Loss", marker="o")
    plt.plot(epochs, metrics["val_loss"], label="Validation Loss", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, metrics["train_accuracy"], label="Train Accuracy", marker="o")
    plt.plot(epochs, metrics["val_accuracy"], label="Validation Accuracy", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)

    # Show or save the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    else:
        plt.show()