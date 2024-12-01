import torch
from utils.accuracy_measures import f1_accuracy, confusion_matrix

class GNNTrainer:
    def __init__(self, model, train_loader, val_loader, test_loader, loss_func, **kwargs):
        """
        Initialize the trainer with model, data loaders, and device setup.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        lr = kwargs.get('lr', 0.01)

        self.model.to(self.device)
        self.criterion = loss_func
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=5e-4)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=5,
        )

    def train_one_epoch(self):
        """
        Trains the model for one epoch and returns the average loss.
        """
        self.model.train()
        total_loss = 0

        for batch in self.train_loader:
            batch = batch.to(self.device)  # Move batch to device
            self.optimizer.zero_grad()

            # Forward pass
            out = self.model(batch)
            ground_truth = batch.y[:batch.batch_size]
            loss = self.criterion(out, ground_truth)
            total_loss += loss.item()

            # Backward pass and optimizer step
            loss.backward()
            self.optimizer.step()

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def evaluate(self, loader):
        """
        Evaluate the model on the given loader.
        Returns accuracy and average loss.
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        for batch in loader:
            batch = batch.to(self.device)
            out = self.model(batch)
            ground_truth = batch.y[:batch.batch_size]
            loss = self.criterion(out, ground_truth)
            total_loss += loss.item()

            # Compute predictions
            preds = out.argmax(dim=1)
            correct += (preds == ground_truth).sum().item()
            total += ground_truth.size(0)

        accuracy = correct / total
        avg_loss = total_loss / len(loader)
        return accuracy, avg_loss

    def train(self, num_epochs=50, save_path="best_model.pth"):
        """
        Train the model for a specified number of epochs.
        """
        best_val_acc = 0

        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_one_epoch()
            val_acc, val_loss = self.evaluate(self.val_loader)

            print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.4f}")

            self.scheduler.step(val_acc)

            # Save the best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), save_path)
                print(f"Best model saved with Val Accuracy: {val_acc:.4f}")

        # Load the best model for testing
        self.model.load_state_dict(torch.load(save_path))
        test_acc, test_loss = self.evaluate(self.test_loader)
        print(f"Test Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")

    def predict(self, loader):
        """
        Predict labels for the given loader.
        Returns predictions and corresponding labels.
        """
        self.model.eval()
        predictions = []
        labels = []

        for batch in loader:
            batch = batch.to(self.device)
            out = self.model(batch)
            preds = out.argmax(dim=1)
            predictions.append(preds.cpu())
            labels.append(batch.y[:batch.size].cpu())

        return torch.cat(predictions), torch.cat(labels)

    @torch.no_grad()
    def compute_f1_score(self, loader):
        """
        Compute the F1-score for the given loader.
        """
        self.model.eval()
        all_preds = []
        all_labels = []

        for batch in loader:
            batch = batch.to(self.device)
            out = self.model(batch)

            # Predicted labels
            preds = out.argmax(dim=1).cpu()
            # Ground truth labels
            labels = batch.y[:batch.batch_size].cpu()

            all_preds.append(preds)
            all_labels.append(labels)

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        # Compute F1-score using sklearn
        f1 = f1_accuracy(all_preds, all_labels)

        return f1

    @torch.no_grad()
    def compute_confusion_matrix(self, loader):
        """
        Compute the Confusion Matrix for the given loader.
        """
        self.model.eval()
        all_preds = []
        all_labels = []

        for batch in loader:
            batch = batch.to(self.device)
            out = self.model(batch)

            # Predicted labels
            preds = out.argmax(dim=1).cpu()
            # Ground truth labels
            labels = batch.y[:batch.batch_size].cpu()

            all_preds.append(preds)
            all_labels.append(labels)

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        cm = confusion_matrix(all_preds, all_labels)

        return cm
