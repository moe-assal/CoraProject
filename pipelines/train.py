import torch


class GNNTrainer:
    def __init__(self, model, train_loader, val_loader, test_loader, lr=0.01):
        """
        Initialize the trainer with model, data loaders, and device setup.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
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
            out = self.model(batch.x, batch.edge_index)
            loss = self.criterion(out[batch.batch], batch.y[batch.batch])
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
            out = self.model(batch.x, batch.edge_index)
            loss = self.criterion(out[batch.batch], batch.y[batch.batch])
            total_loss += loss.item()

            # Compute predictions
            preds = out.argmax(dim=1)[batch.batch]
            correct += (preds == batch.y[batch.batch]).sum().item()
            total += batch.y[batch.batch].size(0)

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
            out = self.model(batch.x, batch.edge_index)
            preds = out.argmax(dim=1)[batch.batch]
            predictions.append(preds.cpu())
            labels.append(batch.y[batch.batch].cpu())

        return torch.cat(predictions), torch.cat(labels)
