import torch

class SequentialSearch:
    def __init__(self, model_class, trainer_class, train_loader, val_loader, test_loader, param_options):
        """
        Sequential optimization of hyperparameters.

        Args:
            model_class (type): Class of the model to instantiate.
            trainer_class (type): Trainer class for training the model.
            train_loader (DataLoader): DataLoader for the training set.
            val_loader (DataLoader): DataLoader for the validation set.
            test_loader (DataLoader): DataLoader for the test set.
            param_options (dict): Hyperparameter options to search, with values as lists.
        """
        self.model_class = model_class
        self.trainer_class = trainer_class
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.param_options = param_options
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.best_params = {key: values[0] for key, values in
                            param_options.items()}  # Start with first option for each param
        self.results = []

    def optimize_param(self, param_name):
        """
        Optimize a single parameter by iterating through its values.

        Args:
            param_name (str): Name of the parameter to optimize.

        Returns:
            str, value: The name and the best value of the parameter.
        """
        best_val_acc = 0
        best_value = None

        print(f"Optimizing {param_name}...")
        for value in self.param_options[param_name]:
            # Update the parameter being optimized
            current_params = self.best_params.copy()
            current_params[param_name] = value

            print(f"Testing {param_name} = {value} with params: {current_params}")

            # Train and validate model
            model = self.model_class(**current_params).to(self.device)
            trainer = self.trainer_class(
                model=model,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                test_loader=self.test_loader,
                device=self.device,
            )
            trainer.train(num_epochs=50)
            val_acc, _ = trainer.evaluate(self.val_loader)

            print(f"Validation Accuracy for {param_name} = {value}: {val_acc:.4f}")

            # Track best value for this parameter
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_value = value

        # Update best params with the best value of this parameter
        self.best_params[param_name] = best_value
        print(f"Best {param_name}: {best_value} with Val Accuracy: {best_val_acc:.4f}")

        return param_name, best_value

    def run_search(self):
        """
        Perform sequential parameter optimization.
        """
        for param_name in self.param_options.keys():
            self.optimize_param(param_name)

        print("\n=== Sequential Search Completed ===")
        print(f"Optimized Parameters: {self.best_params}")

        return self.best_params
