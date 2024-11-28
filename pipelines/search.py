import torch
from typing import List


class SequentialSearch:
    def __init__(self, model_class, trainer_class, loss_class, loaders, param_options):
        self.model_class = model_class
        self.trainer_class = trainer_class
        self.loss_class = loss_class
        self.train_loader, self.val_loader, self.test_loader = loaders

        self.param_options = param_options
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.best_params = {key: (values[0] if isinstance(values, List) else values) for key, values in
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
            loss_func = self.loss_class(**current_params)
            trainer = self.trainer_class(
                model=model,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                test_loader=self.test_loader,
                loss_func=loss_func,
                **current_params
            )
            trainer.train(num_epochs=70)
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

    def run_search(self, num_times):
        """
        Perform sequential parameter optimization.
        """
        for _ in range(num_times):
            for param_name in self.param_options.keys():
                if not isinstance(self.param_options[param_name], List):
                    continue
                self.optimize_param(param_name)

        print("\n=== Sequential Search Completed ===")
        print(f"Optimized Parameters: {self.best_params}")

        return self.best_params
