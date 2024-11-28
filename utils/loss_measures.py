import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for classification tasks.

    Args:
        alpha (float, optional): Weighting factor for the positive class. Default is 1.0.
        gamma (float, optional): Focusing parameter to down-weight easy examples. Default is 2.0.
    """

    def __init__(self, alpha=1.0, gamma=2.0, **kwargs):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        """
        Compute the Focal Loss.

        Args:
            inputs (torch.Tensor): Logits (unnormalized predictions) of shape (batch_size, num_classes).
            targets (torch.Tensor): Ground truth labels of shape (batch_size,).

        Returns:
            torch.Tensor: The computed focal loss.
        """
        # Apply softmax to logits to get probabilities
        probs = F.softmax(inputs, dim=-1)

        # Gather the probabilities corresponding to the target classes
        probs = probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

        # Compute the focal loss
        focal_weight = (1.0 - probs) ** self.gamma
        loss = -self.alpha * focal_weight * torch.log(probs + 1e-8)  # Add epsilon to avoid log(0)

        return loss.mean()