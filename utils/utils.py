import random
import numpy as np
import torch

def set_seed(seed: int):
    """
    Set the random seed for reproducibility.

    Args:
        seed (int): Seed value.
    """
    # Set Python's random seed
    random.seed(seed)

    # Set NumPy random seed
    np.random.seed(seed)

    # Set PyTorch random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
