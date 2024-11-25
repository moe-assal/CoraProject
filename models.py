import torch

# PyTorch version
print(f"PyTorch version: {torch.__version__}")

# CUDA availability
print(f"Is CUDA available: {torch.cuda.is_available()}")

# CUDA version (if available)
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")

# Device information
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

