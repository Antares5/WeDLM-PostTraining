# coding=utf-8
"""Device / accelerator utility."""

import torch


def get_device() -> torch.device:
    """Get the best available device (cuda > mps > cpu)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
