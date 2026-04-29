# coding=utf-8
"""Attention backend registry: availability checks and factory.
Migrated from dpo/src/attention.py (check_backend_available, get_available_backend,
get_attention_wrapper)."""

from typing import Optional
from .magi import _MAGI_AVAILABLE, MagiAttentionWrapper
from .dense import DenseAttentionWrapper


def check_backend_available(backend: str) -> bool:
    """Check if the specified attention backend is available."""
    if backend == "magi":
        return _MAGI_AVAILABLE
    elif backend == "dense":
        return True
    return False


def get_available_backend() -> str:
    """Get the best available backend."""
    if _MAGI_AVAILABLE:
        return "magi"
    return "dense"


def get_attention_wrapper(
    backend: str,
    head_dim: int,
    softmax_scale: Optional[float] = None,
    deterministic: bool = False,
):
    """Get the appropriate attention wrapper.

    Args:
        backend: "magi" or "dense"
        head_dim: Dimension per attention head
        softmax_scale: Optional custom softmax scale
        deterministic: Whether to use deterministic operations (magi only)
    """
    if backend == "magi":
        return MagiAttentionWrapper(head_dim, softmax_scale, deterministic=deterministic)
    elif backend == "dense":
        return DenseAttentionWrapper(head_dim, softmax_scale)
    else:
        raise ValueError(f"Unknown backend: {backend}. Choose from: magi, dense")
