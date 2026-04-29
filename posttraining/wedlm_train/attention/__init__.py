# attention - Dense / Magi attention wrapper
from .dense import DenseAttentionWrapper
from .magi import MagiAttentionWrapper
from .registry import (
    check_backend_available,
    get_available_backend,
    get_attention_wrapper,
)

__all__ = [
    "DenseAttentionWrapper",
    "MagiAttentionWrapper",
    "check_backend_available",
    "get_available_backend",
    "get_attention_wrapper",
]
