# coding=utf-8
"""Magi attention wrapper for Flex Flash Attention.
Migrated from dpo/src/attention.py (MagiAttentionWrapper)."""

from typing import Optional, Dict
import torch
import torch.nn as nn

# Try to import magi_attention (optional)
_MAGI_AVAILABLE = False
try:
    from magi_attention.functional.flex_flash_attn import flex_flash_attn_func
    _MAGI_AVAILABLE = True
except ImportError:
    flex_flash_attn_func = None


class MagiAttentionWrapper(nn.Module):
    """Wrapper for Magi Flex Flash Attention."""

    def __init__(
        self,
        head_dim: int,
        softmax_scale: Optional[float] = None,
        deterministic: bool = False,
    ):
        super().__init__()
        if not _MAGI_AVAILABLE:
            raise ImportError(
                "magi_attention is not installed. Install with: pip install magi-attention"
            )
        self.head_dim = head_dim
        self.softmax_scale = softmax_scale or (head_dim ** -0.5)
        self.deterministic = deterministic

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        magi_plan: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Run Magi flex flash attention.

        Args:
            q: Query tensor [T, H, D]
            k: Key tensor [T, Hkv, D]
            v: Value tensor [T, Hkv, D]
            magi_plan: Dict containing q_ranges, k_ranges, attn_type_map,
                       max_seqlen_q, max_seqlen_k
        """
        out, _ = flex_flash_attn_func(
            q if q.is_contiguous() else q.contiguous(),
            k if k.is_contiguous() else k.contiguous(),
            v if v.is_contiguous() else v.contiguous(),
            q_ranges=magi_plan["q_ranges"],
            k_ranges=magi_plan["k_ranges"],
            max_seqlen_q=magi_plan["max_seqlen_q"],
            max_seqlen_k=magi_plan["max_seqlen_k"],
            attn_type_map=magi_plan["attn_type_map"],
            softmax_scale=self.softmax_scale,
            softcap=0.0,
            deterministic=self.deterministic,
        )
        return out
