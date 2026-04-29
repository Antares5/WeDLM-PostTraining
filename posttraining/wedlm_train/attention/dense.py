# coding=utf-8
"""Dense attention wrapper using PyTorch SDPA with 2D mask.
Migrated from dpo/src/attention.py (DenseAttentionWrapper)."""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseAttentionWrapper(nn.Module):
    """Dense attention using PyTorch SDPA with 2D mask."""

    def __init__(self, head_dim: int, softmax_scale: Optional[float] = None):
        super().__init__()
        self.head_dim = head_dim
        self.softmax_scale = softmax_scale or (head_dim ** -0.5)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask_2d: torch.Tensor,
    ) -> torch.Tensor:
        """Run dense attention with 2D mask.

        Args:
            q: Query tensor [T, H, D]
            k: Key tensor [T, Hkv, D]
            v: Value tensor [T, Hkv, D]
            attn_mask_2d: Attention mask [T, T] bool, True=ALLOW
        """
        T, H_q, D = q.shape
        H_kv = k.shape[1]

        # GQA expansion
        if H_q != H_kv:
            expand_ratio = H_q // H_kv
            k = k.repeat_interleave(expand_ratio, dim=1)
            v = v.repeat_interleave(expand_ratio, dim=1)

        # Convert mask: True=ALLOW -> 0.0, False -> -inf
        sdpa_mask = torch.where(attn_mask_2d, 0.0, float("-inf")).to(dtype=q.dtype, device=q.device)
        sdpa_mask = sdpa_mask.unsqueeze(0).unsqueeze(0)

        # Reshape: [T, H, D] -> [1, H, T, D]
        q_b = q.permute(1, 0, 2).unsqueeze(0)
        k_b = k.permute(1, 0, 2).unsqueeze(0)
        v_b = v.permute(1, 0, 2).unsqueeze(0)

        out = F.scaled_dot_product_attention(
            q_b, k_b, v_b,
            attn_mask=sdpa_mask,
            dropout_p=0.0,
            is_causal=False,
            scale=self.softmax_scale,
        )

        return out.squeeze(0).permute(1, 0, 2).contiguous()
