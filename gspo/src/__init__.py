# coding=utf-8
"""GSPO - Group Sample Policy Optimization for WeDLM (on-policy RL training)."""

from gspo.src.config import GSPOConfig
from gspo.src.loss import compute_gspo_loss

__all__ = [
    "GSPOConfig",
    "compute_gspo_loss",
]
