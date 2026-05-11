# coding=utf-8
"""GSPO - Group Sample Policy Optimization for WeDLM (on-policy RL training)."""

from gspo.src.config import GSPOConfig
from gspo.src.loss import compute_gspo_loss, compute_block_scores
from gspo.src.batch import WeDLMBatch, build_wedlm_batch, build_wedlm_batch_from_response
from gspo.src.model import wedlm_forward, wedlm_attention_forward
from gspo.src.attention import (
    check_backend_available,
    get_available_backend,
    get_attention_wrapper,
)
from gspo.src.data import GSPOPromptDataset, gspo_collate_fn
from gspo.src.masking import (
    sample_block_mask_ratios,
    sample_mask_indices,
    reorder_block,
    build_2d_attention_mask,
    build_magi_plan,
)
from gspo.src.trainer import GSPOTrainer, GSPOMockResponseDataset

__all__ = [
    # Config
    "GSPOConfig",
    # Loss
    "compute_gspo_loss",
    "compute_block_scores",
    # Batch
    "WeDLMBatch",
    "build_wedlm_batch",
    "build_wedlm_batch_from_response",
    # Model
    "wedlm_forward",
    "wedlm_attention_forward",
    # Attention
    "check_backend_available",
    "get_available_backend",
    "get_attention_wrapper",
    # Data
    "GSPOPromptDataset",
    "gspo_collate_fn",
    # Masking
    "sample_block_mask_ratios",
    "sample_mask_indices",
    "reorder_block",
    "build_2d_attention_mask",
    "build_magi_plan",
    # Trainer
    "GSPOTrainer",
    "GSPOMockResponseDataset",
]
