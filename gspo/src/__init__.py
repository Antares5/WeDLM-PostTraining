# coding=utf-8
"""GSPO - Group Sampling Policy Optimization for WeDLM Block Diffusion LM."""

from src.config import GSPOTrainingConfig
from src.data import GSPOPromptDataset, GSPOCollateFunction, get_im_end_token_id
from src.batch import WeDLMBatch, build_wedlm_batch
from src.model import wedlm_forward, wedlm_attention_forward
from src.loss import (
    compute_mlm_loss,
    compute_ar_loss,
    compute_masked_token_logps,
    compute_block_scores,
    compute_gspo_loss,
    compute_gspo_coefficients,
)
from src.attention import (
    check_backend_available,
    get_available_backend,
    get_attention_wrapper,
)
from src.generator import WeDLMGenerator
from src.reward import MathReward
from src.trainer import GSPOTrainer

__all__ = [
    # Config
    "GSPOTrainingConfig",
    # Data
    "GSPOPromptDataset",
    "GSPOCollateFunction",
    "get_im_end_token_id",
    # Batch
    "WeDLMBatch",
    "build_wedlm_batch",
    # Model
    "wedlm_forward",
    "wedlm_attention_forward",
    # Loss
    "compute_mlm_loss",
    "compute_ar_loss",
    "compute_masked_token_logps",
    "compute_block_scores",
    "compute_gspo_loss",
    "compute_gspo_coefficients",
    # Attention
    "check_backend_available",
    "get_available_backend",
    "get_attention_wrapper",
    # Generator
    "WeDLMGenerator",
    # Reward
    "MathReward",
    # Trainer
    "GSPOTrainer",
]
