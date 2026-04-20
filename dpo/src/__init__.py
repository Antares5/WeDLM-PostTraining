# coding=utf-8
"""WeDLM - A Weighted Diffusion Language Model for SFT training."""

from src.config import WeDLMTrainingConfig
from src.data import (
    WeDLMPackedDataset,
    WeDLMPairwiseDataset,
    WeDLMPromptDataset,
    WeDLMShuffledPackedDataset,
    packed_collate_fn,
    dpo_collate_fn,
    gspo_prompt_collate_fn,
    get_im_end_token_id,
)
from src.batch import WeDLMBatch, build_wedlm_batch
from src.model import wedlm_forward, wedlm_attention_forward
from src.loss import (
    compute_mlm_loss,
    compute_ar_loss,
    compute_masked_token_logps,
    compute_block_scores,
    compute_dpo_loss,
    compute_gspo_loss,
)
from src.attention import (
    check_backend_available,
    get_available_backend,
    get_attention_wrapper,
)
from src.reward import RewardInputs, BaseRewardFunction, build_reward_function
from src.trainer import WeDLMTrainer

__all__ = [
    # Config
    "WeDLMTrainingConfig",
    # Data
    "WeDLMPackedDataset",
    "WeDLMPairwiseDataset",
    "WeDLMPromptDataset",
    "WeDLMShuffledPackedDataset",
    "packed_collate_fn",
    "dpo_collate_fn",
    "gspo_prompt_collate_fn",
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
    "compute_dpo_loss",
    "compute_gspo_loss",
    # Attention
    "check_backend_available",
    "get_available_backend",
    "get_attention_wrapper",
    # Reward
    "RewardInputs",
    "BaseRewardFunction",
    "build_reward_function",
    # Trainer
    "WeDLMTrainer",
]

