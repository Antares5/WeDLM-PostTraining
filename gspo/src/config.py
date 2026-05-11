# coding=utf-8
"""GSPO training configuration (extends WeDLMTrainingConfig)."""

from dataclasses import dataclass
from typing import Optional

from dpo.src.config import WeDLMTrainingConfig


@dataclass
class GSPOConfig(WeDLMTrainingConfig):
    """Configuration for GSPO on-policy RL training.

    Inherits all WeDLM SFT/DPO config fields and adds GSPO-specific ones.
    """

    # Override default training_mode
    training_mode: str = "gspo"

    # ── GSPO core ──
    gspo_group_size: int = 4           # G: responses sampled per prompt
    gspo_num_mask_samples: int = 4     # K: MC masking estimates per response

    # ── Generation ──
    gspo_temperature: float = 0.8       # sampling temperature during generation
    gspo_max_new_tokens: int = 512      # max tokens to generate per response
    gspo_entropy_threshold: float = 0.4  # WeDLM parallel-decoding entropy threshold
    gspo_pos_penalty_factor: float = 0.02  # WeDLM position penalty

    # ── Weight sync ──
    gspo_sync_every_n_steps: int = 10   # sync training weights → generator every N steps

    # ── Generator (engine) config ──
    gspo_window_size: int = 16          # WeDLM sliding window size for generation
    gspo_kvcache_block_size: int = 4096  # KV cache block size

    # ── Data ──
    # train_data: path to a JSONL of prompts (one per line, chat format)
    # reward_model: can be a local model path or an API endpoint
    reward_model_path: Optional[str] = None

    def __post_init__(self):
        # Let parent validate shared fields first
        super().__post_init__()

        # Override: we accept "gspo" as a valid training_mode
        # (parent only accepts "sft"/"dpo", so we re-check here)
        if self.training_mode not in ("sft", "dpo", "gspo"):
            raise ValueError(f"Unknown training_mode: {self.training_mode}")

        if self.gspo_group_size < 2:
            raise ValueError("gspo_group_size must be >= 2 (need at least 2 for group advantage)")

        if self.gspo_num_mask_samples < 1:
            raise ValueError("gspo_num_mask_samples must be >= 1")

        if self.gspo_temperature < 0:
            raise ValueError("gspo_temperature must be >= 0")
