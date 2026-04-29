# coding=utf-8
"""DPO-specific training configuration."""

from dataclasses import dataclass
from typing import Optional
from .sft import SFTConfig


@dataclass
class DPOConfig(SFTConfig):
    """Configuration for DPO (Direct Preference Optimization) training."""

    # ── DPO scaffold ────────────────────────────────────────
    dpo_train_data: Optional[str] = None
    dpo_beta: float = 0.1
    dpo_ref_model_path: Optional[str] = None
    dpo_length_norm: bool = True
    dpo_block_reduce: str = "mean"  # "mean" or "sum"
    dpo_seq_reduce: str = "mean"   # "mean" or "sum"
    dpo_num_mask_samples: int = 1

    def __post_init__(self):
        super().__post_init__()

        if self.dpo_block_reduce not in ["mean", "sum"]:
            raise ValueError(
                f"Unknown dpo_block_reduce: {self.dpo_block_reduce}"
            )

        if self.dpo_seq_reduce not in ["mean", "sum"]:
            raise ValueError(
                f"Unknown dpo_seq_reduce: {self.dpo_seq_reduce}"
            )

        if self.dpo_num_mask_samples < 1:
            raise ValueError("dpo_num_mask_samples must be >= 1")
