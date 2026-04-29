# coding=utf-8
"""SFT-specific training configuration."""

from dataclasses import dataclass
from .base import BaseTrainingConfig


@dataclass
class SFTConfig(BaseTrainingConfig):
    """Configuration for SFT (Supervised Fine-Tuning) training."""

    # ── WeDLM loss ──────────────────────────────────────────
    loss_weighting_scheme: str = "weighted"  # "weighted" (1/γ) or "uniform"

    # ── AR loss ─────────────────────────────────────────────
    enable_ar_loss: bool = True
    ar_loss_weight: float = 1.0

    def __post_init__(self):
        super().__post_init__()

        if self.loss_weighting_scheme not in ["uniform", "weighted"]:
            raise ValueError(
                f"Unknown loss_weighting_scheme: {self.loss_weighting_scheme}"
            )
