# coding=utf-8
"""GSPO training configuration (extends WeDLMTrainingConfig)."""

from dataclasses import dataclass
from typing import Optional

# Ensure dpo/src is on path so we can import the base config.
try:
    from src.config import WeDLMTrainingConfig
except ImportError:
    import sys, os
    _dpo = os.path.join(os.path.dirname(__file__), "..", "..", "dpo")
    sys.path.insert(0, _dpo)
    from src.config import WeDLMTrainingConfig


@dataclass
class GSPOConfig(WeDLMTrainingConfig):
    """Configuration for GSPO (Group-level Online RL) training on WeDLM."""
    training_mode: str = "gspo"

    # ── GRPO core ────────────────────────────────────────────────────
    gspo_group_size: int = 4                # G: completions per prompt
    gspo_clip_epsilon: float = 0.2          # ε: PPO clip range
    gspo_old_model_update_steps: int = 25   # sync old_policy every K steps

    # ── KL regularisation ────────────────────────────────────────────
    gspo_kl_beta: float = 0.01              # β_KL (0 = no KL penalty)
    gspo_kl_estimator: str = "k3"           # "k3" | "reverse_kl" | "none"

    # ── Scoring ──────────────────────────────────────────────────────
    gspo_num_mask_samples: int = 2          # K: mask MC samples for score

    # ── Generation ───────────────────────────────────────────────────
    gspo_gen_max_tokens: int = 256
    gspo_gen_temperature: float = 1.0
    gspo_gen_window_size: int = 16
    gspo_gen_entropy_threshold: Optional[float] = None
    gspo_gen_pos_penalty_factor: float = 0.02

    # ── Reward ───────────────────────────────────────────────────────
    gspo_reward_type: str = "math_verify"   # "math_verify" | "string_match"
    gspo_reward_model_path: Optional[str] = None  # (future) RM path

    # ── Data (prompt-only, no pairwise) ──────────────────────────────
    # train_data is reused from base class; expected JSONL format:
    # {"prompt": "user text", "ground_truth": "answer"}  or
    # {"messages": [...], "ground_truth": "answer"}

    # ── DPO scaffolding (not used by GSPO, keep defaults) ────────────
    dpo_train_data: Optional[str] = None
    dpo_beta: float = 0.1
    dpo_ref_model_path: Optional[str] = None
    dpo_length_norm: bool = True
    dpo_num_mask_samples: int = 1

    def __post_init__(self):
        super().__post_init__()

        if self.gspo_group_size < 2:
            raise ValueError("gspo_group_size must be >= 2")
        if self.gspo_clip_epsilon <= 0:
            raise ValueError("gspo_clip_epsilon must be positive")
        if self.gspo_old_model_update_steps < 1:
            raise ValueError("gspo_old_model_update_steps must be >= 1")
        if self.gspo_num_mask_samples < 1:
            raise ValueError("gspo_num_mask_samples must be >= 1")
        if self.gspo_kl_estimator not in ("k3", "reverse_kl", "none"):
            raise ValueError(f"Unknown kl_estimator: {self.gspo_kl_estimator}")
        if self.gspo_reward_type not in ("math_verify", "string_match"):
            raise ValueError(f"Unknown reward_type: {self.gspo_reward_type}")
