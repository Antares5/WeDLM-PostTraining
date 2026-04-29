# coding=utf-8
"""GSPO-specific training configuration."""

from dataclasses import dataclass
from typing import Optional
from .sft import SFTConfig


@dataclass
class GSPOConfig(SFTConfig):
    """Configuration for GSPO (Group-level Sequence Preference Optimization) training."""

    # ── Online GSPO ─────────────────────────────────────────
    gspo_train_data: Optional[str] = None
    gspo_ref_model_path: Optional[str] = None
    gspo_group_size: int = 4
    gspo_min_candidates_per_group: int = 2
    gspo_max_prompt_length: int = 1536
    gspo_max_new_tokens: int = 256
    gspo_rollout_temperature: float = 0.8
    gspo_rollout_top_p: float = 0.95
    gspo_rollout_top_k: int = 0
    gspo_score_temperature: float = 1.0
    gspo_reward_temperature: float = 1.0
    gspo_ref_alpha: float = 1.0
    gspo_kl_coef: float = 0.0
    gspo_reward_beta: float = 0.1
    gspo_reward_source: str = "policy_ref_margin"
    gspo_reward_length_penalty: float = 0.0
    gspo_reward_callable: Optional[str] = None
    gspo_reward_clip_min: Optional[float] = None
    gspo_reward_clip_max: Optional[float] = None

    # ── DeepMath reward ─────────────────────────────────────
    gspo_deepmath_correct_bonus: float = 1.0
    gspo_deepmath_wrong_penalty: float = 0.0
    gspo_deepmath_numeric_atol: float = 1e-6
    gspo_deepmath_numeric_rtol: float = 1e-5
    gspo_deepmath_penalize_only_when_confident: bool = True

    def __post_init__(self):
        super().__post_init__()

        if self.gspo_group_size < 2:
            raise ValueError("gspo_group_size must be >= 2")

        if self.gspo_min_candidates_per_group < 2:
            raise ValueError("gspo_min_candidates_per_group must be >= 2")

        if self.gspo_min_candidates_per_group > self.gspo_group_size:
            raise ValueError(
                "gspo_min_candidates_per_group must be <= gspo_group_size"
            )

        if self.gspo_max_prompt_length < 1:
            raise ValueError("gspo_max_prompt_length must be >= 1")

        if self.gspo_max_new_tokens < 1:
            raise ValueError("gspo_max_new_tokens must be >= 1")

        if self.gspo_rollout_temperature < 0:
            raise ValueError("gspo_rollout_temperature must be non-negative")

        if not (0.0 < self.gspo_rollout_top_p <= 1.0):
            raise ValueError("gspo_rollout_top_p must be in (0, 1]")

        if self.gspo_rollout_top_k < 0:
            raise ValueError("gspo_rollout_top_k must be non-negative")

        if self.gspo_score_temperature <= 0:
            raise ValueError("gspo_score_temperature must be positive")

        if self.gspo_reward_temperature <= 0:
            raise ValueError("gspo_reward_temperature must be positive")

        if self.gspo_kl_coef < 0:
            raise ValueError("gspo_kl_coef must be non-negative")

        if self.gspo_reward_beta <= 0:
            raise ValueError("gspo_reward_beta must be positive")

        if self.gspo_reward_source not in [
            "policy_ref_margin",
            "length_penalized_margin",
            "deepmath_correctness_margin",
            "callable",
        ]:
            raise ValueError(
                f"Unknown gspo_reward_source: {self.gspo_reward_source}"
            )

        if self.gspo_reward_length_penalty < 0:
            raise ValueError("gspo_reward_length_penalty must be non-negative")

        if self.gspo_deepmath_correct_bonus < 0:
            raise ValueError("gspo_deepmath_correct_bonus must be non-negative")

        if self.gspo_deepmath_wrong_penalty < 0:
            raise ValueError("gspo_deepmath_wrong_penalty must be non-negative")

        if self.gspo_deepmath_numeric_atol < 0:
            raise ValueError("gspo_deepmath_numeric_atol must be non-negative")

        if self.gspo_deepmath_numeric_rtol < 0:
            raise ValueError("gspo_deepmath_numeric_rtol must be non-negative")

        if self.gspo_reward_source == "callable":
            if not self.gspo_reward_callable:
                raise ValueError(
                    "gspo_reward_callable must be set when using callable reward source"
                )

        if (
            self.gspo_reward_clip_min is not None
            and self.gspo_reward_clip_max is not None
            and self.gspo_reward_clip_min > self.gspo_reward_clip_max
        ):
            raise ValueError("gspo_reward_clip_min must be <= gspo_reward_clip_max")
