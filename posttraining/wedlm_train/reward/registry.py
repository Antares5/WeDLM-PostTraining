# coding=utf-8
"""Reward registry: factory to build reward function from config.
Migrated from dpo/src/reward.py (build_reward_function)."""

from .base import BaseRewardFunction
from .margin import PolicyRefMarginReward, LengthPenalizedMarginReward
from .deepmath import DeepMathCorrectnessMarginReward
from .callable import CallableReward
from .clipped import ClippedReward


def build_reward_function(config) -> BaseRewardFunction:
    """Build reward function according to GSPO config.

    Args:
        config: A config object (e.g. GSPOConfig) with gspo_reward_* attributes.
    """
    source = config.gspo_reward_source

    if source == "policy_ref_margin":
        reward_fn: BaseRewardFunction = PolicyRefMarginReward(
            beta=config.gspo_reward_beta
        )
    elif source == "length_penalized_margin":
        reward_fn = LengthPenalizedMarginReward(
            beta=config.gspo_reward_beta,
            length_penalty=config.gspo_reward_length_penalty,
        )
    elif source == "deepmath_correctness_margin":
        reward_fn = DeepMathCorrectnessMarginReward(
            beta=config.gspo_reward_beta,
            correct_bonus=config.gspo_deepmath_correct_bonus,
            wrong_penalty=config.gspo_deepmath_wrong_penalty,
            length_penalty=config.gspo_reward_length_penalty,
            numeric_atol=config.gspo_deepmath_numeric_atol,
            numeric_rtol=config.gspo_deepmath_numeric_rtol,
            penalize_only_when_confident=config.gspo_deepmath_penalize_only_when_confident,
        )
    elif source == "callable":
        if not config.gspo_reward_callable:
            raise ValueError(
                "gspo_reward_callable must be set when gspo_reward_source='callable'"
            )
        reward_fn = CallableReward(config.gspo_reward_callable)
    else:
        raise ValueError(f"Unknown gspo_reward_source: {source}")

    if config.gspo_reward_clip_min is not None or config.gspo_reward_clip_max is not None:
        reward_fn = ClippedReward(
            base_reward_fn=reward_fn,
            clip_min=config.gspo_reward_clip_min,
            clip_max=config.gspo_reward_clip_max,
        )

    return reward_fn
