# coding=utf-8
"""Clipped reward wrapper.
Migrated from dpo/src/reward.py (ClippedReward)."""

from typing import Optional
import torch
from .base import BaseRewardFunction, RewardInputs


class ClippedReward(BaseRewardFunction):
    """Optional reward clipping wrapper."""

    def __init__(
        self,
        base_reward_fn: BaseRewardFunction,
        clip_min: Optional[float],
        clip_max: Optional[float],
    ):
        self.base_reward_fn = base_reward_fn
        self.clip_min = clip_min
        self.clip_max = clip_max

    def __call__(self, inputs: RewardInputs) -> torch.Tensor:
        rewards = self.base_reward_fn(inputs)
        min_v = self.clip_min if self.clip_min is not None else float("-inf")
        max_v = self.clip_max if self.clip_max is not None else float("inf")
        return rewards.clamp(min=min_v, max=max_v)
