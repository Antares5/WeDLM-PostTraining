# coding=utf-8
"""Margin-based reward functions.
Migrated from dpo/src/reward.py (PolicyRefMarginReward, LengthPenalizedMarginReward)."""

import torch
from .base import BaseRewardFunction, RewardInputs


class PolicyRefMarginReward(BaseRewardFunction):
    """Default reward: beta * (policy_score - reference_score)."""

    def __init__(self, beta: float):
        self.beta = float(beta)

    def __call__(self, inputs: RewardInputs) -> torch.Tensor:
        return self.beta * (
            inputs.policy_scores.detach() - inputs.reference_scores.detach()
        )


class LengthPenalizedMarginReward(BaseRewardFunction):
    """Margin reward with a linear completion-length penalty."""

    def __init__(self, beta: float, length_penalty: float):
        self.beta = float(beta)
        self.length_penalty = float(length_penalty)

    def __call__(self, inputs: RewardInputs) -> torch.Tensor:
        base_reward = self.beta * (
            inputs.policy_scores.detach() - inputs.reference_scores.detach()
        )
        lengths = torch.tensor(
            inputs.completion_lengths,
            dtype=base_reward.dtype,
            device=base_reward.device,
        )
        return base_reward - self.length_penalty * lengths
