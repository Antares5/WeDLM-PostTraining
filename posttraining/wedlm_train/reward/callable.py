# coding=utf-8
"""Callable reward — load reward function from external module.
Migrated from dpo/src/reward.py (CallableReward)."""

import importlib
from typing import Callable as TypingCallable

import torch
from .base import BaseRewardFunction, RewardInputs


def _to_reward_tensor(raw_rewards, template: torch.Tensor) -> torch.Tensor:
    """Convert reward outputs into a tensor aligned to policy score shape/device."""
    if isinstance(raw_rewards, torch.Tensor):
        rewards = raw_rewards.to(device=template.device, dtype=template.dtype)
    else:
        rewards = torch.tensor(raw_rewards, device=template.device, dtype=template.dtype)

    if rewards.shape != template.shape:
        raise ValueError(
            f"Reward shape {tuple(rewards.shape)} must match policy score shape {tuple(template.shape)}"
        )

    return rewards


class CallableReward(BaseRewardFunction):
    """Reward function loaded from "module_path:function_name"."""

    def __init__(self, callable_spec: str):
        module_path, sep, function_name = callable_spec.partition(":")
        if sep == "" or not module_path or not function_name:
            raise ValueError(
                "gspo_reward_callable must be in the form 'module_path:function_name'"
            )

        module = importlib.import_module(module_path)
        reward_fn = getattr(module, function_name, None)
        if reward_fn is None or not callable(reward_fn):
            raise ValueError(f"Unable to load callable reward function: {callable_spec}")

        self.reward_fn: TypingCallable[[RewardInputs], torch.Tensor] = reward_fn

    def __call__(self, inputs: RewardInputs) -> torch.Tensor:
        rewards = self.reward_fn(inputs)
        return _to_reward_tensor(rewards, inputs.policy_scores)
