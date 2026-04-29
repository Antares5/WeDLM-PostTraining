# coding=utf-8
"""Reward base classes and input container.
Migrated from dpo/src/reward.py (RewardInputs, BaseRewardFunction)."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch


@dataclass
class RewardInputs:
    """Container for GSPO reward function inputs."""

    prompt_input_ids: List[torch.Tensor]
    candidate_input_ids: List[torch.Tensor]
    candidate_labels: List[torch.Tensor]
    completion_lengths: List[int]
    group_ids: torch.Tensor
    policy_scores: torch.Tensor
    reference_scores: torch.Tensor
    tokenizer: object
    prompt_metadata: Optional[List[Dict[str, Any]]] = None
    candidate_completion_ids: Optional[List[torch.Tensor]] = None
    candidate_completion_texts: Optional[List[str]] = None


class BaseRewardFunction:
    """Base class for reward functions."""

    def __call__(self, inputs: RewardInputs) -> torch.Tensor:
        raise NotImplementedError
