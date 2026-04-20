# coding=utf-8
"""Reward interfaces for online GSPO training."""

from dataclasses import dataclass
import importlib
import re
from typing import Any, Callable, Dict, List, Optional

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


class PolicyRefMarginReward(BaseRewardFunction):
    """Default reward: beta * (policy_score - reference_score)."""

    def __init__(self, beta: float):
        self.beta = float(beta)

    def __call__(self, inputs: RewardInputs) -> torch.Tensor:
        return self.beta * (inputs.policy_scores.detach() - inputs.reference_scores.detach())


class LengthPenalizedMarginReward(BaseRewardFunction):
    """Margin reward with a linear completion-length penalty."""

    def __init__(self, beta: float, length_penalty: float):
        self.beta = float(beta)
        self.length_penalty = float(length_penalty)

    def __call__(self, inputs: RewardInputs) -> torch.Tensor:
        base_reward = self.beta * (inputs.policy_scores.detach() - inputs.reference_scores.detach())
        lengths = torch.tensor(
            inputs.completion_lengths,
            dtype=base_reward.dtype,
            device=base_reward.device,
        )
        return base_reward - self.length_penalty * lengths


def _extract_boxed_answer(text: str) -> Optional[str]:
    matches = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if matches:
        return matches[-1]
    return None


def _normalize_math_answer(text: str) -> str:
    normalized = text.strip()
    normalized = normalized.replace("$", "")
    normalized = normalized.replace(" ", "")
    normalized = normalized.replace("\\,", "")
    normalized = normalized.rstrip(".。")
    return normalized


def extract_math_final_answer(text: str) -> str:
    """Extract a final-answer string from free-form math solution text."""
    boxed = _extract_boxed_answer(text)
    if boxed is not None:
        return _normalize_math_answer(boxed)

    marker_patterns = [
        r"####\s*(.+)$",
        r"Final\s*Answer\s*[:：]\s*(.+)$",
        r"Answer\s*[:：]\s*(.+)$",
    ]
    for pattern in marker_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if match is not None:
            return _normalize_math_answer(match.group(1))

    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if len(lines) == 0:
        return ""
    return _normalize_math_answer(lines[-1])


class DeepMathCorrectnessMarginReward(BaseRewardFunction):
    """DeepMath reward combining margin shaping and exact-match correctness."""

    def __init__(
        self,
        beta: float,
        correct_bonus: float,
        wrong_penalty: float = 0.0,
        length_penalty: float = 0.0,
    ):
        self.beta = float(beta)
        self.correct_bonus = float(correct_bonus)
        self.wrong_penalty = float(wrong_penalty)
        self.length_penalty = float(length_penalty)

    def _resolve_completion_texts(self, inputs: RewardInputs) -> Optional[List[str]]:
        if inputs.candidate_completion_texts is not None:
            return inputs.candidate_completion_texts

        if inputs.candidate_completion_ids is None:
            return None

        if inputs.tokenizer is None:
            return None

        texts: List[str] = []
        for completion_ids in inputs.candidate_completion_ids:
            texts.append(
                inputs.tokenizer.decode(completion_ids.tolist(), skip_special_tokens=True)
            )
        return texts

    def __call__(self, inputs: RewardInputs) -> torch.Tensor:
        rewards = self.beta * (inputs.policy_scores.detach() - inputs.reference_scores.detach())

        completion_texts = self._resolve_completion_texts(inputs)
        prompt_metadata = inputs.prompt_metadata
        if completion_texts is not None and prompt_metadata is not None:
            for i, completion_text in enumerate(completion_texts):
                group_idx = int(inputs.group_ids[i].item())
                if group_idx < 0 or group_idx >= len(prompt_metadata):
                    continue

                metadata = prompt_metadata[group_idx] or {}
                gold_text = metadata.get("ground_truth_answer")
                if not isinstance(gold_text, str) or len(gold_text.strip()) == 0:
                    continue

                pred_answer = extract_math_final_answer(completion_text)
                gold_answer = extract_math_final_answer(gold_text)

                if pred_answer == "" or gold_answer == "":
                    continue

                if pred_answer == gold_answer:
                    rewards[i] = rewards[i] + self.correct_bonus
                elif self.wrong_penalty > 0:
                    rewards[i] = rewards[i] - self.wrong_penalty

        if self.length_penalty > 0:
            lengths = torch.tensor(
                inputs.completion_lengths,
                dtype=rewards.dtype,
                device=rewards.device,
            )
            rewards = rewards - self.length_penalty * lengths

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

        self.reward_fn: Callable[[RewardInputs], torch.Tensor] = reward_fn

    def __call__(self, inputs: RewardInputs) -> torch.Tensor:
        rewards = self.reward_fn(inputs)
        return _to_reward_tensor(rewards, inputs.policy_scores)


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


def build_reward_function(config) -> BaseRewardFunction:
    """Build reward function according to GSPO config."""
    source = config.gspo_reward_source

    if source == "policy_ref_margin":
        reward_fn: BaseRewardFunction = PolicyRefMarginReward(beta=config.gspo_reward_beta)
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
        )
    elif source == "callable":
        if not config.gspo_reward_callable:
            raise ValueError("gspo_reward_callable must be set when gspo_reward_source='callable'")
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
