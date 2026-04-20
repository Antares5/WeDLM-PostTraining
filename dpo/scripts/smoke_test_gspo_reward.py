#!/usr/bin/env python
# coding=utf-8
"""Smoke test for GSPO reward interface."""

import torch

from src import WeDLMTrainingConfig
from src.reward import RewardInputs, build_reward_function


# Example custom reward for callable mode.
def custom_reward(inputs: RewardInputs):
    lengths = torch.tensor(inputs.completion_lengths, dtype=inputs.policy_scores.dtype)
    return 0.01 * lengths


def _build_fake_inputs():
    prompt_input_ids = [torch.tensor([1, 2, 3]), torch.tensor([4, 5])]
    candidate_input_ids = [
        torch.tensor([1, 2, 3, 11, 12]),
        torch.tensor([1, 2, 3, 21, 22, 23]),
        torch.tensor([4, 5, 31]),
        torch.tensor([4, 5, 41, 42]),
    ]
    candidate_labels = []
    for idx, ids in enumerate(candidate_input_ids):
        labels = ids.clone()
        prompt_len = 3 if idx < 2 else 2
        labels[:prompt_len] = -100
        candidate_labels.append(labels)

    return RewardInputs(
        prompt_input_ids=prompt_input_ids,
        candidate_input_ids=candidate_input_ids,
        candidate_labels=candidate_labels,
        completion_lengths=[2, 3, 1, 2],
        group_ids=torch.tensor([0, 0, 1, 1], dtype=torch.long),
        policy_scores=torch.tensor([0.2, 0.0, 0.3, -0.1], dtype=torch.float32),
        reference_scores=torch.tensor([0.1, -0.05, 0.25, -0.2], dtype=torch.float32),
        tokenizer=None,
        prompt_metadata=[
            {"ground_truth_answer": "2"},
            {"ground_truth_answer": "4"},
        ],
        candidate_completion_texts=[
            "The answer is 2",
            "Final Answer: 3",
            "\\boxed{4}",
            "Final Answer: 5",
        ],
    )


def main():
    inputs = _build_fake_inputs()

    # Builtin: policy_ref_margin
    cfg = WeDLMTrainingConfig(training_mode="gspo")
    cfg.gspo_reward_source = "policy_ref_margin"
    cfg.gspo_reward_beta = 0.2
    reward_fn = build_reward_function(cfg)
    reward_a = reward_fn(inputs)

    # Builtin: length_penalized_margin
    cfg.gspo_reward_source = "length_penalized_margin"
    cfg.gspo_reward_length_penalty = 0.01
    reward_fn = build_reward_function(cfg)
    reward_b = reward_fn(inputs)

    # Callable reward
    cfg.gspo_reward_source = "callable"
    cfg.gspo_reward_callable = "__main__:custom_reward"
    reward_fn = build_reward_function(cfg)
    reward_c = reward_fn(inputs)

    # DeepMath correctness reward
    cfg.gspo_reward_source = "deepmath_correctness_margin"
    cfg.gspo_deepmath_correct_bonus = 1.0
    cfg.gspo_deepmath_wrong_penalty = 0.5
    reward_fn = build_reward_function(cfg)
    reward_d = reward_fn(inputs)

    print("=== GSPO Reward Interface Smoke Test ===")
    print(f"policy_ref_margin: {reward_a.tolist()}")
    print(f"length_penalized_margin: {reward_b.tolist()}")
    print(f"callable: {reward_c.tolist()}")
    print(f"deepmath_correctness_margin: {reward_d.tolist()}")
    print("GSPO reward interface smoke test passed.")


if __name__ == "__main__":
    main()
