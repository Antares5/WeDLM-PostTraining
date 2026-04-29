#!/usr/bin/env python
# coding=utf-8
"""Smoke test for GSPO loss utility (migrated to wedlm_train).

Usage:
    cd posttraining && python scripts/smoke_tests/test_gspo_loss.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from wedlm_train.loss import compute_gspo_loss


def main():
    torch.manual_seed(42)

    policy_scores = torch.tensor([0.1, 0.3, -0.2, 0.5, 0.2, -0.1], dtype=torch.float32)
    reference_scores = torch.tensor([0.0, 0.1, -0.1, 0.3, 0.15, -0.2], dtype=torch.float32)
    rewards = 0.1 * (policy_scores - reference_scores)
    group_ids = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)

    loss, logs = compute_gspo_loss(
        policy_scores=policy_scores,
        reference_scores=reference_scores,
        group_ids=group_ids,
        rewards=rewards,
        score_temperature=1.0,
        reward_temperature=1.0,
        ref_alpha=1.0,
        kl_coef=0.01,
    )

    print("=== GSPO Loss Smoke Test ===")
    print(f"loss: {float(loss.item()):.6f}")
    for key in sorted(logs.keys()):
        value = logs[key]
        if hasattr(value, "item"):
            value = float(value.item())
        print(f"{key}: {value:.6f}")
    print("GSPO loss smoke test passed.")


if __name__ == "__main__":
    main()
