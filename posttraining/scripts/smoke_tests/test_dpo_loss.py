#!/usr/bin/env python
# coding=utf-8
"""Smoke test for DPO loss utilities (migrated to wedlm_train).

Usage:
    cd posttraining && python scripts/smoke_tests/test_dpo_loss.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from wedlm_train.loss import compute_block_scores, compute_dpo_loss


def _build_fake_stream(total_tokens: int, vocab_size: int, masked_ratio: float = 0.35):
    logits = torch.randn(total_tokens, vocab_size)
    targets = torch.randint(low=0, high=vocab_size, size=(total_tokens,), dtype=torch.long)

    masked_indices = torch.zeros(total_tokens, dtype=torch.bool)
    num_masked = max(1, int(total_tokens * masked_ratio))
    masked_positions = torch.randperm(total_tokens)[:num_masked]
    masked_indices[masked_positions] = True

    p_mask = torch.zeros(total_tokens, dtype=torch.float32)
    p_mask[masked_indices] = torch.empty(num_masked).uniform_(0.1, 0.9)

    return logits, targets, masked_indices, p_mask


def main():
    torch.manual_seed(42)

    block_size = 8
    vocab_size = 128

    cum_seqlens = torch.tensor([0, 32, 64], dtype=torch.long)
    logical_positions = torch.cat([
        torch.arange(16), torch.arange(16),
        torch.arange(16), torch.arange(16),
    ]).long()

    pc_logits, pc_targets, pc_masked, pc_pmask = _build_fake_stream(64, vocab_size)
    pr_logits, pr_targets, pr_masked, pr_pmask = _build_fake_stream(64, vocab_size)
    rc_logits, rc_targets, rc_masked, rc_pmask = _build_fake_stream(64, vocab_size)
    rr_logits, rr_targets, rr_masked, rr_pmask = _build_fake_stream(64, vocab_size)

    pc_scores, pc_logs = compute_block_scores(
        logits=pc_logits, targets=pc_targets, masked_indices=pc_masked,
        p_mask=pc_pmask, logical_positions=logical_positions,
        cum_seqlens=cum_seqlens, block_size=block_size,
        weighting_scheme="weighted", block_reduce="mean", seq_reduce="mean",
    )
    pr_scores, _ = compute_block_scores(
        logits=pr_logits, targets=pr_targets, masked_indices=pr_masked,
        p_mask=pr_pmask, logical_positions=logical_positions,
        cum_seqlens=cum_seqlens, block_size=block_size,
        weighting_scheme="weighted", block_reduce="mean", seq_reduce="mean",
    )
    rc_scores, _ = compute_block_scores(
        logits=rc_logits, targets=rc_targets, masked_indices=rc_masked,
        p_mask=rc_pmask, logical_positions=logical_positions,
        cum_seqlens=cum_seqlens, block_size=block_size,
        weighting_scheme="weighted", block_reduce="mean", seq_reduce="mean",
    )
    rr_scores, _ = compute_block_scores(
        logits=rr_logits, targets=rr_targets, masked_indices=rr_masked,
        p_mask=rr_pmask, logical_positions=logical_positions,
        cum_seqlens=cum_seqlens, block_size=block_size,
        weighting_scheme="weighted", block_reduce="mean", seq_reduce="mean",
    )

    loss, logs = compute_dpo_loss(
        policy_chosen_scores=pc_scores,
        policy_rejected_scores=pr_scores,
        reference_chosen_scores=rc_scores,
        reference_rejected_scores=rr_scores,
        beta=0.1,
    )

    print("=== DPO Loss Smoke Test ===")
    print(f"loss: {float(loss.item()):.6f}")
    for key in sorted(logs.keys()):
        value = logs[key]
        if hasattr(value, "item"):
            value = float(value.item())
        print(f"{key}: {value:.6f}")
    print("DPO loss smoke test passed.")


if __name__ == "__main__":
    main()
