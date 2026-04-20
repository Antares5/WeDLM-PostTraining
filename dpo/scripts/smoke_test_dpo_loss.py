#!/usr/bin/env python
# coding=utf-8
"""Smoke test for increment-3 DPO scoring utilities.

This script validates:
1) compute_block_scores on synthetic masked WeDLM streams
2) compute_dpo_loss from chosen/rejected policy/reference scores

Usage:
    python scripts/smoke_test_dpo_loss.py
"""

import torch

from src.loss import compute_block_scores, compute_dpo_loss


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

    # 2 sequences in one packed stream.
    # Sequence lengths in packed domain are 2*L each in WeDLM.
    cum_seqlens = torch.tensor([0, 32, 64], dtype=torch.long)
    logical_positions = torch.cat([
        torch.arange(16),
        torch.arange(16),
        torch.arange(16),
        torch.arange(16),
    ]).long()

    # Policy chosen / rejected
    pc_logits, pc_targets, pc_masked, pc_pmask = _build_fake_stream(64, vocab_size)
    pr_logits, pr_targets, pr_masked, pr_pmask = _build_fake_stream(64, vocab_size)

    # Reference chosen / rejected
    rc_logits, rc_targets, rc_masked, rc_pmask = _build_fake_stream(64, vocab_size)
    rr_logits, rr_targets, rr_masked, rr_pmask = _build_fake_stream(64, vocab_size)

    pc_scores, pc_logs = compute_block_scores(
        logits=pc_logits,
        targets=pc_targets,
        masked_indices=pc_masked,
        p_mask=pc_pmask,
        logical_positions=logical_positions,
        cum_seqlens=cum_seqlens,
        block_size=block_size,
        weighting_scheme="weighted",
        block_reduce="mean",
        seq_reduce="mean",
    )
    pr_scores, _ = compute_block_scores(
        logits=pr_logits,
        targets=pr_targets,
        masked_indices=pr_masked,
        p_mask=pr_pmask,
        logical_positions=logical_positions,
        cum_seqlens=cum_seqlens,
        block_size=block_size,
        weighting_scheme="weighted",
        block_reduce="mean",
        seq_reduce="mean",
    )
    rc_scores, _ = compute_block_scores(
        logits=rc_logits,
        targets=rc_targets,
        masked_indices=rc_masked,
        p_mask=rc_pmask,
        logical_positions=logical_positions,
        cum_seqlens=cum_seqlens,
        block_size=block_size,
        weighting_scheme="weighted",
        block_reduce="mean",
        seq_reduce="mean",
    )
    rr_scores, _ = compute_block_scores(
        logits=rr_logits,
        targets=rr_targets,
        masked_indices=rr_masked,
        p_mask=rr_pmask,
        logical_positions=logical_positions,
        cum_seqlens=cum_seqlens,
        block_size=block_size,
        weighting_scheme="weighted",
        block_reduce="mean",
        seq_reduce="mean",
    )

    dpo_loss, dpo_logs = compute_dpo_loss(
        policy_chosen_scores=pc_scores,
        policy_rejected_scores=pr_scores,
        reference_chosen_scores=rc_scores,
        reference_rejected_scores=rr_scores,
        beta=0.1,
    )

    print("=== Increment-3 DPO Loss Smoke Test ===")
    print(f"policy_chosen_scores shape: {tuple(pc_scores.shape)}")
    print(f"policy_rejected_scores shape: {tuple(pr_scores.shape)}")
    print(f"reference_chosen_scores shape: {tuple(rc_scores.shape)}")
    print(f"reference_rejected_scores shape: {tuple(rr_scores.shape)}")
    print(f"dpo_loss: {dpo_loss.item():.6f}")

    print("\nBlock score logs:")
    for k, v in pc_logs.items():
        print(f"  {k}: {float(v.item()):.6f}")

    print("\nDPO logs:")
    for k, v in dpo_logs.items():
        print(f"  {k}: {float(v.item()):.6f}")

    print("Smoke test passed.")


if __name__ == "__main__":
    main()
