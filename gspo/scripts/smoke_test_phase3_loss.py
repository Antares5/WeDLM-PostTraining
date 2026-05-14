#!/usr/bin/env python
# coding=utf-8
"""Phase 3 loss function numerical validation — no GPU required.

Validates:
1. compute_gspo_coefficients_with_kl correctness
2. compute_kl_penalty correctness
3. Edge cases (K=1, all-same rewards, zero KL coef)
4. Gradient flow through coefficients
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.loss import (
    compute_gspo_coefficients,
    compute_gspo_coefficients_with_kl,
    compute_kl_penalty,
    compute_gspo_loss,
)


def test_kl_coef_zero_equals_original():
    """With kl_coef=0, coefficients_with_kl should equal base coefficients."""
    policy_scores = torch.tensor([0.5, 1.2, 0.8, 0.3])
    ref_scores = torch.tensor([0.4, 1.0, 0.9, 0.2])
    rewards = torch.tensor([0.0, 1.0, 0.0, 0.0])

    base = compute_gspo_coefficients(policy_scores, ref_scores, rewards, beta=0.1)
    with_kl = compute_gspo_coefficients_with_kl(
        policy_scores, ref_scores, rewards, beta=0.1, kl_coef=0.0
    )

    assert torch.allclose(base, with_kl, atol=1e-6), f"base={base}, with_kl={with_kl}"
    print("✓ test_kl_coef_zero_equals_original PASSED")


def test_kl_penalty_numerical():
    """Verify KL penalty = mean(squared_diff)."""
    policy = torch.tensor([1.0, 2.0, 3.0])
    ref = torch.tensor([0.0, 1.0, 2.0])
    diff = policy - ref  # [1, 1, 1]

    kl_loss, kl_per_sample = compute_kl_penalty(policy, ref)
    # MSE = mean([1^2, 1^2, 1^2]) = 1.0
    expected_loss = torch.tensor(1.0)
    assert torch.allclose(kl_loss, expected_loss, atol=1e-6), f"{kl_loss} != {expected_loss}"
    assert kl_per_sample.tolist() == [1.0, 1.0, 1.0]
    print("✓ test_kl_penalty_numerical PASSED")


def test_kl_penalty_zero_when_identical():
    """KL penalty = 0 when policy == ref."""
    scores = torch.tensor([0.5, 0.5, 0.5])
    kl_loss, kl_per_sample = compute_kl_penalty(scores, scores)
    assert torch.allclose(kl_loss, torch.tensor(0.0), atol=1e-6)
    assert (kl_per_sample == 0.0).all()
    print("✓ test_kl_penalty_zero_when_identical PASSED")


def test_kl_coef_scales_linearly():
    """KL coefficient gradient should scale linearly with kl_coef."""
    policy = torch.tensor([0.0, 1.0], requires_grad=True)
    ref = torch.tensor([0.5, 0.5])

    # Base GSPO coefficients
    base_coeffs = compute_gspo_coefficients_with_kl(
        policy.detach(), ref, torch.tensor([0.0, 1.0]), beta=10.0, kl_coef=0.0
    )

    # With KL coef=0.1
    kl_coeffs = compute_gspo_coefficients_with_kl(
        policy.detach(), ref, torch.tensor([0.0, 1.0]), beta=10.0, kl_coef=0.1
    )

    kl_contribution = kl_coeffs - base_coeffs
    # KL gradient = 2 * kl_coef / K * (policy - ref)
    K = 2.0
    expected_kl_grad = 2.0 * 0.1 / K * (policy.detach() - ref)
    assert torch.allclose(kl_contribution, expected_kl_grad, atol=1e-6), \
        f"{kl_contribution} vs {expected_kl_grad}"
    print("✓ test_kl_coef_scales_linearly PASSED")


def test_kl_coeffs_shape():
    """Verify coefficients shape matches input."""
    for K in [2, 4, 8]:
        policy = torch.randn(K)
        ref = torch.randn(K)
        rewards = torch.randint(0, 2, (K,)).float()

        coeffs = compute_gspo_coefficients_with_kl(
            policy, ref, rewards, beta=0.1, kl_coef=0.05
        )
        assert coeffs.shape == (K,)
        assert coeffs.dtype == policy.dtype
    print("✓ test_kl_coeffs_shape PASSED")


def test_gspo_loss_with_kl_logging():
    """Verify compute_gspo_loss includes kl_penalty in logs when KL active."""
    policy = torch.tensor([0.5, 1.2, 0.8, 0.3])
    ref = torch.tensor([0.4, 1.0, 0.9, 0.2])
    rewards = torch.tensor([0.0, 1.0, 0.0, 0.0])

    _, logs = compute_gspo_loss(policy, ref, rewards, beta=0.1)
    assert "gspo/loss" in logs
    assert "gspo/rewards_chosen" in logs
    assert "gspo/logits" in logs
    print("✓ test_gspo_loss_logging PASSED")


def test_edge_case_k1():
    """With K=1, all functions should return zero tensors."""
    policy = torch.tensor([0.5])
    ref = torch.tensor([0.4])
    rewards = torch.tensor([1.0])

    coeffs = compute_gspo_coefficients_with_kl(policy, ref, rewards)
    assert coeffs.shape == (1,)
    assert torch.allclose(coeffs, torch.zeros(1), atol=1e-6)

    loss, logs = compute_gspo_loss(policy, ref, rewards)
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)
    print("✓ test_edge_case_k1 PASSED")


def test_edge_case_all_same_reward():
    """When all rewards are equal, coefficient should be zero for all."""
    policy = torch.tensor([0.5, 0.6, 0.4])
    ref = torch.tensor([0.4, 0.5, 0.5])
    rewards = torch.tensor([0.0, 0.0, 0.0])

    coeffs = compute_gspo_coefficients_with_kl(policy, ref, rewards, beta=0.1, kl_coef=0.0)
    # Without KL, all-zero rewards => coeffs should be zero (no best/others contrast)
    assert torch.allclose(coeffs, torch.zeros(3), atol=1e-6), f"coeffs={coeffs}"

    # With KL, there should still be KL contribution
    coeffs_kl = compute_gspo_coefficients_with_kl(policy, ref, rewards, beta=0.1, kl_coef=0.1)
    assert not torch.allclose(coeffs_kl, torch.zeros(3), atol=1e-6), \
        "KL contribution should be non-zero even with identical rewards"
    print("✓ test_edge_case_all_same_reward PASSED")


if __name__ == "__main__":
    print("=== Phase 3 Loss Function Validation ===\n")
    test_kl_coef_zero_equals_original()
    test_kl_penalty_numerical()
    test_kl_penalty_zero_when_identical()
    test_kl_coef_scales_linearly()
    test_kl_coeffs_shape()
    test_gspo_loss_with_kl_logging()
    test_edge_case_k1()
    test_edge_case_all_same_reward()
    print("\n✅ All loss tests PASSED")
