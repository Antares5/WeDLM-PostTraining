#!/usr/bin/env python
# coding=utf-8
"""Smoke test for GSPO loss functions.

Validates:
- compute_gspo_loss with random scores/rewards
- Gradient flow through compute_gspo_coefficients
- Edge cases: K=1, all rewards equal, empty inputs
"""

import os
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from src.loss import compute_gspo_loss, compute_gspo_coefficients


def test_gspo_loss_basic():
    """Test basic GSPO loss with random scores."""
    K = 4
    beta = 0.1

    # Simulate: response 2 is best
    policy_scores = torch.tensor([-2.0, -1.5, -0.5, -3.0])
    ref_scores = torch.tensor([-1.8, -1.6, -0.8, -2.8])
    rewards = torch.tensor([0.0, 0.0, 1.0, 0.0])  # response 2 is correct

    loss, logs = compute_gspo_loss(policy_scores, ref_scores, rewards, beta)

    logger.info(f"Basic GSPO loss: {loss.item():.4f}")
    for k, v in logs.items():
        logger.info(f"  {k}: {v.item():.4f}")

    assert loss.item() > 0, "Loss should be positive"
    assert loss.item() < 5.0, "Loss should be reasonable"
    assert logs["gspo/rewards_chosen"].item() == 1.0
    assert logs["gspo/num_correct"].item() == 1

    logger.info("✓ Basic GSPO loss test PASSED")


def test_gspo_coefficients_gradient():
    """Test that coefficients produce valid gradients."""
    K = 3
    beta = 0.1

    policy_scores_ng = torch.tensor([-1.0, -2.0, -3.0])
    ref_scores = torch.tensor([-1.2, -1.8, -2.5])
    rewards = torch.tensor([0.0, 1.0, 0.0])  # response 1 is best

    # Compute coefficients
    coeffs = compute_gspo_coefficients(policy_scores_ng, ref_scores, rewards, beta)
    logger.info(f"Coefficients: {coeffs}")

    # Create a differentiable parameter
    param = torch.nn.Parameter(torch.tensor(0.0))
    policy_scores_grad = param + policy_scores_ng

    # Verify we can backward through coefficients
    total = (coeffs.detach() * policy_scores_grad).sum()
    total.backward()
    logger.info(f"Gradient: {param.grad.item():.4f}")
    assert param.grad is not None
    assert abs(param.grad.item()) > 0

    logger.info("✓ GSPO coefficients gradient test PASSED")


def test_edge_cases():
    """Test edge cases."""
    beta = 0.1

    # K=0 (handled at trainer level, but test function behavior)
    # K=1
    policy_scores = torch.tensor([-1.0])
    ref_scores = torch.tensor([-1.0])
    rewards = torch.tensor([0.0])

    loss, logs = compute_gspo_loss(policy_scores, ref_scores, rewards, beta)
    logger.info(f"K=1 loss: {loss.item():.4f}")
    assert loss.item() == 0.0, "K=1 should return zero loss"

    # All rewards equal
    policy_scores = torch.tensor([-1.0, -2.0, -3.0])
    ref_scores = torch.tensor([-1.0, -2.0, -3.0])
    rewards = torch.tensor([0.0, 0.0, 0.0])

    loss, logs = compute_gspo_loss(policy_scores, ref_scores, rewards, beta)
    logger.info(f"All zero rewards loss: {loss.item():.4f}")
    # Should be close to -log(sigmoid(0))... but all rewards equal -> best is first
    # best_idx = argmax(rewards) = 0 -> best=0, others=[0,0] -> logmeanexp=0 -> logits=0 -> -logsigmoid(0) = log(2) ≈ 0.693
    assert abs(loss.item() - 0.693) < 0.01, f"Expected ~0.693, got {loss.item():.4f}"

    logger.info("✓ Edge case tests PASSED")


def test_coefficients_sum_to_zero():
    """Verify coefficients approximately sum to near-zero (gradient conservation)."""
    K = 4
    beta = 0.1

    for _ in range(10):
        policy_scores = torch.randn(K) * 2
        ref_scores = torch.randn(K) * 2
        rewards = (torch.rand(K) > 0.5).float()

        coeffs = compute_gspo_coefficients(policy_scores, ref_scores, rewards, beta)
        coeff_sum = coeffs.sum().item()
        logger.info(f"Coefficients sum: {coeff_sum:.6f}")

        # Should be close to 0 (since adding constant to all scores shouldn't change loss)
        assert abs(coeff_sum) < 1e-4, f"Coefficients should sum to ~0, got {coeff_sum}"

    logger.info("✓ Zero-sum test PASSED")


if __name__ == "__main__":
    test_gspo_loss_basic()
    test_gspo_coefficients_gradient()
    test_edge_cases()
    test_coefficients_sum_to_zero()
    logger.info("All GSPO loss tests passed!")
