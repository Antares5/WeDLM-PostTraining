# coding=utf-8
"""Smoke test for gspo/src/loss.py — Step 3 validation.

Validates GRPO clipped loss, group advantage normalisation, KL estimators,
and reward functions.

Usage:
    python scripts/smoke_test_loss.py
"""

from __future__ import annotations

import os
import sys
import logging

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
_PARENT_DIR = os.path.dirname(_PROJECT_DIR)
sys.path.insert(0, os.path.join(_PARENT_DIR, "dpo"))
sys.path.insert(0, os.path.join(_PROJECT_DIR, "src"))

import torch

from loss import (
    compute_group_advantage,
    compute_k3_kl,
    compute_reverse_kl,
    compute_grpo_loss,
    math_reward,
    compute_rewards,
    extract_gsm8k_answer,
    extract_boxed_answer,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_group_advantage():
    """Verify group-level normalisation produces zero-mean unit-var groups."""
    logger.info("=== test_group_advantage ===")

    # B=2 groups, G=4 completions each.
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0,    # group 0: mean=2.5, std≈1.12
                            10.0, 10.0, 10.0, 10.0])  # group 1: all same → std=0

    adv = compute_group_advantage(rewards, group_size=4)

    # Group 0: should be zero-mean.
    g0 = adv[:4]
    assert abs(g0.mean().item()) < 1e-6, f"g0 mean {g0.mean()} != 0"
    logger.info("  ✓ group 0: zero mean (%e)", g0.mean().item())

    # Group 1: all equal → advantages = 0.
    g1 = adv[4:]
    assert (g1.abs() < 1e-6).all(), f"g1 advantages not zero: {g1}"
    logger.info("  ✓ group 1: zero advantages (uniform rewards)")

    # Shape.
    assert adv.shape == (8,), f"shape {adv.shape}"
    logger.info("  ✓ shape correct")

    logger.info("  PASSED\n")


def test_k3_kl():
    """Verify k3 KL estimator properties."""
    logger.info("=== test_k3_kl ===")

    s_policy = torch.tensor([0.0, -1.0, -2.0, -3.0], requires_grad=True)
    s_ref = torch.tensor([0.0, -1.0, -2.0, -3.0])

    # Same scores → KL ≈ 0.
    kl_eq = compute_k3_kl(s_policy, s_ref)
    assert abs(kl_eq.item()) < 1e-5, f"KL(same) = {kl_eq.item()} != 0"
    logger.info("  ✓ KL(same scores) ≈ 0: %e", kl_eq.item())

    # Higher policy scores → positive KL.
    s_policy2 = torch.tensor([-0.5, -0.5, -0.5, -0.5], requires_grad=True)
    s_ref2 = torch.tensor([-0.5, 0.0, -0.5, -0.5])  # one is slightly higher
    kl_pos = compute_k3_kl(s_policy2, s_ref2)
    assert kl_pos.item() >= -1e-5, f"k3 KL should be non-negative in expectation"
    logger.info("  ✓ KL is non-negative: %e", kl_pos.item())

    # Gradient flows through s_policy.
    kl_grad = compute_k3_kl(s_policy, s_ref)
    kl_grad.backward()
    assert s_policy.grad is not None, "no grad on s_policy"
    assert (s_policy.grad.abs().sum() > 0), "gradient is all zero"
    logger.info("  ✓ gradient flows through k3 KL")

    logger.info("  PASSED\n")


def test_grpo_loss_basic():
    """Verify GRPO loss produces sensible values and gradients."""
    logger.info("=== test_grpo_loss_basic ===")

    B, G = 2, 4
    BxG = B * G

    # Scenario: scores slightly higher for better rewards.
    s_policy = torch.randn(BxG, requires_grad=True) * 0.1
    s_old = s_policy.detach().clone() + torch.randn(BxG) * 0.05
    rewards = torch.cat([
        torch.tensor([0.0, 0.0, 1.0, 1.0]),    # group 0
        torch.tensor([0.0, 1.0, 1.0, 0.0]),    # group 1
    ])

    loss, logs = compute_grpo_loss(
        s_policy=s_policy,
        s_old=s_old,
        s_ref=None,
        rewards=rewards,
        group_size=G,
        clip_epsilon=0.2,
        kl_beta=0.0,
    )

    assert loss.ndim == 0, f"loss should be scalar, got shape {tuple(loss.shape)}"
    logger.info("  ✓ loss is scalar: %.4f", loss.item())

    # Gradient flows.
    loss.backward()
    assert s_policy.grad is not None
    assert s_policy.grad.abs().sum() > 0
    logger.info("  ✓ gradient flows back")

    # All log keys present.
    expected_keys = [
        "grpo/loss", "grpo/total_loss", "grpo/approx_kl",
        "grpo/clip_frac", "grpo/mean_rho", "grpo/mean_advantage",
        "grpo/reward_mean", "grpo/reward_std",
    ]
    for key in expected_keys:
        assert key in logs, f"missing log key: {key}"
    logger.info("  ✓ all expected log keys present")

    # clip_frac should be in [0, 1].
    cf = logs["grpo/clip_frac"].item()
    assert 0.0 <= cf <= 1.0, f"clip_frac {cf} out of range"
    logger.info("  ✓ clip_frac in [0,1]: %.3f", cf)

    # With small random differences, most ρ should be near 1 → clip_frac small.
    logger.info("  logs: %s", {k: f"{v.item():.4f}" for k, v in logs.items()})
    logger.info("  PASSED\n")


def test_grpo_loss_with_kl():
    """Verify GRPO loss with KL penalty enabled."""
    logger.info("=== test_grpo_loss_with_kl ===")

    BxG = 8
    s_policy = torch.zeros(BxG, requires_grad=True)
    s_old = torch.zeros(BxG)
    s_ref = torch.randn(BxG) * 0.1
    rewards = torch.ones(BxG)  # uniform → advantages = 0

    # With advantages=0, pure loss = KL term.
    loss, logs = compute_grpo_loss(
        s_policy=s_policy, s_old=s_old, s_ref=s_ref,
        rewards=rewards, group_size=4,
        clip_epsilon=0.2, kl_beta=1.0, kl_estimator="k3",
    )

    assert loss.ndim == 0
    logger.info("  ✓ loss with KL: %.4f", loss.item())
    assert logs["grpo/kl_loss"].item() > 0 or abs(logs["grpo/kl_loss"].item()) < 1e-3
    logger.info("  ✓ kl_loss in logs: %.4f", logs["grpo/kl_loss"].item())

    logger.info("  PASSED\n")


def test_grpo_loss_clipping():
    """Verify that clipping activates for extreme importance ratios."""
    logger.info("=== test_grpo_loss_clipping ===")

    BxG = 4
    # Force large log-ratio → ρ far from 1 → high clip_frac.
    s_policy = torch.tensor([5.0, -5.0, 0.0, 0.0], requires_grad=True)
    s_old = torch.tensor([0.0, 0.0, 0.0, 0.0])
    rewards = torch.tensor([1.0, 0.0, 0.5, 0.5])

    loss, logs = compute_grpo_loss(
        s_policy=s_policy, s_old=s_old, s_ref=None,
        rewards=rewards, group_size=2,
        clip_epsilon=0.2,
    )

    # ρ for first element = exp(5) ≈ 148 → should be clipped.
    cf = logs["grpo/clip_frac"].item()
    assert cf > 0.0, f"clip_frac should be > 0 for extreme ρ, got {cf}"
    logger.info("  ✓ clipping active: clip_frac = %.3f", cf)
    logger.info("    ρ values: %s", (s_policy - s_old).exp().detach().tolist())
    logger.info("  PASSED\n")


def test_math_reward():
    """Verify math reward extraction logic."""
    logger.info("=== test_math_reward ===")

    # GSM8K format.
    assert math_reward("The answer is #### 42", "42") == 1.0
    assert math_reward("#### 42\n", "42") == 1.0
    assert math_reward("#### 42", "43") == 0.0
    logger.info("  ✓ GSM8K #### format")

    # Boxed format.
    assert math_reward(r"The answer is \boxed{42}", "42") == 1.0
    assert math_reward(r"\boxed{3.14}", "3.14") == 1.0
    logger.info("  ✓ \\boxed format")

    # Last number fallback.
    assert math_reward("The result is 100.", "100") == 1.0
    logger.info("  ✓ last-number fallback")

    # No number.
    assert math_reward("No answer here.", "42") == 0.0
    logger.info("  ✓ no answer → 0.0")

    # Float tolerance.
    assert math_reward("#### 3.1416", "3.14159") == 1.0
    assert math_reward("#### 3.0", "3.14") == 0.0
    logger.info("  ✓ float tolerance")

    logger.info("  PASSED\n")


def test_compute_rewards_batch():
    """Verify batch reward computation."""
    logger.info("=== test_compute_rewards_batch ===")

    completions = ["#### 10", "wrong", "#### 20"]
    truths = ["10", "10", "30"]
    rewards = compute_rewards(completions, truths)

    assert rewards.shape == (3,)
    assert rewards[0].item() == 1.0
    assert rewards[1].item() == 0.0
    assert rewards[2].item() == 0.0
    logger.info("  ✓ batch rewards: %s", rewards.tolist())

    logger.info("  PASSED\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    logger.info("=" * 60)
    logger.info("GSPO Loss Smoke Test")
    logger.info("=" * 60)
    logger.info("Torch version: %s", torch.__version__)
    logger.info("")

    test_group_advantage()
    test_k3_kl()
    test_grpo_loss_basic()
    test_grpo_loss_with_kl()
    test_grpo_loss_clipping()
    test_math_reward()
    test_compute_rewards_batch()

    logger.info("=" * 60)
    logger.info("All tests passed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
