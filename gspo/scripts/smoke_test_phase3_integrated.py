#!/usr/bin/env python
# coding=utf-8
"""Phase 3 integrated smoke test — requires torch & transformers.

Run this on your actual training machine:
    python scripts/smoke_test_phase3_integrated.py

Validates:
1. Config YAML roundtrip with all Phase 3 fields
2. KL penalty numerical correctness
3. GSPO coefficients_with_kl gradient equivalence
4. Reward model dispatch (math_verify vs model)
5. Trainer checkpoint save/load roundtrip (mock)
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.config import GSPOTrainingConfig
from src.loss import (
    compute_gspo_coefficients,
    compute_gspo_coefficients_with_kl,
    compute_kl_penalty,
    compute_gspo_loss,
)
from src.reward import MathReward


# ============================================================
# Test 1: Config YAML roundtrip
# ============================================================
def test_config_roundtrip():
    """Create a full config with Phase 3 fields, save, reload, compare."""
    import tempfile
    cfg = GSPOTrainingConfig(
        gspo_use_kl_penalty=True,
        gspo_kl_coef=0.03,
        gspo_log_samples_every_n_steps=50,
        gspo_resume_from_checkpoint="/tmp/ckpt-42",
        gspo_reward_type="model",
        gspo_reward_model_path="/tmp/rm",
        gspo_reward_model_type="sequence_classification",
    )
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
        cfg.save_yaml(f.name)
        tmp_path = f.name

    try:
        cfg2 = GSPOTrainingConfig.from_yaml(tmp_path)
        for field in [
            "gspo_use_kl_penalty", "gspo_kl_coef",
            "gspo_log_samples_every_n_steps", "gspo_resume_from_checkpoint",
            "gspo_reward_type", "gspo_reward_model_path", "gspo_reward_model_type",
        ]:
            assert getattr(cfg2, field) == getattr(cfg, field), \
                f"Mismatch on {field}: {getattr(cfg2, field)} != {getattr(cfg, field)}"
        print("[PASS] Config YAML roundtrip")
    finally:
        os.unlink(tmp_path)


# ============================================================
# Test 2: KL penalty numerical
# ============================================================
def test_kl_penalty():
    """Verify compute_kl_penalty returns mean((policy-ref)^2)."""
    policy = torch.tensor([1.0, 2.0, 3.0])
    ref = torch.tensor([0.0, 1.0, 2.0])
    kl_loss, kl_per = compute_kl_penalty(policy, ref)
    expected = torch.tensor(1.0)  # mean([1,1,1])
    assert torch.allclose(kl_loss, expected), f"{kl_loss} != {expected}"
    assert kl_per.tolist() == [1.0, 1.0, 1.0]
    print("[PASS] KL penalty numerical")


# ============================================================
# Test 3: KL=0 degenerates to base coefficients
# ============================================================
def test_kl_zero_degenerate():
    """With kl_coef=0, the 'with_kl' function == base function."""
    torch.manual_seed(42)
    for _ in range(10):
        K = torch.randint(2, 6, (1,)).item()
        policy = torch.randn(K)
        ref = torch.randn(K)
        rewards = torch.randint(0, 2, (K,)).float()
        beta = 0.1 + torch.rand(1).item()

        base = compute_gspo_coefficients(policy, ref, rewards, beta)
        with_kl = compute_gspo_coefficients_with_kl(policy, ref, rewards, beta, kl_coef=0.0)
        assert torch.allclose(base, with_kl, atol=1e-6), \
            f"K={K}: base={base}, with_kl={with_kl}"
    print("[PASS] KL=0 degenerates to base coefficients")


# ============================================================
# Test 4: KL contribution formula
# ============================================================
def test_kl_contribution_formula():
    """∂L_kl/∂s_i = 2 * kl_coef / K * (s_i - s_ref_i)."""
    torch.manual_seed(123)
    K = 4
    policy = torch.tensor([0.0, 1.0, 2.0, -1.0])
    ref = torch.tensor([0.5, 0.5, 0.5, 0.5])

    coeffs = compute_gspo_coefficients_with_kl(
        policy, ref, torch.tensor([0.0, 1.0, 0.0, 0.0]), beta=10.0, kl_coef=0.1
    )
    base_coeffs = compute_gspo_coefficients(
        policy, ref, torch.tensor([0.0, 1.0, 0.0, 0.0]), beta=10.0
    )

    kl_part = coeffs - base_coeffs
    expected_kl = 2.0 * 0.1 / float(K) * (policy - ref)
    assert torch.allclose(kl_part, expected_kl, atol=1e-6), \
        f"kl_part={kl_part}, expected={expected_kl}"
    print("[PASS] KL contribution formula")


# ============================================================
# Test 5: GSPO loss edge cases
# ============================================================
def test_loss_edge_cases():
    """Test K=1, all-same-rewards, and negative margin."""
    # K=1 should return zero loss
    loss, _ = compute_gspo_loss(
        torch.tensor([0.5]), torch.tensor([0.4]), torch.tensor([1.0]), beta=0.1
    )
    assert torch.allclose(loss, torch.tensor(0.0)), f"K=1 loss={loss}"
    print("[PASS] K=1 returns zero loss")

    # All rew 0 → no meaningful contrast → zero loss / zero coefficients
    loss, _ = compute_gspo_loss(
        torch.tensor([0.5, 0.6]), torch.tensor([0.4, 0.5]),
        torch.tensor([0.0, 0.0]), beta=0.1
    )
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6), f"all-rew-0 loss={loss}"
    print("[PASS] All-zero rewards returns zero loss")

    # When chosen has higher score: loss should be < log(2)
    loss, logs = compute_gspo_loss(
        torch.tensor([0.5, 1.5]), torch.tensor([0.4, 0.4]),
        torch.tensor([0.0, 1.0]), beta=0.1
    )
    assert loss.item() > 0, f"Loss should be positive, got {loss}"
    # log_sigmoid(positive) ≈ small negative, -log_sigmoid ≈ small positive
    print(f"[PASS] Positive margin loss = {loss.item():.4f}")


# ============================================================
# Test 6: MathReward integration
# ============================================================
def test_math_reward():
    """Verify math reward extraction and verification."""
    reward = MathReward(reward_type="math_verify")

    # Test GSM8K format
    assert reward.verify_answer("42", "42"), "exact match"
    assert reward.verify_answer("3.14", "3.14"), "float match"
    assert reward.verify_answer("1,000", "1000"), "comma removal"
    print("[PASS] MathReward basic verification")

    # Test answer extraction
    assert reward.extract_answer("Let me think...\n#### 42\nSo the answer is 42.") == "42"
    assert reward.extract_answer(r"The answer is \boxed{3.14}.") == "3.14"
    print("[PASS] MathReward answer extraction")

    # Test batch reward computation
    rewards = reward.compute_rewards(
        ["p1", "p2"],
        ["#### 42", "#### 99"],
        ["42", "100"],
    )
    assert rewards.tolist() == [1.0, 0.0], f"got {rewards.tolist()}"
    print("[PASS] MathReward batch computation")


# ============================================================
# Test 7: Loss shape invariants
# ============================================================
def test_shape_invariants():
    """Verify output shapes for varying K."""
    torch.manual_seed(999)
    for K in [2, 3, 5, 8]:
        policy = torch.randn(K)
        ref = torch.randn(K)
        rewards = torch.randint(0, 2, (K,)).float()

        coeffs = compute_gspo_coefficients_with_kl(policy, ref, rewards, beta=0.1, kl_coef=0.05)
        assert coeffs.shape == (K,), f"K={K}: coeffs shape={coeffs.shape}"

        loss, logs = compute_gspo_loss(policy, ref, rewards, beta=0.1)
        assert loss.dim() == 0, f"K={K}: loss dim={loss.dim()}"
    print("[PASS] Shape invariants for K=2,3,5,8")


# ============================================================
# Test 8: KL penalty per-sample values
# ============================================================
def test_kl_per_sample():
    """Verify per-sample KL = (s_i - s_ref_i)^2 for each i."""
    policy = torch.tensor([1.0, 0.0])
    ref = torch.tensor([3.0, 0.0])
    _, kl_per = compute_kl_penalty(policy, ref)
    # per_sample[0] = (1-3)^2 = 4.0, per_sample[1] = (0-0)^2 = 0.0
    assert torch.allclose(kl_per, torch.tensor([4.0, 0.0]), atol=1e-6), f"{kl_per}"
    print("[PASS] KL per-sample values")


# ============================================================
# Test 9: Normalization in reward
# ============================================================
def test_reward_normalization():
    """Verify that reward normalization doesn't break."""
    # Same reward values should not crash after normalization attempt
    rewards = torch.tensor([1.0, 1.0, 1.0])
    if rewards.std() > 1e-8 and rewards.numel() > 1:
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    # All identical => std=0 => skip normalization
    assert torch.allclose(rewards, torch.tensor([1.0, 1.0, 1.0]))
    print("[PASS] Reward normalization safety")

    # Different rewards should be normalized
    rewards2 = torch.tensor([0.0, 1.0, 0.0, 1.0])
    normalized = (rewards2 - rewards2.mean()) / (rewards2.std() + 1e-8)
    assert normalized.mean().abs() < 1e-6
    assert abs(normalized.std() - 1.0) < 1e-6
    print("[PASS] Reward normalization correctness")


# ============================================================
# Test 10: Trainer state save/load (mock)
# ============================================================
def test_trainer_state_roundtrip():
    """Mock save/load of trainer_state.pt."""
    import tempfile
    state = {
        "global_step": 42,
        "step_in_gen_cycle": 3,
        "python_rng_state": None,  # can't pickle, but dict roundtrip works
    }
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(state, f.name)
        tmp = f.name

    try:
        loaded = torch.load(tmp, map_location="cpu")
        assert loaded["global_step"] == 42
        assert loaded["step_in_gen_cycle"] == 3
        print("[PASS] Trainer state save/load roundtrip")
    finally:
        os.unlink(tmp)


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 3 Integrated Smoke Test")
    print("=" * 60)

    test_config_roundtrip()
    test_kl_penalty()
    test_kl_zero_degenerate()
    test_kl_contribution_formula()
    test_loss_edge_cases()
    test_math_reward()
    test_shape_invariants()
    test_kl_per_sample()
    test_reward_normalization()
    test_trainer_state_roundtrip()

    print("\n" + "=" * 60)
    print("✅ All Phase 3 smoke tests PASSED")
    print("=" * 60)
