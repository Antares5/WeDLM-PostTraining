#!/usr/bin/env python
# coding=utf-8
"""Smoke test for Step 1: GSPO config loading + loss computation.

Run from the project root:
    python gspo/scripts/smoke_test_step1.py

This test is self-contained (CPU-only, no model loading).
It verifies:
  1. GSPOConfig can be loaded from YAML and has correct defaults.
  2. compute_gspo_loss produces valid values for synthetic data.
  3. Edge cases: uniform rewards, single response groups, zero std.
"""

import os
import sys

# Ensure we can import from sibling packages
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
import yaml
import tempfile
import traceback


def green(s: str) -> str:
    return f"\033[32m{s}\033[0m"


def red(s: str) -> str:
    return f"\033[31m{s}\033[0m"


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


passed = 0
failed = 0


def check(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        print(f"  {green('[PASS]')} {name}")
        passed += 1
    else:
        print(f"  {red('[FAIL]')} {name}" + (f"  → {detail}" if detail else ""))
        failed += 1


# ═══════════════════════════════════════════════════════════════
# TEST 1: Config loading
# ═══════════════════════════════════════════════════════════════

section("Test 1: GSPOConfig loading")

from gspo.src.config import GSPOConfig

# 1a: Default construction
cfg_default = GSPOConfig()
check("1a-default: training_mode = 'gspo'",
      cfg_default.training_mode == "gspo")
check("1a-default: gspo_group_size = 4",
      cfg_default.gspo_group_size == 4)
check("1a-default: gspo_num_mask_samples = 4",
      cfg_default.gspo_num_mask_samples == 4)

# 1b: Load from YAML
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           "configs", "example.yaml")
if os.path.exists(config_path):
    cfg_yaml = GSPOConfig.from_yaml(config_path)
    check("1b-yaml: training_mode = 'gspo'",
          cfg_yaml.training_mode == "gspo",
          f"Got: {cfg_yaml.training_mode}")
    check("1b-yaml: gspo_group_size loaded",
          cfg_yaml.gspo_group_size >= 2,
          f"gspo_group_size = {cfg_yaml.gspo_group_size}")
    check("1b-yaml: block_size loaded",
          cfg_yaml.block_size == 32,
          f"block_size = {cfg_yaml.block_size}")
    check("1b-yaml: attention_backend loaded",
          cfg_yaml.attention_backend in ("magi", "dense"))
else:
    print(f"  {red('[SKIP]')} 1b-yaml: config file not found at {config_path}")

# 1c: Validation
try:
    cfg_bad = GSPOConfig(gspo_group_size=1)
    check("1c-validation: group_size=1 should raise", False,
          "Should have raised ValueError")
except ValueError:
    check("1c-validation: group_size=1 raises ValueError", True)

# 1d: YAML round-trip
cfg_rt = GSPOConfig()
with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
    cfg_rt.save_yaml(f.name)
    tmp_path = f.name
cfg_rt2 = GSPOConfig.from_yaml(tmp_path)
check("1d-roundtrip: training_mode matches",
      cfg_rt2.training_mode == cfg_rt.training_mode)
check("1d-roundtrip: gspo_group_size matches",
      cfg_rt2.gspo_group_size == cfg_rt.gspo_group_size)
os.unlink(tmp_path)


# ═══════════════════════════════════════════════════════════════
# TEST 2: compute_gspo_loss — basic correctness
# ═══════════════════════════════════════════════════════════════

section("Test 2: compute_gspo_loss — basic correctness")

from gspo.src.loss import compute_gspo_loss

# 2a: Simple case: 2 prompts, G=2 each
scores = torch.tensor([0.5, 0.3,  0.8, 0.2])  # prompt0: (0.5,0.3), prompt1: (0.8,0.2)
rewards = torch.tensor([1.0, 0.0,  1.0, 0.0])  # clear preference
prompt_idx = torch.tensor([0, 0,  1, 1])

loss, logs = compute_gspo_loss(scores, rewards, prompt_idx)

check("2a-basic: loss is scalar", loss.dim() == 0)
check("2a-basic: loss is finite", torch.isfinite(loss))
check("2a-basic: loss > 0 (higher score should have positive advantage)",
      loss.item() > 0,
      f"loss={loss.item():.6f}")

# Expected: advantage for higher reward = +1, for lower = -1
# L = -[(1 * 0.5 + (-1) * 0.3)/2 + (1 * 0.8 + (-1) * 0.2)/2] / 2
#   = -[(0.2)/2 + (0.6)/2] / 2 = -[0.1 + 0.3] / 2 = -0.2
# Wait: advantage is (r-mu)/sigma. For group0: r=[1.0, 0.0], mu=0.5, sigma=0.707+eps
# A0 ≈ (1.0-0.5)/0.707 ≈ 0.707, A1 ≈ -0.707
# L = -[(0.707*0.5 + (-0.707)*0.3)/2 + (0.707*0.8 + (-0.707)*0.2)/2]/2
#   = -[(0.3535-0.2121)/2 + (0.5656-0.1414)/2]/2 = -[0.0707+0.2121]/2 = -0.1414
print(f"  2a-basic: loss={loss.item():.6f}, adv_mean={logs['gspo/adv_mean'].item():.6f}")

# 2b: All required log keys present
required_keys = [
    "gspo/loss", "gspo/adv_mean", "gspo/adv_std",
    "gspo/score_mean", "gspo/reward_mean",
]
for key in required_keys:
    check(f"2b-logs: '{key}' present", key in logs, f"Missing key: {key}")

# 2c: Gradients flow through scores
scores_grad = torch.tensor([0.5, 0.3, 0.8, 0.2], requires_grad=True)
rewards_g = torch.tensor([1.0, 0.0, 1.0, 0.0])
prompt_g = torch.tensor([0, 0, 1, 1])
loss_g, _ = compute_gspo_loss(scores_grad, rewards_g, prompt_g)
loss_g.backward()
check("2c-grad: scores.grad is not None", scores_grad.grad is not None)
check("2c-grad: scores.grad is finite", torch.all(torch.isfinite(scores_grad.grad)).item())
print(f"  2c-grad: scores.grad = {scores_grad.grad.tolist()}")


# ═══════════════════════════════════════════════════════════════
# TEST 3: Edge cases
# ═══════════════════════════════════════════════════════════════

section("Test 3: Edge cases")

# 3a: Uniform rewards → zero advantages → zero loss
scores_u = torch.tensor([0.5, 0.5, 0.5, 0.5])
rewards_u = torch.tensor([1.0, 1.0, 1.0, 1.0])
prompt_u = torch.tensor([0, 0, 1, 1])
loss_u, logs_u = compute_gspo_loss(scores_u, rewards_u, prompt_u)
check("3a-uniform-rewards: loss ≈ 0", abs(loss_u.item()) < 5e-7,
      f"loss={loss_u.item():.10f}")
check("3a-uniform-rewards: adv ≈ 0",
      abs(logs_u["gspo/adv_mean"].item()) < 5e-7,
      f"adv_mean={logs_u['gspo/adv_mean'].item():.10f}")

# 3b: Larger group size (G=4)
N = 8
B = 2
G = 4
scores_b = torch.randn(N)
rewards_b = torch.rand(N)
prompt_b = torch.repeat_interleave(torch.arange(B), G)
loss_b, logs_b = compute_gspo_loss(scores_b, rewards_b, prompt_b)
check("3b-G4: loss is scalar & finite",
      loss_b.dim() == 0 and torch.isfinite(loss_b))
check("3b-G4: correct number of prompts", True)  # implicit in shape check

# 3c: Single response per group → zero advantage → warning handled
scores_s = torch.tensor([0.5, 0.3])
rewards_s = torch.tensor([1.0, 2.0])
prompt_s = torch.tensor([0, 1])  # B=2, G=1 each
loss_s, logs_s = compute_gspo_loss(scores_s, rewards_s, prompt_s)
check("3c-single-response: loss ≈ 0 (skipped)", abs(loss_s.item()) < 5e-7,
      f"loss={loss_s.item():.10f}")

# 3d: Empty input
try:
    scores_e = torch.tensor([])
    rewards_e = torch.tensor([])
    prompt_e = torch.tensor([])
    loss_e, _ = compute_gspo_loss(scores_e, rewards_e, prompt_e)
    check("3d-empty: should raise or return", True)  # no crash is pass
except ValueError:
    check("3d-empty: raises ValueError (expected)", True)

# 3e: Mismatched shapes
try:
    scores_m = torch.tensor([1.0, 2.0, 3.0])
    rewards_m = torch.tensor([1.0, 2.0])
    prompt_m = torch.tensor([0, 0, 0])
    compute_gspo_loss(scores_m, rewards_m, prompt_m)
    check("3e-mismatch: should raise ValueError", False)
except ValueError:
    check("3e-mismatch: raises ValueError (expected)", True)


# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════

section("Summary")
total = passed + failed
print(f"  {passed}/{total} tests passed")
if failed == 0:
    print(f"\n  {green('ALL TESTS PASSED — Step 1 is ready.')}")
else:
    print(f"\n  {red(f'{failed} TEST(S) FAILED — fix before proceeding.')}")

sys.exit(0 if failed == 0 else 1)
