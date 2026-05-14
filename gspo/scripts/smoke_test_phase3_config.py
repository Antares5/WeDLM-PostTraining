#!/usr/bin/env python
# coding=utf-8
"""Phase 3 config validation — no GPU required."""

import os
import sys
import shutil

# Ensure gspo/src is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

# Import directly to avoid __init__.py pulling in torch
import importlib.util
def _import_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_config_path = os.path.join(os.path.dirname(__file__), "..", "src", "config.py")
config_mod = _import_from_file("config", _config_path)
GSPOTrainingConfig = config_mod.GSPOTrainingConfig


def test_default_config():
    """Verify all Phase 3 fields have correct defaults."""
    cfg = GSPOTrainingConfig()
    assert cfg.gspo_reward_model_path is None
    assert cfg.gspo_reward_model_type == "auto"
    assert cfg.gspo_use_kl_penalty is False
    assert cfg.gspo_kl_coef == 0.01
    assert cfg.gspo_resume_from_checkpoint is None
    assert cfg.gspo_log_samples_every_n_steps == 100
    print("✓ test_default_config PASSED")


def test_reward_type_model_requires_path():
    """Verify model reward type enforces path."""
    try:
        cfg = GSPOTrainingConfig(gspo_reward_type="model")
        print("✗ Expected ValueError for missing RM path")
    except ValueError as e:
        assert "gspo_reward_model_path is required" in str(e)
        print("✓ test_reward_type_model_requires_path PASSED")


def test_kl_penalty_config():
    """Verify KL penalty config fields."""
    cfg = GSPOTrainingConfig(gspo_use_kl_penalty=True, gspo_kl_coef=0.05)
    assert cfg.gspo_use_kl_penalty is True
    assert cfg.gspo_kl_coef == 0.05
    print("✓ test_kl_penalty_config PASSED")


def test_kl_coef_negative_rejected():
    """Verify negative KL coef raises error."""
    try:
        cfg = GSPOTrainingConfig(gspo_kl_coef=-1.0)
        print("✗ Expected ValueError for negative KL coef")
    except ValueError:
        print("✓ test_kl_coef_negative_rejected PASSED")


def test_log_samples_zero_disabled():
    """Verify log_samples=0 means disabled."""
    cfg = GSPOTrainingConfig(gspo_log_samples_every_n_steps=0)
    assert cfg.gspo_log_samples_every_n_steps == 0
    print("✓ test_log_samples_zero_disabled PASSED")

    try:
        cfg = GSPOTrainingConfig(gspo_log_samples_every_n_steps=-1)
        print("✗ Expected ValueError for negative log_samples")
    except ValueError:
        print("✓ test_log_samples_negative_rejected PASSED")


def test_resume_from_checkpoint():
    """Verify checkpoint resume field."""
    cfg = GSPOTrainingConfig(gspo_resume_from_checkpoint="/tmp/ckpt-100")
    assert cfg.gspo_resume_from_checkpoint == "/tmp/ckpt-100"
    print("✓ test_resume_from_checkpoint PASSED")


def test_save_load_yaml():
    """Verify YAML roundtrip includes new fields."""
    import tempfile
    cfg = GSPOTrainingConfig(
        gspo_use_kl_penalty=True,
        gspo_kl_coef=0.03,
        gspo_log_samples_every_n_steps=50,
        gspo_resume_from_checkpoint="/tmp/ckpt-42",
    )
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
        cfg.save_yaml(f.name)
        tmp_path = f.name

    try:
        cfg2 = GSPOTrainingConfig.from_yaml(tmp_path)
        assert cfg2.gspo_use_kl_penalty == cfg.gspo_use_kl_penalty
        assert cfg2.gspo_kl_coef == cfg.gspo_kl_coef
        assert cfg2.gspo_log_samples_every_n_steps == cfg.gspo_log_samples_every_n_steps
        assert cfg2.gspo_resume_from_checkpoint == cfg.gspo_resume_from_checkpoint
        print("✓ test_save_load_yaml PASSED")
    finally:
        os.unlink(tmp_path)


def test_reward_model_type_validation():
    """Verify reward model type validation."""
    for valid_type in ["auto", "sequence_classification", "causal_lm"]:
        cfg = GSPOTrainingConfig(gspo_reward_model_type=valid_type)
        assert cfg.gspo_reward_model_type == valid_type

    try:
        cfg = GSPOTrainingConfig(gspo_reward_model_type="unknown")
        print("✗ Expected ValueError for invalid RM type")
    except ValueError:
        print("✓ test_reward_model_type_validation PASSED")


if __name__ == "__main__":
    print("=== Phase 3 Config Validation ===")
    test_default_config()
    test_reward_type_model_requires_path()
    test_kl_penalty_config()
    test_kl_coef_negative_rejected()
    test_log_samples_zero_disabled()
    test_resume_from_checkpoint()
    test_save_load_yaml()
    test_reward_model_type_validation()
    print("\n✅ All config tests PASSED")
