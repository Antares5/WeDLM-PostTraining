# coding=utf-8
"""Smoke test for GSPO trainer pipeline — Step 4 validation.

Validates config, dataset, and (optionally) full trainer initialisation.

Usage:
    # Phase A: config + data (no model)
    python scripts/smoke_test_trainer.py

    # Phase B: trainer init (needs GPU + model)
    python scripts/smoke_test_trainer.py --model-path tencent/WeDLM-8B-Instruct
"""

from __future__ import annotations

import os
import sys
import argparse
import logging
import tempfile

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
_PARENT_DIR = os.path.dirname(_PROJECT_DIR)
sys.path.insert(0, os.path.join(_PARENT_DIR, "dpo"))       # for src.* (dpo/src)
sys.path.insert(0, os.path.join(_PROJECT_DIR, "src"))       # for gspo modules

import torch
from transformers import AutoTokenizer

from config import GSPOConfig
from data import GSPOPromptDataset, gspo_collate_fn

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Phase A: Config + Data (no model)
# ═══════════════════════════════════════════════════════════════════════════════

def test_config():
    """Verify GSPOConfig loads and validates."""
    logger.info("=== test_config ===")
    cfg = GSPOConfig()
    assert cfg.training_mode == "gspo"
    assert cfg.gspo_group_size == 4
    assert cfg.gspo_clip_epsilon == 0.2
    assert cfg.gspo_kl_estimator == "k3"
    logger.info("  ✓ defaults correct")

    # Validation.
    try:
        bad = GSPOConfig(gspo_group_size=1)
        assert False, "should raise"
    except ValueError:
        pass
    logger.info("  ✓ rejects gspo_group_size < 2")

    try:
        bad = GSPOConfig(gspo_kl_estimator="invalid")
        assert False, "should raise"
    except ValueError:
        pass
    logger.info("  ✓ rejects unknown kl_estimator")

    logger.info("  PASSED\n")


def test_dataset_loading(tmpdir=None):
    """Verify GSPOPromptDataset loads prompt + ground_truth pairs."""
    logger.info("=== test_dataset_loading ===")

    # Use a real tokenizer if available, else skip (Phase A only needs data logic).
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "tencent/WeDLM-8B-Instruct", trust_remote_code=True,
        )
    except Exception:
        # Fallback: use a minimal tokenizer from a common model.
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
        except Exception:
            logger.warning("  ⚠ no tokenizer available, skipping data loading test")
            logger.info("  SKIPPED\n")
            return

    data_path = os.path.join(_PROJECT_DIR, "data", "gspo_prompts.jsonl")
    if not os.path.exists(data_path):
        logger.warning("  ⚠ data/gspo_prompts.jsonl not found, skipping")
        logger.info("  SKIPPED\n")
        return

    ds = GSPOPromptDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_seq_length=2048,
        num_learnable_im_end=0,
    )
    assert len(ds) == 5, f"expected 5 samples, got {len(ds)}"
    logger.info("  ✓ loaded %d samples", len(ds))

    # Check first sample structure.
    sample = ds[0]
    assert "prompt_text" in sample
    assert "prompt_ids" in sample
    assert "ground_truth" in sample
    assert isinstance(sample["prompt_ids"], list)
    assert len(sample["prompt_ids"]) > 0
    assert sample["ground_truth"] == "4"
    logger.info("  ✓ sample 0: prompt_ids len=%d, gt=%r", len(sample["prompt_ids"]), sample["ground_truth"])

    # Check messages-format sample (index 3).
    sample3 = ds[3]
    assert "France" in sample3["prompt_text"] or "Paris" in sample3["ground_truth"]
    logger.info("  ✓ sample 3 (messages format): gt=%r", sample3["ground_truth"])

    # Collate.
    batch = gspo_collate_fn([ds[0], ds[1], ds[2]])
    assert len(batch["prompt_text"]) == 3
    assert len(batch["prompt_ids"]) == 3
    assert len(batch["ground_truth"]) == 3
    logger.info("  ✓ collate_fn: %d items per key", len(batch["prompt_text"]))

    logger.info("  PASSED\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Phase B: Trainer Init (requires model + GPU)
# ═══════════════════════════════════════════════════════════════════════════════

def test_trainer_init(model_path: str):
    """Verify GSPOTrainer can be initialised."""
    logger.info("=== test_trainer_init (model: %s) ===", model_path)

    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from trainer import GSPOTrainer

    # Create a minimal config pointing to our sample data.
    data_path = os.path.join(_PROJECT_DIR, "data", "gspo_prompts.jsonl")
    config = GSPOConfig(
        model_path=model_path,
        train_data=data_path,
        max_seq_length=128,
        training_mode="gspo",
        gspo_group_size=2,
        gspo_num_mask_samples=1,
        gspo_gen_max_tokens=32,
        gspo_gen_window_size=8,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        attention_backend="dense",
        output_dir=os.path.join(_PROJECT_DIR, "outputs", "test_gspo"),
        use_deepspeed=False,
        num_train_epochs=1,
        seed=42,
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="bf16" if torch.cuda.is_available() else "no",
    )
    set_seed(42)

    trainer = GSPOTrainer(config, accelerator)
    logger.info("  ✓ trainer initialised")

    assert trainer.train_dataset is not None
    assert len(trainer.train_dataset) > 0
    logger.info("  ✓ dataset: %d prompts", len(trainer.train_dataset))

    assert trainer.model is not None
    assert trainer.old_model is not None
    assert trainer.ref_model is not None
    logger.info("  ✓ all 3 models loaded")

    # Run ONE training step to validate end-to-end.
    logger.info("  Running 1 training step (online gen + scoring + loss)...")
    batch = next(iter(trainer.train_dataloader))
    loss, logs = trainer.train_step(batch)

    assert loss.ndim == 0, f"loss not scalar: {loss.shape}"
    assert torch.isfinite(loss), "loss is NaN/Inf"
    logger.info("  ✓ train_step returned scalar loss: %.4f", loss.item())

    for key in ["grpo/loss", "grpo/clip_frac", "grpo/reward_mean", "gen/avg_comp_len"]:
        assert key in logs, f"missing log key: {key}"
    logger.info("  ✓ all expected log keys present")
    logger.info("  logs: %s", {k: f"{v.item():.4f}" for k, v in logs.items()})

    # Clean up.
    del trainer, config
    torch.cuda.empty_cache()
    logger.info("  PASSED\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Smoke test for GSPO trainer")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to WeDLM model for Phase B integration test")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("GSPO Trainer Smoke Test")
    logger.info("=" * 60)
    logger.info("Torch version: %s", torch.__version__)
    logger.info("CUDA available: %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        logger.info("CUDA device: %s", torch.cuda.get_device_name(0))
    logger.info("")

    # ── Phase A ──
    logger.info("-" * 40)
    logger.info("Phase A: Config + Data (no model)")
    logger.info("-" * 40)
    test_config()
    test_dataset_loading()
    logger.info("Phase A: ALL PASSED ✓")

    # ── Phase B (optional) ──
    if args.model_path:
        logger.info("-" * 40)
        logger.info("Phase B: Trainer init + 1 training step")
        logger.info("-" * 40)
        test_trainer_init(args.model_path)
        logger.info("Phase B: PASSED ✓")

    logger.info("=" * 60)
    logger.info("All tests passed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
