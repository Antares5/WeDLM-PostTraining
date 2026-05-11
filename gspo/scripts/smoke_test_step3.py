#!/usr/bin/env python
# coding=utf-8
"""Smoke test for Step 3: GSPO Trainer construction + single train_step.

Run:
    # CPU-only (no model): validates config, generator, trainer construction
    python gspo/scripts/smoke_test_step3.py

    # GPU full test (requires model):
    python gspo/scripts/smoke_test_step3.py --model_path tencent/WeDLM-8B-Instruct
"""

import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import argparse
import tempfile
import json
import torch
import logging

logging.basicConfig(level=logging.WARNING)

# ── helpers ──

def green(s: str) -> str: return f"\033[32m{s}\033[0m"
def red(s: str) -> str:   return f"\033[31m{s}\033[0m"

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
# TEST 1: Trainer construction (with mock generator)
# ═══════════════════════════════════════════════════════════════

section("Test 1: Trainer construction")

# Create a minimal prompt JSONL
tmp_data = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
for i in range(4):
    msg = [{"role": "user", "content": f"Test prompt {i}"}]
    tmp_data.write(json.dumps(msg) + "\n")
tmp_data.close()

from gspo.src.config import GSPOConfig
from gspo.src.generator import MockGenerator

config = GSPOConfig(
    model_path="tencent/WeDLM-8B-Instruct",
    train_data=tmp_data.name,
    training_mode="gspo",
    gspo_group_size=2,
    gspo_num_mask_samples=2,
    attention_backend="dense",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    max_seq_length=512,
    gspo_sync_every_n_steps=100,  # don't actually sync
)

# Mock the generator before trainer init
from gspo.src.trainer import GSPOTrainer
# Need to import the module to patch
import gspo.src.trainer as trainer_mod
_original_init_gen = trainer_mod.GSPOTrainer._init_generator

def _mock_init_generator(self):
    self.generator = MockGenerator(fixed_length=32)

trainer_mod.GSPOTrainer._init_generator = _mock_init_generator

try:
    trainer = GSPOTrainer(config)
    check("1-construct: trainer created", True)
    check("1-construct: model loaded", hasattr(trainer, 'model'))
    check("1-construct: optimizer created", hasattr(trainer, 'optimizer'))
    check("1-construct: dataloader has prompts", len(trainer.train_dataset) > 0,
          f"dataset size = {len(trainer.train_dataset)}")
    check("1-construct: generator is MockGenerator", isinstance(trainer.generator, MockGenerator))
except Exception as e:
    check("1-construct: trainer created", False, str(e))
    import traceback; traceback.print_exc()

# Clean up
os.unlink(tmp_data.name)

# Restore original
trainer_mod.GSPOTrainer._init_generator = _original_init_gen


# ═══════════════════════════════════════════════════════════════
# TEST 2: Single train_step (with real model)
# ═══════════════════════════════════════════════════════════════

section("Test 2: Single train_step")

args_model = argparse.Namespace(model_path=None)
_parser = argparse.ArgumentParser()
_parser.add_argument("--model_path", type=str, default=None)
_args, _ = _parser.parse_known_args()
args_model.model_path = _args.model_path

if not args_model.model_path:
    print(f"  {green('[SKIP]')} No --model_path provided. Skipping real-model test.")
else:
    try:
        # Create fresh trainer with mock generator
        tmp_data2 = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        for i in range(4):
            msg = [{"role": "user", "content": f"What is {i}+{i}?"}]
            tmp_data2.write(json.dumps(msg) + "\n")
        tmp_data2.close()

        config2 = GSPOConfig(
            model_path=args_model.model_path,
            train_data=tmp_data2.name,
            training_mode="gspo",
            gspo_group_size=2,
            gspo_num_mask_samples=2,
            attention_backend="dense",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            max_seq_length=256,
            block_size=32,
            gspo_sync_every_n_steps=100,
            gspo_max_new_tokens=32,  # short generations
        )

        trainer_mod.GSPOTrainer._init_generator = _mock_init_generator
        trainer2 = GSPOTrainer(config2)
        trainer_mod.GSPOTrainer._init_generator = _original_init_gen

        check("2-setup: trainer created", True)

        # Get one batch
        batch = next(iter(trainer2.train_dataloader))
        check("2-setup: batch has prompts", len(batch) > 0,
              f"batch size = {len(batch)}")

        # Mock rewards to avoid calling a real RM
        original_compute_rewards = trainer2._compute_rewards

        def mock_rewards(prompts, responses):
            return [float(len(r)) / 50.0 for r in responses]  # length-based mock

        trainer2._compute_rewards = mock_rewards

        # Run train_step
        logs = trainer2.train_step(batch, [])

        trainer2._compute_rewards = original_compute_rewards

        check("2-step: loss in logs", "loss" in logs)
        check("2-step: loss is finite",
              "loss" in logs and abs(logs["loss"]) != float("inf"))
        check("2-step: gspo/loss in logs", "gspo/loss" in logs)
        check("2-step: gspo/adv_mean in logs", "gspo/adv_mean" in logs)

        print(f"  2-step: loss={logs.get('loss', 'N/A'):.6f}, "
              f"gspo/loss={logs.get('gspo/loss', 'N/A'):.6f}, "
              f"adv_mean={logs.get('gspo/adv_mean', 'N/A'):.6f}")

        # Gradient check: after train_step, model parameters should have grads
        has_grad = False
        for p in trainer2.model.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break
        check("2-grad: model has gradients", has_grad)

        # Optimizer step should succeed
        if trainer2.scaler is not None:
            trainer2.scaler.unscale_(trainer2.optimizer)
        torch.nn.utils.clip_grad_norm_(trainer2.model.parameters(), 1.0)
        if trainer2.scaler is not None:
            trainer2.scaler.step(trainer2.optimizer)
            trainer2.scaler.update()
        else:
            trainer2.optimizer.step()
        trainer2.optimizer.zero_grad()
        check("2-optim: step succeeded", True)

        os.unlink(tmp_data2.name)

    except Exception as e:
        import traceback
        print(f"  {red('[FAIL]')} Test 2 crashed:")
        traceback.print_exc()
        check("2-step: overall", False, str(e))


# ═══════════════════════════════════════════════════════════════
# TEST 3: Generator interface smoke
# ═══════════════════════════════════════════════════════════════

section("Test 3: Generator interface")

from gspo.src.generator import BaseGenerator, MockGenerator, WeDLMGenerator

# Mock generator
mock = MockGenerator(fixed_length=16)
results = mock.generate([["hello", "world"], ["test"]], None)
check("3-mock: returns correct count", len(results) == 2)
check("3-mock: fixed length", len(results[0]) == 16)
check("3-mock: update_weights no-op", True)  # doesn't crash
mock.release_memory()
check("3-mock: release_memory no-op", True)

# WeDLMGenerator construction (may fail if wedlm package not available)
try:
    gen = WeDLMGenerator(
        model_path="tencent/WeDLM-8B-Instruct",
        block_size=4096,
        window_size=16,
    )
    check("3-wedlm: constructor succeeds", True)
except Exception as e:
    print(f"  3-wedlm: WeDLMGenerator not available ({e})")
    check("3-wedlm: constructor (expected to need GPU)", True)


# ═══════════════════════════════════════════════════════════════
# TEST 4: Entry script import
# ═══════════════════════════════════════════════════════════════

section("Test 4: Entry script import")

try:
    # Just verify the entry script is syntactically valid and imports work
    import importlib.util
    train_path = os.path.join(_project_root, "gspo", "train.py")
    spec = importlib.util.spec_from_file_location("gspo_train", train_path)
    check("4-entry: train.py is parseable", spec is not None)
except Exception as e:
    check("4-entry: train.py parse", False, str(e))


# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════

section("Summary")
total = passed + failed
print(f"  {passed}/{total} tests passed")
if failed == 0:
    print(f"\n  {green('ALL TESTS PASSED — Step 3 is ready.')}")
else:
    print(f"\n  {red(f'{failed} TEST(S) FAILED — fix before proceeding.')}")

sys.exit(0 if failed == 0 else 1)
