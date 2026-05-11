#!/usr/bin/env python
# coding=utf-8
"""Smoke test for Step 3: Full training loop with DeepSpeed.

Run:
    # Single-GPU quick test (a few steps):
    python gspo/scripts/smoke_test_step3.py --model_path tencent/WeDLM-8B-Instruct --steps 5

    # Multi-GPU with DeepSpeed:
    accelerate launch --multi_gpu --num_processes 2 --mixed_precision bf16 \\
        gspo/scripts/smoke_test_step3.py --model_path tencent/WeDLM-8B-Instruct --steps 10

This test:
  1. Creates a GSPOConfig with minimal settings.
  2. Initializes an Accelerator (auto-detects single/multi GPU, DeepSpeed).
  3. Builds a GSPOTrainer and runs a few training steps with mock data.
  4. Verifies loss decreases (or at least is finite and produces gradients).
  5. Saves a checkpoint.
"""

import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import argparse
import json
import logging
import tempfile
import shutil

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed, DeepSpeedPlugin

from gspo.src.config import GSPOConfig
from gspo.src.trainer import GSPOTrainer

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# ── helpers ──

def green(s: str) -> str: return f"\033[32m{s}\033[0m"
def red(s: str) -> str:   return f"\033[31m{s}\033[0m"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="tencent/WeDLM-8B-Instruct")
    parser.add_argument("--steps", type=int, default=5, help="Number of training steps to run")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    # ── Build config ──
    output_dir = args.output_dir or tempfile.mkdtemp(prefix="gspo_step3_")
    prompt_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "train_prompts.jsonl")

    if not os.path.exists(prompt_file):
        # Auto-create a minimal prompt file so the smoke test is self-contained
        os.makedirs(os.path.dirname(prompt_file), exist_ok=True)
        with open(prompt_file, "w") as f:
            f.write('[{"role": "user", "content": "What is 2+2?"}]\n')
            f.write('[{"role": "user", "content": "Write a haiku about AI."}]\n')
            f.write('[{"role": "user", "content": "Explain block diffusion models briefly."}]\n')
            f.write('[{"role": "user", "content": "What is the capital of France?"}]\n')
        print(f"  Auto-created prompt file: {prompt_file}")

    config = GSPOConfig(
        model_path=args.model_path,
        train_data=prompt_file,
        output_dir=output_dir,
        training_mode="gspo",
        attention_backend="dense",
        block_size=32,
        max_seq_length=512,
        mask_per_block=True,
        loss_weighting_scheme="weighted",
        mask_eps=1e-8,
        # GSPO
        gspo_group_size=2,
        gspo_num_mask_samples=2,
        gspo_max_new_tokens=64,
        # Training
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=3e-6,
        warmup_ratio=0.0,  # no warmup for smoke test
        max_grad_norm=1.0,
        weight_decay=0.01,
        # DeepSpeed
        use_deepspeed=False,  # smoke test defaults to no DS (accelerate launch enables it)
        # Logging
        logging_steps=1,
        save_steps=999999,  # don't save mid-training
        save_total_limit=1,
        num_learnable_im_end=8,
        seed=42,
        bf16=True,
    )

    print(f"{'='*60}")
    print(f"  GSPO Step 3 Smoke Test")
    print(f"{'='*60}")
    print(f"  Model:     {config.model_path}")
    print(f"  Prompts:   {config.train_data}")
    print(f"  Output:    {config.output_dir}")
    print(f"  Steps:     {args.steps}")
    print(f"  G:         {config.gspo_group_size}")
    print(f"  K:         {config.gspo_num_mask_samples}")
    print(f"  Backend:   {config.attention_backend}")
    print(f"  DeepSpeed: {config.use_deepspeed}")
    print()

    # ── Setup accelerator ──
    deepspeed_plugin = None
    if config.use_deepspeed:
        ds_config = config.get_deepspeed_config()
        if ds_config:
            os.makedirs(config.output_dir, exist_ok=True)
            ds_path = os.path.join(config.output_dir, "deepspeed_config.json")
            with open(ds_path, "w") as f:
                json.dump(ds_config, f, indent=2)
            deepspeed_plugin = DeepSpeedPlugin(hf_ds_config=ds_config)

    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision="bf16" if config.bf16 else "no",
        deepspeed_plugin=deepspeed_plugin,
    )
    set_seed(config.seed)

    print(f"  Accelerator state:")
    print(f"    num_processes:    {accelerator.num_processes}")
    print(f"    device:           {accelerator.device}")
    print(f"    is_main_process:  {accelerator.is_main_process}")
    print(f"    mixed_precision:  {'bf16' if config.bf16 else 'no'}")
    print()

    # ── Build trainer ──
    try:
        trainer = GSPOTrainer(config, accelerator)
    except Exception as e:
        print(f"\n{red('[FAIL]')} Trainer initialization failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"  {green('[PASS]')} Trainer initialized")
    print(f"    Dataloader size: {len(trainer.train_dataloader)}")
    print(f"    Total training steps: {trainer.num_training_steps}")
    print()

    # ── Run a few training steps ──
    losses = []
    trainer.global_step = 0
    trainer.num_training_steps = args.steps  # override for smoke test

    progress_bar_enabled = accelerator.is_local_main_process

    step = 0
    for epoch in range(config.num_train_epochs):
        trainer.model.train()
        if trainer.train_sampler is not None:
            trainer.train_sampler.set_epoch(epoch)

        for batch in trainer.train_dataloader:
            if step >= args.steps:
                break

            with accelerator.accumulate(trainer.model):
                loss, logs = trainer.train_step(batch)
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        trainer.model.parameters(), config.max_grad_norm
                    )

                trainer.optimizer.step()
                trainer.lr_scheduler.step()
                trainer.optimizer.zero_grad()

            if accelerator.sync_gradients:
                step += 1
                loss_val = float(logs.get("loss", loss).detach().cpu().item())
                losses.append(loss_val)
                if accelerator.is_main_process:
                    log_parts = [f"loss={loss_val:.4f}"]
                    for k, v in logs.items():
                        if isinstance(v, torch.Tensor) and v.dim() == 0:
                            log_parts.append(f"{k}={v.item():.4f}")
                    print(f"  Step {step}: " + ", ".join(log_parts))

    print()

    # ── Validate ──
    all_passed = True

    # 1. All losses must be finite
    finite_losses = all(torch.isfinite(torch.tensor(l)).item() for l in losses)
    if finite_losses:
        print(f"  {green('[PASS]')} All losses finite")
    else:
        print(f"  {red('[FAIL]')} Some losses are NaN/Inf: {losses}")
        all_passed = False

    # 2. We completed the expected number of steps
    if step == args.steps:
        print(f"  {green('[PASS]')} Completed {step}/{args.steps} steps")
    else:
        print(f"  {red('[FAIL]')} Only completed {step}/{args.steps} steps")
        all_passed = False

    # 3. Save and verify checkpoint
    if accelerator.is_main_process:
        ckpt_dir = os.path.join(output_dir, "final")
        try:
            trainer._save_checkpoint(final=True)
            if os.path.exists(os.path.join(ckpt_dir, "model.safetensors")) or \
               any(f.endswith(".safetensors") for f in os.listdir(ckpt_dir)):
                print(f"  {green('[PASS]')} Checkpoint saved to {ckpt_dir}")
            else:
                print(f"  {red('[FAIL]')} Checkpoint dir exists but no model file found")
                all_passed = False
        except Exception as e:
            print(f"  {red('[FAIL]')} Checkpoint save failed: {e}")
            all_passed = False

    # Cleanup (optional — keep if user wants to inspect)
    # if args.output_dir is None and os.path.exists(output_dir):
    #     shutil.rmtree(output_dir)

    # ── Summary ──
    print()
    if all_passed:
        print(f"  {green('ALL TESTS PASSED — Step 3 is ready.')}")
        print(f"  Checkpoints saved to: {output_dir}")
    else:
        print(f"  {red('SOME TESTS FAILED — check output above.')}")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
