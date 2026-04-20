#!/usr/bin/env python
# coding=utf-8
"""Smoke test for increment-4 DPO trainer step wiring.

This script validates that:
1) DPO data collate outputs required tensor keys.
2) Trainer can be initialized in DPO mode.
3) A single train_step_dpo call returns scalar loss and log dictionary.

Usage:
    python scripts/smoke_test_dpo_train_step.py --config configs/example.yaml --data_path data/pairwise_train.jsonl
"""

import argparse

from accelerate import Accelerator

from src import WeDLMTrainer, WeDLMTrainingConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke test DPO trainer step")
    parser.add_argument("--config", type=str, default=None, help="Path to config yaml")
    parser.add_argument("--data_path", type=str, default=None, help="Path to pairwise jsonl")
    parser.add_argument("--batch_size", type=int, default=None, help="Override per-device train batch size")
    parser.add_argument("--max_steps", type=int, default=1, help="Number of local test steps")
    return parser.parse_args()


def main():
    args = parse_args()
    config = WeDLMTrainingConfig.from_yaml(args.config) if args.config else WeDLMTrainingConfig()

    config.training_mode = "dpo"
    if args.data_path is not None:
        config.dpo_train_data = args.data_path

    if args.batch_size is not None:
        config.per_device_train_batch_size = args.batch_size

    # This is only a smoke test, keep runtime short.
    config.num_train_epochs = 1
    config.gradient_accumulation_steps = 1

    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision="bf16" if config.bf16 else "no",
    )
    trainer = WeDLMTrainer(config, accelerator)

    iterator = iter(trainer.train_dataloader)
    for step in range(args.max_steps):
        batch = next(iterator)
        loss, logs = trainer.train_step_dpo(batch)
        print(f"step={step} loss={float(loss.detach().item()):.6f}")
        print(f"logged_keys={sorted(list(logs.keys()))[:12]}...")

    print("DPO trainer step smoke test passed.")


if __name__ == "__main__":
    main()
