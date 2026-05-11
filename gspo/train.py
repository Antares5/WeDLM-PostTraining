#!/usr/bin/env python
# coding=utf-8
"""GSPO on-policy RL training entry script.

Usage:
    # Single GPU:
    python gspo/train.py --config gspo/configs/example.yaml

    # Multi-GPU with DeepSpeed ZeRO-2:
    accelerate launch --multi_gpu --num_processes 4 --mixed_precision bf16 \\
        gspo/train.py --config gspo/configs/example.yaml

    # Multi-GPU with DeepSpeed ZeRO-3:
    accelerate launch --multi_gpu --num_processes 8 --mixed_precision bf16 \\
        gspo/train.py --config gspo/configs/example.yaml
"""

import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import argparse
import json
import logging

from accelerate import Accelerator
from accelerate.utils import set_seed, DeepSpeedPlugin

from gspo.src.config import GSPOConfig
from gspo.src.trainer import GSPOTrainer

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="GSPO On-Policy RL Training")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML file")
    parser.add_argument("--model_path", type=str, default=None, help="Override model path")
    parser.add_argument("--train_data", type=str, default=None, help="Override training data path (JSONL of prompts)")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--attention_backend", type=str, choices=["magi", "dense"], default=None,
                        help="Attention backend: magi or dense")
    parser.add_argument("--gspo_group_size", type=int, default=None, help="G: responses per prompt")
    parser.add_argument("--gspo_num_mask_samples", type=int, default=None, help="K: MC masking samples")
    parser.add_argument("--gspo_temperature", type=float, default=None, help="Generation temperature")
    parser.add_argument("--gspo_max_new_tokens", type=int, default=None, help="Max tokens to generate")
    parser.add_argument("--gspo_sync_every_n_steps", type=int, default=None, help="Weight sync interval")
    parser.add_argument("--rebuild_cache", action="store_true", help="Rebuild data cache")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    if args.config:
        config = GSPOConfig.from_yaml(args.config)
    else:
        config = GSPOConfig()

    # CLI overrides
    for key in [
        "model_path", "train_data", "output_dir", "attention_backend",
        "gspo_group_size", "gspo_num_mask_samples", "gspo_temperature",
        "gspo_max_new_tokens", "gspo_sync_every_n_steps",
    ]:
        val = getattr(args, key, None)
        if val is not None:
            setattr(config, key, val)
    config.rebuild_cache = args.rebuild_cache

    # DeepSpeed setup
    deepspeed_plugin = None
    if config.use_deepspeed:
        ds_config = config.get_deepspeed_config()
        if ds_config:
            os.makedirs(config.output_dir, exist_ok=True)
            ds_path = os.path.join(config.output_dir, "deepspeed_config.json")
            with open(ds_path, "w") as f:
                json.dump(ds_config, f, indent=2)
            deepspeed_plugin = DeepSpeedPlugin(hf_ds_config=ds_config)
            logger.info(f"DeepSpeed ZeRO-{config.deepspeed_zero_stage} enabled")

    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision="bf16" if config.bf16 else "no",
        deepspeed_plugin=deepspeed_plugin,
    )
    set_seed(config.seed)

    # Save config
    if accelerator.is_main_process:
        os.makedirs(config.output_dir, exist_ok=True)
        config.save_yaml(os.path.join(config.output_dir, "gspo_config.yaml"))
        logger.info(f"Config saved to {config.output_dir}/gspo_config.yaml")

    # Build trainer and run
    trainer = GSPOTrainer(config, accelerator)
    trainer.train()


if __name__ == "__main__":
    main()
