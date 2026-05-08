#!/usr/bin/env python
# coding=utf-8
"""GSPO Training Entry Script.

Usage:
    accelerate launch --multi_gpu --num_processes 4 --mixed_precision bf16 \\
        train.py --config configs/example.yaml
"""

import os
import sys
import argparse
import logging
import json

# Ensure dpo/src and gspo/src are on Python path.
# IMPORTANT: dpo/ must come FIRST so that ``import src`` resolves to dpo/src.
# gspo-specific modules (config, data, generator, scorer, loss, trainer)
# are imported by their module name directly AFTER gspo/src is on path.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, os.path.join(_PARENT_DIR, "dpo"))       # for src.* (dpo/src)
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "src"))        # for gspo modules

from accelerate import Accelerator
from accelerate.utils import set_seed, DeepSpeedPlugin

from config import GSPOConfig
from trainer import GSPOTrainer

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="GSPO Training for WeDLM")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    # Overrides.
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--train_data", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--attention_backend", type=str, choices=["magi", "dense"], default=None)
    parser.add_argument("--gspo_group_size", type=int, default=None)
    parser.add_argument("--gspo_clip_epsilon", type=float, default=None)
    parser.add_argument("--gspo_old_model_update_steps", type=int, default=None)
    parser.add_argument("--gspo_kl_beta", type=float, default=None)
    parser.add_argument("--gspo_reward_type", type=str, choices=["math_verify", "string_match"], default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = GSPOConfig.from_yaml(args.config)

    # Apply CLI overrides.
    overrides = [
        "model_path", "train_data", "output_dir", "attention_backend",
        "gspo_group_size", "gspo_clip_epsilon", "gspo_old_model_update_steps",
        "gspo_kl_beta", "gspo_reward_type",
    ]
    for key in overrides:
        val = getattr(args, key, None)
        if val is not None:
            setattr(config, key, val)

    # DeepSpeed.
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

    if accelerator.is_main_process:
        os.makedirs(config.output_dir, exist_ok=True)
        config.save_yaml(os.path.join(config.output_dir, "training_config.yaml"))
        logger.info("Config saved to %s", os.path.join(config.output_dir, "training_config.yaml"))

    trainer = GSPOTrainer(config, accelerator)
    trainer.train()


if __name__ == "__main__":
    main()
