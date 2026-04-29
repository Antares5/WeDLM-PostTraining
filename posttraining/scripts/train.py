#!/usr/bin/env python
# coding=utf-8
"""WeDLM PostTraining unified entry script.

Usage:
    accelerate launch --multi_gpu --num_processes 4 --mixed_precision bf16 \
        scripts/train.py --config configs/sft_example.yaml

    # With overrides
    accelerate launch scripts/train.py --config configs/dpo_example.yaml \
        --override training_mode=dpo --override dpo.beta=0.5
"""

import os
import argparse
import logging
import json
import sys

# Ensure wedlm_train is importable from the posttraining parent
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

from accelerate import Accelerator
from accelerate.utils import set_seed as acc_set_seed, DeepSpeedPlugin

from wedlm_train.config import from_yaml
from wedlm_train.trainer import SFTTrainer, DPOTrainer, GSPOTrainer

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="WeDLM PostTraining")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config YAML file"
    )
    parser.add_argument(
        "--override",
        type=str,
        nargs="*",
        default=[],
        help="Override config values in key=value format (e.g. dpo.beta=0.5)",
    )
    parser.add_argument(
        "--rebuild_cache",
        action="store_true",
        help="Rebuild data cache",
    )
    return parser.parse_args()


def apply_overrides(config, overrides):
    """Apply CLI overrides to config fields (supports nested keys with dot)."""
    for override_str in overrides:
        if "=" not in override_str:
            logger.warning(f"Skipping malformed override: {override_str}")
            continue

        key, value_str = override_str.split("=", 1)
        key = key.strip()
        value_str = value_str.strip()

        # Type coercion
        if value_str.lower() == "true":
            value = True
        elif value_str.lower() == "false":
            value = False
        elif value_str.lower() == "none":
            value = None
        else:
            try:
                value = int(value_str)
            except ValueError:
                try:
                    value = float(value_str)
                except ValueError:
                    value = value_str

        if hasattr(config, key):
            setattr(config, key, value)
            logger.info(f"Override: {key} = {value}")
        else:
            logger.warning(f"Unknown config key: {key}, skipping override")


def get_trainer_cls(config):
    """Return the appropriate Trainer class by training_mode."""
    mode = config.training_mode
    if mode == "dpo":
        return DPOTrainer
    elif mode == "gspo":
        return GSPOTrainer
    else:
        return SFTTrainer


def main():
    """Main entry point."""
    args = parse_args()

    # Load config
    config = from_yaml(args.config)
    apply_overrides(config, args.override)
    config.rebuild_cache = args.rebuild_cache

    logger.info(f"Training mode: {config.training_mode}")
    logger.info(f"Model: {config.model_path}")

    # Setup DeepSpeed if enabled
    deepspeed_plugin = None
    if config.use_deepspeed:
        ds_config = config.get_deepspeed_config()
        if ds_config:
            os.makedirs(config.output_dir, exist_ok=True)
            ds_path = os.path.join(config.output_dir, "deepspeed_config.json")
            with open(ds_path, "w") as f:
                json.dump(ds_config, f, indent=2)
            deepspeed_plugin = DeepSpeedPlugin(hf_ds_config=ds_config)

    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision="bf16" if config.bf16 else "no",
        deepspeed_plugin=deepspeed_plugin,
    )

    acc_set_seed(config.seed)

    # Save config
    if accelerator.is_main_process:
        os.makedirs(config.output_dir, exist_ok=True)
        config.save_yaml(os.path.join(config.output_dir, "training_config.yaml"))

    # Train
    trainer_cls = get_trainer_cls(config)
    trainer = trainer_cls(config, accelerator)
    trainer.train()


if __name__ == "__main__":
    main()
