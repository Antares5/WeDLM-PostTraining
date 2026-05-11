#!/usr/bin/env python
# coding=utf-8
"""GSPO On-Policy Training Entry Script.

Usage:
    # Basic training
    python gspo/train.py --config gspo/configs/example.yaml

    # With DeepSpeed (if supported by your environment)
    deepspeed gspo/train.py --config gspo/configs/example.yaml \
        --deepspeed --deepspeed_config ds_config.json
"""

import os
import sys
import argparse
import logging

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="GSPO On-Policy Training")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config YAML file")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Override model path")
    parser.add_argument("--train_data", type=str, default=None,
                        help="Override training data path")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory")
    parser.add_argument("--mock_generator", action="store_true",
                        help="Use mock generator (for testing without WeDLM engine)")
    return parser.parse_args()


def main():
    args = parse_args()

    from gspo.src.config import GSPOConfig
    from gspo.src.trainer import GSPOTrainer
    from gspo.src.generator import MockGenerator

    config = GSPOConfig.from_yaml(args.config)

    # CLI overrides
    if args.model_path:
        config.model_path = args.model_path
    if args.train_data:
        config.train_data = args.train_data
    if args.output_dir:
        config.output_dir = args.output_dir

    os.makedirs(config.output_dir, exist_ok=True)
    config.save_yaml(os.path.join(config.output_dir, "config.yaml"))

    logger.info(f"GSPO Config: G={config.gspo_group_size}, "
                f"K={config.gspo_num_mask_samples}, "
                f"sync_every={config.gspo_sync_every_n_steps}")

    trainer = GSPOTrainer(config)

    if args.mock_generator:
        logger.info("Using MockGenerator (no real generation)")
        trainer.generator = MockGenerator(fixed_length=64)

    trainer.train()


if __name__ == "__main__":
    main()
