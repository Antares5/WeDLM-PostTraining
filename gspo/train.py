#!/usr/bin/env python
# coding=utf-8
"""GSPO On-Policy Training Entry Script.

Usage:
    accelerate launch --multi_gpu --num_processes 4 --mixed_precision bf16 train.py --config configs/example.yaml
"""

import os
import argparse
import logging
import json

from accelerate import Accelerator
from accelerate.utils import set_seed, DeepSpeedPlugin

from src import GSPOTrainingConfig, GSPOTrainer

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="GSPO On-Policy Training")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to config YAML file"
    )
    # Model
    parser.add_argument("--model_path", type=str, default=None, help="Override model path")
    # Data
    parser.add_argument(
        "--gspo_prompt_data", type=str, default=None, help="Override prompt data path"
    )
    parser.add_argument(
        "--gspo_prompt_format",
        type=str,
        choices=["messages", "deepmath"],
        default=None,
        help="Prompt data format",
    )
    # GSPO Core
    parser.add_argument(
        "--gspo_num_samples", type=int, default=None, help="K: number of responses per prompt"
    )
    parser.add_argument(
        "--gspo_beta", type=float, default=None, help="GSPO temperature coefficient"
    )
    parser.add_argument(
        "--gspo_ref_model_path",
        type=str,
        default=None,
        help="Reference model path (null = use model_path)",
    )
    parser.add_argument(
        "--gspo_block_reduce",
        type=str,
        choices=["mean", "sum"],
        default=None,
        help="Block-level score reduction",
    )
    parser.add_argument(
        "--gspo_seq_reduce",
        type=str,
        choices=["mean", "sum"],
        default=None,
        help="Sequence-level score reduction",
    )
    parser.add_argument(
        "--gspo_num_mask_samples",
        type=int,
        default=None,
        help="MC mask samples for scoring",
    )
    # Generation
    parser.add_argument(
        "--gen_max_new_tokens", type=int, default=None, help="Max new tokens to generate"
    )
    parser.add_argument(
        "--gen_temperature", type=float, default=None, help="Generation temperature"
    )
    parser.add_argument(
        "--gen_top_p", type=float, default=None, help="Nucleus sampling top-p"
    )
    parser.add_argument(
        "--gen_top_k", type=int, default=None, help="Top-k sampling"
    )
    # Training
    parser.add_argument(
        "--max_seq_length", type=int, default=None, help="Override max sequence length"
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=None,
        help="Override per-device train batch size",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Override output directory"
    )
    parser.add_argument(
        "--attention_backend",
        type=str,
        choices=["magi", "dense"],
        default=None,
        help="Attention backend",
    )
    parser.add_argument(
        "--loss_weighting_scheme",
        type=str,
        choices=["uniform", "weighted"],
        default=None,
        help="Loss weighting scheme",
    )
    parser.add_argument(
        "--rebuild_cache", action="store_true", help="Rebuild data cache"
    )
    parser.add_argument(
        "--reward_type",
        type=str,
        choices=["math_verify", "model"],
        default=None,
        help="Reward type",
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Load config
    config = (
        GSPOTrainingConfig.from_yaml(args.config)
        if args.config
        else GSPOTrainingConfig()
    )

    # Override config with CLI args
    overrides = {
        "model_path": args.model_path,
        "gspo_prompt_data": args.gspo_prompt_data,
        "gspo_prompt_format": args.gspo_prompt_format,
        "gspo_num_samples": args.gspo_num_samples,
        "gspo_beta": args.gspo_beta,
        "gspo_ref_model_path": args.gspo_ref_model_path,
        "gspo_block_reduce": args.gspo_block_reduce,
        "gspo_seq_reduce": args.gspo_seq_reduce,
        "gspo_num_mask_samples": args.gspo_num_mask_samples,
        "gen_max_new_tokens": args.gen_max_new_tokens,
        "gen_temperature": args.gen_temperature,
        "gen_top_p": args.gen_top_p,
        "gen_top_k": args.gen_top_k,
        "max_seq_length": args.max_seq_length,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "output_dir": args.output_dir,
        "attention_backend": args.attention_backend,
        "loss_weighting_scheme": args.loss_weighting_scheme,
        "gspo_reward_type": args.reward_type,
    }
    for key, value in overrides.items():
        if value is not None:
            setattr(config, key, value)
    config.rebuild_cache = args.rebuild_cache

    # Setup DeepSpeed
    deepspeed_plugin = None
    if config.use_deepspeed:
        ds_config = config.get_deepspeed_config()
        if ds_config:
            os.makedirs(config.output_dir, exist_ok=True)
            ds_path = os.path.join(config.output_dir, "deepspeed_config.json")
            with open(ds_path, "w") as f:
                json.dump(ds_config, f, indent=2)
            deepspeed_plugin = DeepSpeedPlugin(
                hf_ds_config=ds_config,
                zero3_init_flag=(config.deepspeed_zero_stage == 3),
            )

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
        config.save_yaml(os.path.join(config.output_dir, "training_config.yaml"))
        logger.info(f"Training config:\n{json.dumps(config.__dict__, indent=2, default=str)}")

    # Create trainer and start training
    trainer = GSPOTrainer(config, accelerator)
    trainer.train()


if __name__ == "__main__":
    main()
