#!/usr/bin/env python
# coding=utf-8
"""WeDLM SFT Training Entry Script.

Usage:
    accelerate launch --multi_gpu --num_processes 4 --mixed_precision bf16 train.py --config configs/example.yaml
"""

import os
import argparse
import logging
import json

from accelerate import Accelerator
from accelerate.utils import set_seed, DeepSpeedPlugin

from src import WeDLMTrainingConfig, WeDLMTrainer

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="WeDLM SFT Training")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML file")
    parser.add_argument("--training_mode", type=str, choices=["sft", "dpo", "gspo"], default=None,
                        help="Training mode: sft, dpo or gspo")
    parser.add_argument("--model_path", type=str, default=None, help="Override model path")
    parser.add_argument("--train_data", type=str, default=None, help="Override training data path")
    parser.add_argument("--dpo_train_data", type=str, default=None,
                        help="Override DPO pairwise training data path")
    parser.add_argument("--gspo_train_data", type=str, default=None,
                        help="Override GSPO prompt data path")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--attention_backend", type=str, choices=["magi", "dense"], default=None,
                        help="Attention backend: magi or dense")
    parser.add_argument("--loss_weighting_scheme", type=str, choices=["uniform", "weighted"], default=None,
                        help="Loss weighting scheme")
    parser.add_argument("--dpo_beta", type=float, default=None, help="DPO beta temperature")
    parser.add_argument("--dpo_ref_model_path", type=str, default=None, help="Reference model path for DPO")
    parser.add_argument("--dpo_block_reduce", type=str, choices=["mean", "sum"], default=None,
                        help="Reduction for block-level scores in DPO")
    parser.add_argument("--dpo_seq_reduce", type=str, choices=["mean", "sum"], default=None,
                        help="Reduction for sequence-level scores in DPO")
    parser.add_argument("--dpo_num_mask_samples", type=int, default=None,
                        help="Number of mask Monte Carlo samples for DPO scoring")
    parser.add_argument("--dpo_length_norm", action=argparse.BooleanOptionalAction, default=None,
                        help="Enable length normalization in DPO scoring")
    parser.add_argument("--gspo_ref_model_path", type=str, default=None,
                        help="Reference model path for GSPO")
    parser.add_argument("--gspo_group_size", type=int, default=None,
                        help="Number of online candidates sampled per prompt")
    parser.add_argument("--gspo_min_candidates_per_group", type=int, default=None,
                        help="Minimum valid candidates required for each prompt group")
    parser.add_argument("--gspo_max_prompt_length", type=int, default=None,
                        help="Maximum prompt length for GSPO prompt dataset")
    parser.add_argument("--gspo_max_new_tokens", type=int, default=None,
                        help="Maximum new tokens during GSPO online rollout")
    parser.add_argument("--gspo_rollout_temperature", type=float, default=None,
                        help="Sampling temperature for GSPO online rollout")
    parser.add_argument("--gspo_rollout_top_p", type=float, default=None,
                        help="Top-p for GSPO online rollout")
    parser.add_argument("--gspo_rollout_top_k", type=int, default=None,
                        help="Top-k for GSPO online rollout")
    parser.add_argument("--gspo_score_temperature", type=float, default=None,
                        help="Temperature for GSPO policy distribution over group candidates")
    parser.add_argument("--gspo_reward_temperature", type=float, default=None,
                        help="Temperature for GSPO target reward distribution")
    parser.add_argument("--gspo_ref_alpha", type=float, default=None,
                        help="Reference score coefficient in GSPO policy logits")
    parser.add_argument("--gspo_kl_coef", type=float, default=None,
                        help="KL regularization coefficient for GSPO")
    parser.add_argument("--gspo_reward_beta", type=float, default=None,
                        help="Reward scaling beta for GSPO")
    parser.add_argument(
        "--gspo_reward_source",
        type=str,
        choices=[
            "policy_ref_margin",
            "length_penalized_margin",
            "deepmath_correctness_margin",
            "callable",
        ],
        default=None,
        help="GSPO reward function source",
    )
    parser.add_argument("--gspo_reward_length_penalty", type=float, default=None,
                        help="Linear length penalty coefficient for GSPO reward")
    parser.add_argument("--gspo_reward_callable", type=str, default=None,
                        help="Reward callable path in the form module:function")
    parser.add_argument("--gspo_reward_clip_min", type=float, default=None,
                        help="Optional min clip for GSPO reward")
    parser.add_argument("--gspo_reward_clip_max", type=float, default=None,
                        help="Optional max clip for GSPO reward")
    parser.add_argument("--gspo_deepmath_correct_bonus", type=float, default=None,
                        help="Correctness bonus for deepmath_correctness_margin reward")
    parser.add_argument("--gspo_deepmath_wrong_penalty", type=float, default=None,
                        help="Wrong-answer penalty for deepmath_correctness_margin reward")
    parser.add_argument("--rebuild_cache", action="store_true", help="Rebuild data cache")
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Load config
    config = WeDLMTrainingConfig.from_yaml(args.config) if args.config else WeDLMTrainingConfig()
    
    # Override config with command line arguments
    for key in [
        "training_mode",
        "model_path",
        "train_data",
        "dpo_train_data",
        "gspo_train_data",
        "output_dir",
        "attention_backend",
        "loss_weighting_scheme",
        "dpo_beta",
        "dpo_ref_model_path",
        "dpo_block_reduce",
        "dpo_seq_reduce",
        "dpo_num_mask_samples",
        "dpo_length_norm",
        "gspo_ref_model_path",
        "gspo_group_size",
        "gspo_min_candidates_per_group",
        "gspo_max_prompt_length",
        "gspo_max_new_tokens",
        "gspo_rollout_temperature",
        "gspo_rollout_top_p",
        "gspo_rollout_top_k",
        "gspo_score_temperature",
        "gspo_reward_temperature",
        "gspo_ref_alpha",
        "gspo_kl_coef",
        "gspo_reward_beta",
        "gspo_reward_source",
        "gspo_reward_length_penalty",
        "gspo_reward_callable",
        "gspo_reward_clip_min",
        "gspo_reward_clip_max",
        "gspo_deepmath_correct_bonus",
        "gspo_deepmath_wrong_penalty",
    ]:
        if getattr(args, key, None) is not None:
            setattr(config, key, getattr(args, key))
    config.rebuild_cache = args.rebuild_cache
    
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
    
    set_seed(config.seed)
    
    # Save config
    if accelerator.is_main_process:
        os.makedirs(config.output_dir, exist_ok=True)
        config.save_yaml(os.path.join(config.output_dir, "training_config.yaml"))
    
    # Train
    trainer = WeDLMTrainer(config, accelerator)
    trainer.train()


if __name__ == "__main__":
    main()
