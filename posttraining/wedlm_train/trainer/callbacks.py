# coding=utf-8
"""Training callbacks: wandb logging, checkpoint saving.
Migrated from dpo/src/trainer.py (_init_wandb + _log_metrics + _save_checkpoint)."""

import os
import logging
from typing import Any, Dict, Optional

import torch
from accelerate import Accelerator

logger = logging.getLogger(__name__)

# Lazy import wandb
_wandb = None


def init_wandb(config, accelerator: Accelerator):
    """Initialize wandb if enabled (main process only)."""
    if not config.use_wandb or not accelerator.is_main_process:
        return None

    global _wandb
    try:
        import wandb
        _wandb = wandb
    except ImportError:
        logger.warning("wandb not installed, skipping wandb logging")
        return None

    if config.wandb_host:
        os.environ["WANDB_BASE_URL"] = config.wandb_host
    if config.wandb_key:
        os.environ["WANDB_API_KEY"] = config.wandb_key

    wandb.init(
        project=config.wandb_project or "wedlm-sft",
        entity=config.wandb_team,
        group=config.wandb_group,
        config={k: v for k, v in config.__dict__.items() if not k.startswith("_")},
    )
    return wandb


def log_metrics(
    wandb_run,
    logs: Dict[str, torch.Tensor],
    epoch: int,
    global_step: int,
    is_main_process: bool,
):
    """Log training metrics to console and wandb."""
    if is_main_process:
        log_str = f"Epoch {epoch} Step {global_step}: "
        log_str += ", ".join(
            f"{k}={v.item():.4f}" for k, v in logs.items() if isinstance(v, torch.Tensor)
        )
        logger.info(log_str)

        if wandb_run:
            wandb_run.log(
                {k: v.item() if hasattr(v, "item") else v for k, v in logs.items()},
                step=global_step,
            )


def save_checkpoint(
    accelerator: Accelerator,
    model: torch.nn.Module,
    tokenizer,
    output_dir: str,
    global_step: int,
    is_final: bool = False,
):
    """Save model + tokenizer checkpoint."""
    accelerator.wait_for_everyone()
    save_path = os.path.join(
        output_dir, "final" if is_final else f"checkpoint-{global_step}"
    )

    if accelerator.is_main_process:
        os.makedirs(save_path, exist_ok=True)
        accelerator.unwrap_model(model).save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        logger.info(f"Saved checkpoint to {save_path}")
