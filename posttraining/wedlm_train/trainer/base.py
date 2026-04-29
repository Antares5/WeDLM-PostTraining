# coding=utf-8
"""Base Trainer with shared training infrastructure.
Refactored from dpo/src/trainer.py (WeDLMTrainer common parts)."""

import os
import math
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler

from ..batch import WeDLMBatch, build_wedlm_batch
from ..model import wedlm_forward
from ..loss import compute_mlm_loss, compute_ar_loss, compute_block_scores
from ..attention import check_backend_available, get_available_backend, get_attention_wrapper
from ..data import get_im_end_token_id
from .callbacks import init_wandb, log_metrics, save_checkpoint

logger = logging.getLogger(__name__)

MASK_TOKEN_ID = 151665


class BaseTrainer(ABC):
    """Abstract base trainer for WeDLM post-training.

    Subclasses must implement:
        _setup()         — dataset, dataloader, model/ref_model initialization
        train_step()     — single training step returning (loss, logs)
    """

    def __init__(self, config, accelerator: Accelerator):
        self.config = config
        self.accelerator = accelerator
        self.wandb = init_wandb(config, accelerator)

        # Validate attention backend
        if not check_backend_available(self.config.attention_backend):
            self.config.attention_backend = get_available_backend()
        logger.info(f"Attention backend: {self.config.attention_backend}")
        logger.info(f"Training mode: {self.config.training_mode}")

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path, trust_remote_code=self.config.trust_remote_code
        )
        self.im_end_token_id = get_im_end_token_id(self.tokenizer)
        self.tokenizer.pad_token_id = self.im_end_token_id

        # Model
        model_kwargs = {
            "trust_remote_code": self.config.trust_remote_code,
            "torch_dtype": torch.bfloat16 if self.config.bf16 else torch.float32,
            "attn_implementation": "eager",
        }
        if self.config.use_deepspeed and self.config.deepspeed_zero_stage == 3:
            model_kwargs["low_cpu_mem_usage"] = True
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path, **model_kwargs
        )
        self.ref_model: Optional[torch.nn.Module] = None

        # Attention wrapper
        head_dim = self.model.config.hidden_size // self.model.config.num_attention_heads
        self.attn_wrapper = get_attention_wrapper(
            self.config.attention_backend,
            head_dim,
            deterministic=False,
        )
        if hasattr(self.attn_wrapper, "to"):
            self.attn_wrapper = self.attn_wrapper.to(self.accelerator.device)

        # Subclass-specific setup (dataset, dataloader, ref_model, reward_fn)
        self._setup()
        self._prepare_training()

    @abstractmethod
    def _setup(self):
        """Subclasses override to initialize dataset, dataloader, ref model, etc."""
        ...

    @abstractmethod
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        """Subclasses override to implement mode-specific training step."""
        ...

    # ── Common infrastructure ──────────────────────────────────────

    def _prepare_training(self):
        """Prepare optimizer, scheduler, and accelerator."""
        steps_per_epoch = len(self.train_dataloader)

        num_update_steps_per_epoch = math.ceil(
            steps_per_epoch / self.config.gradient_accumulation_steps
        )
        self.num_training_steps = (
            num_update_steps_per_epoch * self.config.num_train_epochs
        )
        num_warmup_steps = int(self.num_training_steps * self.config.warmup_ratio)

        if self.accelerator.is_main_process:
            total_batches = len(self.train_dataset)
            logger.info("=== Training Configuration ===")
            logger.info(f"Number of GPUs: {self.accelerator.num_processes}")
            logger.info(f"Total batches in dataset: {total_batches}")
            logger.info(f"Batches per GPU per epoch: {steps_per_epoch}")
            logger.info(
                f"Gradient accumulation steps: {self.config.gradient_accumulation_steps}"
            )
            logger.info(f"Update steps per epoch: {num_update_steps_per_epoch}")
            logger.info(f"Total training steps: {self.num_training_steps}")
            logger.info(f"Warmup steps: {num_warmup_steps}")

        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_groups = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = torch.optim.AdamW(
            optimizer_groups, lr=self.config.learning_rate
        )

        self.lr_scheduler = get_scheduler(
            self.config.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=self.num_training_steps,
        )

        (
            self.model,
            self.optimizer,
            self.train_dataloader,
            self.lr_scheduler,
        ) = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader, self.lr_scheduler
        )

        # Prepare reference model for DPO/GSPO
        if self.config.training_mode in ["dpo", "gspo"] and self.ref_model is not None:
            try:
                self.ref_model = self.accelerator.prepare_model(
                    self.ref_model, evaluation_mode=True
                )
            except Exception as err:
                logger.warning(
                    f"Failed to prepare reference model with Accelerator ({err}), fallback to .to(device)."
                )
                self.ref_model = self.ref_model.to(self.accelerator.device)
            self.ref_model.eval()

        self.global_step = 0

    def train(self):
        """Main training loop (shared across all modes)."""
        logger.info(
            f"Starting training: {len(self.train_dataloader)} batches per GPU, "
            f"{self.num_training_steps} total update steps"
        )

        progress_bar = tqdm(
            total=self.num_training_steps,
            disable=not self.accelerator.is_local_main_process,
        )

        for epoch in range(self.config.num_train_epochs):
            if self.config.training_mode == "dpo":
                self.model.eval()
            else:
                self.model.train()

            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)

            for batch in self.train_dataloader:
                with self.accelerator.accumulate(self.model):
                    loss, logs = self.train_step(batch)
                    if self.config.training_mode != "dpo":
                        self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.model.parameters(), self.config.max_grad_norm
                        )

                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                if self.accelerator.sync_gradients:
                    self.global_step += 1
                    progress_bar.update(1)
                    progress_bar.set_postfix(loss=f"{logs['loss'].item():.4f}")

                    if self.global_step % self.config.logging_steps == 0:
                        log_metrics(
                            self.wandb,
                            logs,
                            epoch,
                            self.global_step,
                            self.accelerator.is_main_process,
                        )
                    if self.global_step % self.config.save_steps == 0:
                        save_checkpoint(
                            self.accelerator,
                            self.model,
                            self.tokenizer,
                            self.config.output_dir,
                            self.global_step,
                            is_final=False,
                        )

        progress_bar.close()
        save_checkpoint(
            self.accelerator,
            self.model,
            self.tokenizer,
            self.config.output_dir,
            self.global_step,
            is_final=True,
        )
        if self.wandb:
            self.wandb.finish()
        logger.info("Training complete!")

    # ── Shared helpers ─────────────────────────────────────────────

    def _forward_wedlm_logits(
        self, model: torch.nn.Module, batch: WeDLMBatch
    ) -> torch.Tensor:
        """Forward helper for WeDLM logits."""
        try:
            forward_model = self.accelerator.unwrap_model(model)
        except Exception:
            forward_model = model

        return wedlm_forward(
            forward_model, batch, self.attn_wrapper, self.config.attention_backend
        )

    def _compute_block_scores_for_batch(
        self, logits: torch.Tensor, batch: WeDLMBatch
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute sequence scores from masked block log-probabilities."""
        seq_reduce = (
            "mean" if self.config.dpo_length_norm else self.config.dpo_seq_reduce
        )
        return compute_block_scores(
            logits=logits,
            targets=batch.original_ids,
            masked_indices=batch.masked_indices,
            p_mask=batch.p_mask,
            logical_positions=batch.logical_positions,
            cum_seqlens=batch.cum_seqlens,
            block_size=self.config.block_size,
            weighting_scheme=self.config.loss_weighting_scheme,
            block_reduce=self.config.dpo_block_reduce,
            seq_reduce=seq_reduce,
            eps=self.config.mask_eps,
        )

    def _compute_ar_loss(
        self, logits: torch.Tensor, packed_labels: torch.Tensor, batch: WeDLMBatch
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Extract x0 stream and compute AR loss."""
        device = logits.device
        bs = batch.base_cum_seqlens.numel() - 1

        x0_logits, x0_labels = [], []
        for si in range(bs):
            pst = batch.cum_seqlens[si].item()
            L = (batch.cum_seqlens[si + 1].item() - pst) // 2
            orig_st = batch.base_cum_seqlens[si].item()

            if L > 0:
                x0_logits.append(logits[pst : pst + L])
                x0_labels.append(packed_labels[orig_st : orig_st + L])

        if x0_logits:
            return compute_ar_loss(torch.cat(x0_logits), torch.cat(x0_labels))
        return torch.tensor(0.0, device=device), {}

    def _trim_completion_ids(
        self, completion_ids: torch.Tensor, max_new_tokens: int
    ) -> torch.Tensor:
        """Trim generated completion by max length and EOS/PAD boundaries."""
        trimmed = completion_ids[:max_new_tokens]

        stop_ids: List[int] = []
        if self.tokenizer.eos_token_id is not None:
            stop_ids.append(int(self.tokenizer.eos_token_id))
        if self.tokenizer.pad_token_id is not None:
            stop_ids.append(int(self.tokenizer.pad_token_id))

        cut = int(trimmed.numel())
        for stop_id in stop_ids:
            positions = (trimmed == stop_id).nonzero(as_tuple=True)[0]
            if positions.numel() > 0:
                cut = min(cut, int(positions[0].item()))

        return trimmed[:cut]
