# coding=utf-8
"""SFT Trainer — supervised fine-tuning with WeDLM masking.
Refactored from dpo/src/trainer.py (SFT branch)."""

import os
import logging
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .base import BaseTrainer, MASK_TOKEN_ID
from ..batch import build_wedlm_batch
from ..model import wedlm_forward
from ..loss import compute_mlm_loss
from ..data import WeDLMPackedDataset, packed_collate_fn

logger = logging.getLogger(__name__)


class SFTTrainer(BaseTrainer):
    """Trainer for WeDLM MLM + AR SFT training."""

    def _setup(self):
        """Initialize SFT dataset and dataloader."""
        self.train_dataset = WeDLMPackedDataset(
            data_path=self.config.train_data,
            tokenizer=self.tokenizer,
            max_seq_length=self.config.max_seq_length,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            num_learnable_im_end=self.config.num_learnable_im_end,
            cache_dir=os.path.join(self.config.output_dir, ".packed_cache"),
            seed=self.config.seed,
            rebuild_cache=self.config.rebuild_cache,
        )

        self._init_dataloader(
            dataset=self.train_dataset,
            batch_size=1,
            collate_fn=packed_collate_fn,
        )

    def _init_dataloader(self, dataset, batch_size: int, collate_fn):
        """Shared dataloader initializer."""
        if self.accelerator.num_processes > 1:
            self.train_sampler = DistributedSampler(
                dataset,
                num_replicas=self.accelerator.num_processes,
                rank=self.accelerator.process_index,
                shuffle=True,
                seed=self.config.seed,
            )
            shuffle = False
            logger.info(
                f"Using DistributedSampler with {self.accelerator.num_processes} processes"
            )
        else:
            self.train_sampler = None
            shuffle = True

        self.train_dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=self.train_sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
        )

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        """Single SFT training step."""
        device = self.accelerator.device

        wedlm_batch = build_wedlm_batch(
            packed_input_ids=batch["packed_input_ids"].to(device),
            packed_labels=batch["packed_labels"].to(device),
            cum_seqlens=batch["cum_seqlens"].to(device),
            block_size=self.config.block_size,
            mask_token_id=MASK_TOKEN_ID,
            mask_per_block=self.config.mask_per_block,
            backend=self.config.attention_backend,
            eps=self.config.mask_eps,
        )

        logits = wedlm_forward(
            self.accelerator.unwrap_model(self.model),
            wedlm_batch,
            self.attn_wrapper,
            self.config.attention_backend,
        )

        mlm_loss, mlm_logs = compute_mlm_loss(
            logits,
            wedlm_batch.original_ids,
            wedlm_batch.masked_indices,
            wedlm_batch.p_mask,
            self.config.loss_weighting_scheme,
            self.config.mask_eps,
        )

        ar_loss, ar_logs = torch.tensor(0.0, device=device), {}
        if self.config.enable_ar_loss and self.config.ar_loss_weight > 0:
            ar_loss, ar_logs = self._compute_ar_loss(
                logits, batch["packed_labels"].to(device), wedlm_batch
            )

        ar_w = self.config.ar_loss_weight if self.config.enable_ar_loss else 0.0
        total_loss = (
            (mlm_loss + ar_w * ar_loss) / (1.0 + ar_w)
            if ar_w > 0
            else mlm_loss
        )

        return total_loss, {"loss": total_loss.detach(), **mlm_logs, **ar_logs}
