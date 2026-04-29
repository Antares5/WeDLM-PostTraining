# coding=utf-8
"""DPO Trainer — Direct Preference Optimization.
Refactored from dpo/src/trainer.py (DPO branch)."""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from .base import BaseTrainer, MASK_TOKEN_ID
from .sft_trainer import SFTTrainer
from ..batch import build_wedlm_batch
from ..data import WeDLMPairwiseDataset, dpo_collate_fn

logger = logging.getLogger(__name__)


class DPOTrainer(SFTTrainer):
    """Trainer for WeDLM block-level DPO training.

    Uses low-memory backward strategy: runs no-grad passes for DPO coefficients,
    then runs chosen/rejected gradient passes separately.
    """

    def _setup(self):
        """Initialize DPO dataset, reference model, and dataloader."""
        # Load reference model
        ref_model_path = self.config.dpo_ref_model_path or self.config.model_path
        model_kwargs = {
            "trust_remote_code": self.config.trust_remote_code,
            "torch_dtype": torch.bfloat16 if self.config.bf16 else torch.float32,
            "attn_implementation": "eager",
        }
        if self.config.use_deepspeed and self.config.deepspeed_zero_stage == 3:
            model_kwargs["low_cpu_mem_usage"] = True
        self.ref_model = type(self.model).from_pretrained(
            ref_model_path, **model_kwargs
        )
        for param in self.ref_model.parameters():
            param.requires_grad = False
        self.ref_model.eval()
        logger.info(f"Loaded reference model from {ref_model_path}")

        # Load pairwise dataset
        dpo_data_path = self.config.dpo_train_data or self.config.train_data
        self.train_dataset = WeDLMPairwiseDataset(
            data_path=dpo_data_path,
            tokenizer=self.tokenizer,
            max_seq_length=self.config.max_seq_length,
            num_learnable_im_end=self.config.num_learnable_im_end,
        )
        if len(self.train_dataset) == 0:
            raise RuntimeError("No valid pairwise samples found for DPO training.")
        logger.info(f"Loaded DPO pairwise dataset from {dpo_data_path}")

        self._init_dataloader(
            dataset=self.train_dataset,
            batch_size=self.config.per_device_train_batch_size,
            collate_fn=dpo_collate_fn,
        )

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        """Single DPO training step with low-memory backward."""
        if self.ref_model is None:
            raise RuntimeError("Reference model is not initialized for DPO mode.")

        device = self.accelerator.device
        num_mask_samples = max(int(self.config.dpo_num_mask_samples), 1)
        sample_scale = 1.0 / float(num_mask_samples)

        ar_w = self.config.ar_loss_weight if self.config.enable_ar_loss else 0.0
        if ar_w < 0:
            raise ValueError("ar_loss_weight must be non-negative")

        dpo_mix = 1.0 / (1.0 + ar_w) if ar_w > 0 else 1.0
        ar_mix = ar_w / (1.0 + ar_w) if ar_w > 0 else 0.0

        chosen_input_ids = batch["chosen_packed_input_ids"].to(device)
        chosen_labels = batch["chosen_packed_labels"].to(device)
        chosen_cum_seqlens = batch["chosen_cum_seqlens"].to(device)

        rejected_input_ids = batch["rejected_packed_input_ids"].to(device)
        rejected_labels = batch["rejected_packed_labels"].to(device)
        rejected_cum_seqlens = batch["rejected_cum_seqlens"].to(device)

        dpo_loss_sum = torch.tensor(0.0, device=device)
        ar_loss_sum = torch.tensor(0.0, device=device)
        total_logs: Dict[str, torch.Tensor] = {}

        for _ in range(num_mask_samples):
            chosen_batch = build_wedlm_batch(
                packed_input_ids=chosen_input_ids,
                packed_labels=chosen_labels,
                cum_seqlens=chosen_cum_seqlens,
                block_size=self.config.block_size,
                mask_token_id=MASK_TOKEN_ID,
                mask_per_block=self.config.mask_per_block,
                backend=self.config.attention_backend,
                eps=self.config.mask_eps,
            )
            rejected_batch = build_wedlm_batch(
                packed_input_ids=rejected_input_ids,
                packed_labels=rejected_labels,
                cum_seqlens=rejected_cum_seqlens,
                block_size=self.config.block_size,
                mask_token_id=MASK_TOKEN_ID,
                mask_per_block=self.config.mask_per_block,
                backend=self.config.attention_backend,
                eps=self.config.mask_eps,
            )

            with torch.no_grad():
                policy_chosen_logits_ng = self._forward_wedlm_logits(
                    self.model, chosen_batch
                )
                policy_chosen_scores_ng, chosen_logs = (
                    self._compute_block_scores_for_batch(
                        policy_chosen_logits_ng, chosen_batch
                    )
                )
                del policy_chosen_logits_ng

                policy_rejected_logits_ng = self._forward_wedlm_logits(
                    self.model, rejected_batch
                )
                policy_rejected_scores_ng, rejected_logs = (
                    self._compute_block_scores_for_batch(
                        policy_rejected_logits_ng, rejected_batch
                    )
                )
                del policy_rejected_logits_ng

                reference_chosen_logits = self._forward_wedlm_logits(
                    self.ref_model, chosen_batch
                )
                reference_chosen_scores, _ = self._compute_block_scores_for_batch(
                    reference_chosen_logits, chosen_batch
                )
                del reference_chosen_logits

                reference_rejected_logits = self._forward_wedlm_logits(
                    self.ref_model, rejected_batch
                )
                reference_rejected_scores, _ = self._compute_block_scores_for_batch(
                    reference_rejected_logits, rejected_batch
                )
                del reference_rejected_logits

                beta = float(self.config.dpo_beta)
                z = beta * (
                    (policy_chosen_scores_ng - policy_rejected_scores_ng)
                    - (reference_chosen_scores - reference_rejected_scores)
                )
                dpo_loss = -F.logsigmoid(z).mean()
                dpo_loss_sum = dpo_loss_sum + dpo_loss.detach()

                batch_size_val = max(int(z.numel()), 1)
                coeff_chosen = (
                    beta * (torch.sigmoid(z) - 1.0) / float(batch_size_val)
                )
                coeff_rejected = -coeff_chosen

            # Chosen branch backward (policy only)
            policy_chosen_logits = self._forward_wedlm_logits(
                self.model, chosen_batch
            )
            policy_chosen_scores, _ = self._compute_block_scores_for_batch(
                policy_chosen_logits, chosen_batch
            )
            chosen_objective = (
                (coeff_chosen.detach() * policy_chosen_scores).sum()
                * sample_scale
                * dpo_mix
            )

            ar_logs: Dict[str, torch.Tensor] = {}
            if ar_mix > 0:
                ar_loss, ar_logs = self._compute_ar_loss(
                    policy_chosen_logits, chosen_labels, chosen_batch
                )
                ar_loss_sum = ar_loss_sum + ar_loss.detach()
                chosen_objective = (
                    chosen_objective + (ar_loss * sample_scale * ar_mix)
                )

            self.accelerator.backward(chosen_objective)
            del policy_chosen_logits, policy_chosen_scores, chosen_objective

            # Rejected branch backward (policy only)
            policy_rejected_logits = self._forward_wedlm_logits(
                self.model, rejected_batch
            )
            policy_rejected_scores, _ = self._compute_block_scores_for_batch(
                policy_rejected_logits, rejected_batch
            )
            rejected_objective = (
                (coeff_rejected.detach() * policy_rejected_scores).sum()
                * sample_scale
                * dpo_mix
            )
            self.accelerator.backward(rejected_objective)
            del policy_rejected_logits, policy_rejected_scores, rejected_objective

            sample_logs: Dict[str, torch.Tensor] = {
                "dpo/policy_chosen_score": policy_chosen_scores_ng.mean().detach(),
                "dpo/policy_rejected_score": policy_rejected_scores_ng.mean().detach(),
                "dpo/reference_chosen_score": reference_chosen_scores.mean().detach(),
                "dpo/reference_rejected_score": reference_rejected_scores.mean().detach(),
                "dpo/policy_margin": (
                    policy_chosen_scores_ng - policy_rejected_scores_ng
                )
                .mean()
                .detach(),
                "dpo/reference_margin": (
                    reference_chosen_scores - reference_rejected_scores
                )
                .mean()
                .detach(),
                "dpo/chosen_num_masked_tokens": chosen_logs[
                    "score/num_masked_tokens"
                ],
                "dpo/chosen_num_blocks": chosen_logs["score/num_blocks"],
                "dpo/rejected_num_masked_tokens": rejected_logs[
                    "score/num_masked_tokens"
                ],
                "dpo/rejected_num_blocks": rejected_logs["score/num_blocks"],
                "dpo/loss": dpo_loss.detach(),
                "dpo/rewards_chosen": (
                    beta
                    * (policy_chosen_scores_ng - reference_chosen_scores)
                )
                .mean()
                .detach(),
                "dpo/rewards_rejected": (
                    beta
                    * (policy_rejected_scores_ng - reference_rejected_scores)
                )
                .mean()
                .detach(),
                "dpo/rewards_margin": (
                    beta
                    * (
                        (policy_chosen_scores_ng - reference_chosen_scores)
                        - (policy_rejected_scores_ng - reference_rejected_scores)
                    )
                )
                .mean()
                .detach(),
                "dpo/rewards_accuracy": (
                    (
                        (policy_chosen_scores_ng - reference_chosen_scores)
                        > (policy_rejected_scores_ng - reference_rejected_scores)
                    )
                    .float()
                    .mean()
                    .detach()
                ),
                "dpo/logits": z.mean().detach(),
            }

            if ar_mix > 0 and ar_logs:
                sample_logs["ar/loss"] = ar_logs["ar/loss"]
                sample_logs["ar/num_tokens"] = ar_logs["ar/num_tokens"]

            for key, value in sample_logs.items():
                if not isinstance(value, torch.Tensor):
                    value = torch.tensor(float(value), device=device)

                total_logs[key] = (
                    total_logs.get(key, torch.zeros_like(value)) + value.detach()
                )

            del policy_chosen_scores_ng, policy_rejected_scores_ng
            del reference_chosen_scores, reference_rejected_scores, z, dpo_loss

        avg_dpo_loss = dpo_loss_sum / float(num_mask_samples)
        if ar_mix > 0:
            avg_ar_loss = ar_loss_sum / float(num_mask_samples)
            total_loss = dpo_mix * avg_dpo_loss + ar_mix * avg_ar_loss
        else:
            avg_ar_loss = torch.tensor(0.0, device=device)
            total_loss = avg_dpo_loss

        logs = {
            "loss": total_loss.detach(),
            "dpo/loss_raw": avg_dpo_loss.detach(),
        }
        if ar_mix > 0:
            logs["ar/loss_raw"] = avg_ar_loss.detach()

        denom = float(num_mask_samples)
        logs.update({key: value / denom for key, value in total_logs.items()})
        return total_loss, logs
