# coding=utf-8
"""GSPO Trainer — Group Sequence Preference Optimization with online rollouts.
Refactored from dpo/src/trainer.py (GSPO branch)."""

import logging
from typing import Dict, List, Optional, Tuple

import torch

from .base import BaseTrainer, MASK_TOKEN_ID
from .sft_trainer import SFTTrainer
from ..batch import build_wedlm_batch
from ..loss import compute_gspo_loss
from ..reward import RewardInputs, BaseRewardFunction, build_reward_function
from ..data import WeDLMPromptDataset, gspo_prompt_collate_fn

logger = logging.getLogger(__name__)


class GSPOTrainer(SFTTrainer):
    """Trainer for WeDLM online GSPO training.

    Each step:
    1. Sample online candidates from the policy
    2. Compute policy & reference scores via masked block scoring
    3. Build rewards using the configured reward function
    4. Compute GSPO group-level loss
    """

    def _setup(self):
        """Initialize GSPO dataset, reference model, reward function, and dataloader."""
        # Reference model
        ref_model_path = (
            self.config.gspo_ref_model_path
            or self.config.dpo_ref_model_path
            or self.config.model_path
        )
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

        # Prompt dataset
        gspo_data_path = (
            self.config.gspo_train_data
            or self.config.dpo_train_data
            or self.config.train_data
        )
        self.train_dataset = WeDLMPromptDataset(
            data_path=gspo_data_path,
            tokenizer=self.tokenizer,
            max_prompt_length=self.config.gspo_max_prompt_length,
        )
        if len(self.train_dataset) == 0:
            raise RuntimeError("No valid prompts found for GSPO training.")
        logger.info(f"Loaded GSPO prompt dataset from {gspo_data_path}")

        # Reward function
        self.gspo_reward_fn: BaseRewardFunction = build_reward_function(self.config)
        logger.info(f"GSPO reward source: {self.config.gspo_reward_source}")

        self._init_dataloader(
            dataset=self.train_dataset,
            batch_size=self.config.per_device_train_batch_size,
            collate_fn=gspo_prompt_collate_fn,
        )

    # ── GSPO-specific methods ────────────────────────────────────

    def _sample_gspo_online_candidates(
        self,
        prompt_input_ids: List[torch.Tensor],
    ) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        torch.Tensor,
        List[int],
        List[torch.Tensor],
    ]:
        """Sample online candidate responses for GSPO groups."""
        device = self.accelerator.device
        policy_model = self.accelerator.unwrap_model(self.model)
        was_training = policy_model.training
        policy_model.eval()

        sampled_input_ids: List[torch.Tensor] = []
        sampled_labels: List[torch.Tensor] = []
        sampled_group_ids: List[int] = []
        completion_lengths: List[int] = []
        sampled_completion_ids: List[torch.Tensor] = []

        do_sample = self.config.gspo_rollout_temperature > 0.0
        temperature = self.config.gspo_rollout_temperature if do_sample else 1.0

        for group_idx, prompt_ids_cpu in enumerate(prompt_input_ids):
            prompt_ids = prompt_ids_cpu.to(device)
            prompt_len = int(prompt_ids.numel())
            max_completion_len = min(
                int(self.config.gspo_max_new_tokens),
                int(self.config.max_seq_length) - prompt_len,
            )
            if max_completion_len <= 0:
                continue

            group_start = len(sampled_group_ids)
            with torch.no_grad():
                generated = policy_model.generate(
                    input_ids=prompt_ids.unsqueeze(0),
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=self.config.gspo_rollout_top_p,
                    top_k=self.config.gspo_rollout_top_k,
                    num_return_sequences=self.config.gspo_group_size,
                    max_new_tokens=max_completion_len,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            for row in generated:
                completion_ids = row[prompt_len:]
                completion_ids = self._trim_completion_ids(
                    completion_ids, max_completion_len
                )
                if completion_ids.numel() == 0:
                    continue

                full_ids = torch.cat([prompt_ids, completion_ids], dim=0)
                labels = full_ids.clone()
                labels[:prompt_len] = -100

                if int((labels != -100).sum().item()) == 0:
                    continue

                sampled_input_ids.append(full_ids)
                sampled_labels.append(labels)
                sampled_group_ids.append(group_idx)
                completion_lengths.append(int(completion_ids.numel()))
                sampled_completion_ids.append(completion_ids.detach().cpu())

            # Enforce minimum candidate count per prompt-group
            valid_count = len(sampled_group_ids) - group_start
            if valid_count < self.config.gspo_min_candidates_per_group:
                while len(sampled_group_ids) > group_start:
                    sampled_group_ids.pop()
                    sampled_input_ids.pop()
                    sampled_labels.pop()
                    completion_lengths.pop()
                    sampled_completion_ids.pop()

        if was_training:
            policy_model.train()

        if len(sampled_group_ids) == 0:
            empty_groups = torch.empty((0,), dtype=torch.long, device=device)
            return (
                sampled_input_ids,
                sampled_labels,
                empty_groups,
                completion_lengths,
                sampled_completion_ids,
            )

        group_ids = torch.tensor(sampled_group_ids, dtype=torch.long, device=device)
        return (
            sampled_input_ids,
            sampled_labels,
            group_ids,
            completion_lengths,
            sampled_completion_ids,
        )

    def _pack_gspo_candidates(
        self,
        sampled_input_ids: List[torch.Tensor],
        sampled_labels: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        """Pack variable-length online GSPO candidates into one stream."""
        device = self.accelerator.device

        packed_ids_parts: List[torch.Tensor] = []
        packed_labels_parts: List[torch.Tensor] = []
        cum_seqlens = [0]
        completion_token_counts: List[int] = []

        for input_ids, labels in zip(sampled_input_ids, sampled_labels):
            packed_ids_parts.append(input_ids)
            packed_labels_parts.append(labels)
            cum_seqlens.append(cum_seqlens[-1] + int(input_ids.numel()))
            completion_token_counts.append(int((labels != -100).sum().item()))

        packed_input_ids = torch.cat(packed_ids_parts, dim=0).to(device)
        packed_labels = torch.cat(packed_labels_parts, dim=0).to(device)
        packed_cum_seqlens = torch.tensor(
            cum_seqlens, dtype=torch.long, device=device
        )
        return (
            packed_input_ids,
            packed_labels,
            packed_cum_seqlens,
            completion_token_counts,
        )

    def _compute_gspo_rewards(
        self,
        prompt_input_ids: List[torch.Tensor],
        prompt_metadata: List[Dict[str, object]],
        sampled_input_ids: List[torch.Tensor],
        sampled_labels: List[torch.Tensor],
        sampled_completion_ids: List[torch.Tensor],
        completion_lengths: List[int],
        group_ids: torch.Tensor,
        policy_scores: torch.Tensor,
        reference_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Compute GSPO rewards via a configurable reward interface."""
        if self.gspo_reward_fn is None:
            raise RuntimeError("GSPO reward function is not initialized.")

        candidate_completion_texts = [
            self.tokenizer.decode(
                completion_ids.tolist(), skip_special_tokens=True
            )
            for completion_ids in sampled_completion_ids
        ]

        reward_inputs = RewardInputs(
            prompt_input_ids=prompt_input_ids,
            candidate_input_ids=sampled_input_ids,
            candidate_labels=sampled_labels,
            completion_lengths=completion_lengths,
            group_ids=group_ids,
            policy_scores=policy_scores,
            reference_scores=reference_scores,
            tokenizer=self.tokenizer,
            prompt_metadata=prompt_metadata,
            candidate_completion_ids=sampled_completion_ids,
            candidate_completion_texts=candidate_completion_texts,
        )
        rewards = self.gspo_reward_fn(reward_inputs)
        if rewards.shape != policy_scores.shape:
            raise ValueError(
                f"Reward shape {tuple(rewards.shape)} must match policy score shape "
                f"{tuple(policy_scores.shape)}"
            )

        return rewards.to(
            device=policy_scores.device, dtype=policy_scores.dtype
        ).detach()

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        """Single online GSPO training step."""
        if self.ref_model is None:
            raise RuntimeError("Reference model is not initialized for GSPO mode.")

        device = self.accelerator.device
        prompt_input_ids = batch["prompt_input_ids"]
        prompt_metadata = batch.get(
            "prompt_metadata", [{} for _ in prompt_input_ids]
        )

        (
            sampled_input_ids,
            sampled_labels,
            group_ids,
            completion_lengths,
            sampled_completion_ids,
        ) = self._sample_gspo_online_candidates(prompt_input_ids)

        if group_ids.numel() == 0:
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            return zero, {
                "loss": zero.detach(),
                "gspo/loss": zero.detach(),
                "gspo/skipped_batch": torch.tensor(1.0, device=device),
                "gspo/num_prompts": torch.tensor(
                    float(len(prompt_input_ids)), device=device
                ),
                "gspo/num_candidates": torch.tensor(0.0, device=device),
            }

        (
            packed_input_ids,
            packed_labels,
            packed_cum_seqlens,
            completion_token_counts,
        ) = self._pack_gspo_candidates(sampled_input_ids, sampled_labels)

        wedlm_batch = build_wedlm_batch(
            packed_input_ids=packed_input_ids,
            packed_labels=packed_labels,
            cum_seqlens=packed_cum_seqlens,
            block_size=self.config.block_size,
            mask_token_id=MASK_TOKEN_ID,
            mask_per_block=self.config.mask_per_block,
            backend=self.config.attention_backend,
            eps=self.config.mask_eps,
        )

        policy_logits = self._forward_wedlm_logits(self.model, wedlm_batch)
        policy_scores, score_logs = self._compute_block_scores_for_batch(
            policy_logits, wedlm_batch
        )

        with torch.no_grad():
            reference_logits = self._forward_wedlm_logits(
                self.ref_model, wedlm_batch
            )
            reference_scores, _ = self._compute_block_scores_for_batch(
                reference_logits, wedlm_batch
            )

        rewards = self._compute_gspo_rewards(
            prompt_input_ids=prompt_input_ids,
            prompt_metadata=prompt_metadata,
            sampled_input_ids=sampled_input_ids,
            sampled_labels=sampled_labels,
            sampled_completion_ids=sampled_completion_ids,
            completion_lengths=completion_lengths,
            group_ids=group_ids,
            policy_scores=policy_scores,
            reference_scores=reference_scores,
        )

        gspo_loss, gspo_logs = compute_gspo_loss(
            policy_scores=policy_scores,
            reference_scores=reference_scores,
            group_ids=group_ids,
            rewards=rewards,
            score_temperature=float(self.config.gspo_score_temperature),
            reward_temperature=float(self.config.gspo_reward_temperature),
            ref_alpha=float(self.config.gspo_ref_alpha),
            kl_coef=float(self.config.gspo_kl_coef),
            eps=self.config.mask_eps,
        )

        logs = {
            "loss": gspo_loss.detach(),
            "gspo/num_prompts": torch.tensor(
                float(len(prompt_input_ids)), device=device
            ),
            "gspo/num_candidates": torch.tensor(
                float(group_ids.numel()), device=device
            ),
            "gspo/avg_completion_tokens": torch.tensor(
                float(sum(completion_lengths)) / max(len(completion_lengths), 1),
                device=device,
            ),
            "gspo/avg_supervised_tokens": torch.tensor(
                float(sum(completion_token_counts))
                / max(len(completion_token_counts), 1),
                device=device,
            ),
            "gspo/reward_min": rewards.min().detach(),
            "gspo/reward_max": rewards.max().detach(),
            "gspo/reward_std": rewards.std(unbiased=False).detach(),
            "gspo/skipped_batch": torch.tensor(0.0, device=device),
            **score_logs,
            **gspo_logs,
        }
        return gspo_loss, logs
