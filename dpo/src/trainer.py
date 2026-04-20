# coding=utf-8
"""WeDLM Trainer for SFT training."""

import os
import math
import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler

from src.config import WeDLMTrainingConfig
from src.data import (
    WeDLMPackedDataset,
    WeDLMPairwiseDataset,
    WeDLMPromptDataset,
    packed_collate_fn,
    dpo_collate_fn,
    gspo_prompt_collate_fn,
    get_im_end_token_id,
)
from src.batch import WeDLMBatch, build_wedlm_batch
from src.model import wedlm_forward
from src.loss import compute_mlm_loss, compute_ar_loss, compute_block_scores, compute_gspo_loss
from src.attention import check_backend_available, get_available_backend, get_attention_wrapper
from src.reward import RewardInputs, BaseRewardFunction, build_reward_function

logger = logging.getLogger(__name__)

MASK_TOKEN_ID = 151665

# Lazy import wandb
_wandb = None

def _init_wandb(config: "WeDLMTrainingConfig", accelerator: Accelerator):
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
    
    import os
    if config.wandb_host:
        os.environ["WANDB_BASE_URL"] = config.wandb_host
    if config.wandb_key:
        os.environ["WANDB_API_KEY"] = config.wandb_key
    
    wandb.init(
        project=config.wandb_project or "wedlm-sft",
        entity=config.wandb_team,
        group=config.wandb_group,
        config={k: v for k, v in config.__dict__.items() if not k.startswith('_')},
    )
    return wandb


class WeDLMTrainer:
    """Trainer for WeDLM SFT/DPO/GSPO."""
    
    def __init__(self, config: WeDLMTrainingConfig, accelerator: Accelerator):
        self.config = config
        self.accelerator = accelerator
        self.wandb = _init_wandb(config, accelerator)
        self.gspo_reward_fn: Optional[BaseRewardFunction] = None
        self._setup()
        self._prepare_training()
    
    def _setup(self):
        """Initialize components."""
        if not check_backend_available(self.config.attention_backend):
            self.config.attention_backend = get_available_backend()
        logger.info(f"Attention backend: {self.config.attention_backend}")
        logger.info(f"Training mode: {self.config.training_mode}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path, trust_remote_code=self.config.trust_remote_code
        )
        self.im_end_token_id = get_im_end_token_id(self.tokenizer)
        self.tokenizer.pad_token_id = self.im_end_token_id
        
        model_kwargs = {
            "trust_remote_code": self.config.trust_remote_code,
            "torch_dtype": torch.bfloat16 if self.config.bf16 else torch.float32,
            "attn_implementation": "eager",
        }
        if self.config.use_deepspeed and self.config.deepspeed_zero_stage == 3:
            model_kwargs["low_cpu_mem_usage"] = True
        self.model = AutoModelForCausalLM.from_pretrained(self.config.model_path, **model_kwargs)
        self.ref_model = None
        if self.config.training_mode in ["dpo", "gspo"]:
            if self.config.training_mode == "gspo":
                ref_model_path = (
                    self.config.gspo_ref_model_path
                    or self.config.dpo_ref_model_path
                    or self.config.model_path
                )
            else:
                ref_model_path = self.config.dpo_ref_model_path or self.config.model_path

            self.ref_model = AutoModelForCausalLM.from_pretrained(ref_model_path, **model_kwargs)
            for param in self.ref_model.parameters():
                param.requires_grad = False

            self.ref_model.eval()
            logger.info(f"Loaded reference model from {ref_model_path}")
        
        head_dim = self.model.config.hidden_size // self.model.config.num_attention_heads
        # 训练时 deterministic=False 更快
        self.attn_wrapper = get_attention_wrapper(
            self.config.attention_backend, 
            head_dim,
            deterministic=False,
        )
        # 将 wrapper 移到正确的设备
        if hasattr(self.attn_wrapper, 'to'):
            self.attn_wrapper = self.attn_wrapper.to(self.accelerator.device)
        
        if self.config.training_mode == "dpo":
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
        elif self.config.training_mode == "gspo":
            gspo_data_path = self.config.gspo_train_data or self.config.dpo_train_data or self.config.train_data
            self.train_dataset = WeDLMPromptDataset(
                data_path=gspo_data_path,
                tokenizer=self.tokenizer,
                max_prompt_length=self.config.gspo_max_prompt_length,
            )
            if len(self.train_dataset) == 0:
                raise RuntimeError("No valid prompts found for GSPO training.")

            logger.info(f"Loaded GSPO prompt dataset from {gspo_data_path}")
            self.gspo_reward_fn = build_reward_function(self.config)
            logger.info(f"GSPO reward source: {self.config.gspo_reward_source}")
        else:
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
        
        if self.accelerator.num_processes > 1:
            self.train_sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=self.accelerator.num_processes,
                rank=self.accelerator.process_index,
                shuffle=True,
                seed=self.config.seed,
            )
            shuffle = False
            logger.info(f"Using DistributedSampler with {self.accelerator.num_processes} processes")
        else:
            self.train_sampler = None
            shuffle = True
        
        if self.config.training_mode == "dpo":
            train_batch_size = self.config.per_device_train_batch_size
            collate_fn = dpo_collate_fn
        elif self.config.training_mode == "gspo":
            train_batch_size = self.config.per_device_train_batch_size
            collate_fn = gspo_prompt_collate_fn
        else:
            train_batch_size = 1
            collate_fn = packed_collate_fn

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=train_batch_size,
            sampler=self.train_sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
        )
    
    def _prepare_training(self):
        """Prepare optimizer, scheduler, and accelerator."""
        steps_per_epoch = len(self.train_dataloader)
        
        num_update_steps_per_epoch = math.ceil(steps_per_epoch / self.config.gradient_accumulation_steps)
        self.num_training_steps = num_update_steps_per_epoch * self.config.num_train_epochs
        num_warmup_steps = int(self.num_training_steps * self.config.warmup_ratio)
        
        if self.accelerator.is_main_process:
            total_batches = len(self.train_dataset)
            logger.info(f"=== Training Configuration ===")
            logger.info(f"Number of GPUs: {self.accelerator.num_processes}")
            logger.info(f"Total batches in dataset: {total_batches}")
            logger.info(f"Batches per GPU per epoch: {steps_per_epoch}")
            logger.info(f"Gradient accumulation steps: {self.config.gradient_accumulation_steps}")
            logger.info(f"Update steps per epoch: {num_update_steps_per_epoch}")
            logger.info(f"Total training steps: {self.num_training_steps}")
            logger.info(f"Warmup steps: {num_warmup_steps}")
        
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_groups = [
            {"params": [p for n, p in self.model.named_parameters() 
                       if not any(nd in n for nd in no_decay) and p.requires_grad],
             "weight_decay": self.config.weight_decay},
            {"params": [p for n, p in self.model.named_parameters() 
                       if any(nd in n for nd in no_decay) and p.requires_grad],
             "weight_decay": 0.0},
        ]
        self.optimizer = torch.optim.AdamW(optimizer_groups, lr=self.config.learning_rate)
        
        self.lr_scheduler = get_scheduler(
            self.config.lr_scheduler_type, optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps, num_training_steps=self.num_training_steps
        )
        
        self.model, self.optimizer, self.train_dataloader, self.lr_scheduler = \
            self.accelerator.prepare(self.model, self.optimizer, self.train_dataloader, self.lr_scheduler)

        if self.config.training_mode in ["dpo", "gspo"] and self.ref_model is not None:
            try:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
            except Exception as err:
                logger.warning(
                    f"Failed to prepare reference model with Accelerator ({err}), fallback to .to(device)."
                )
                self.ref_model = self.ref_model.to(self.accelerator.device)

            self.ref_model.eval()
        
        self.global_step = 0
    
    def train_step_sft(self, batch: Dict[str, torch.Tensor]) -> tuple:
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
            wedlm_batch, self.attn_wrapper, self.config.attention_backend
        )
        
        mlm_loss, mlm_logs = compute_mlm_loss(
            logits, wedlm_batch.original_ids, wedlm_batch.masked_indices,
            wedlm_batch.p_mask, self.config.loss_weighting_scheme, self.config.mask_eps
        )
        
        ar_loss, ar_logs = torch.tensor(0.0, device=device), {}
        if self.config.enable_ar_loss and self.config.ar_loss_weight > 0:
            ar_loss, ar_logs = self._compute_ar_loss(logits, batch["packed_labels"].to(device), wedlm_batch)
        
        ar_w = self.config.ar_loss_weight if self.config.enable_ar_loss else 0.0
        total_loss = (mlm_loss + ar_w * ar_loss) / (1.0 + ar_w) if ar_w > 0 else mlm_loss
        
        return total_loss, {"loss": total_loss.detach(), **mlm_logs, **ar_logs}

    def _forward_wedlm_logits(self, model: torch.nn.Module, batch: WeDLMBatch) -> torch.Tensor:
        """Forward helper for WeDLM logits."""
        try:
            forward_model = self.accelerator.unwrap_model(model)
        except Exception:
            forward_model = model

        return wedlm_forward(forward_model, batch, self.attn_wrapper, self.config.attention_backend)

    def _compute_block_scores_for_batch(
        self, logits: torch.Tensor, batch: WeDLMBatch
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute sequence scores from masked block log-probabilities."""
        seq_reduce = "mean" if self.config.dpo_length_norm else self.config.dpo_seq_reduce
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

    def _trim_completion_ids(self, completion_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
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

    def _sample_gspo_online_candidates(
        self,
        prompt_input_ids: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, List[int], List[torch.Tensor]]:
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
                completion_ids = self._trim_completion_ids(completion_ids, max_completion_len)
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

            # Enforce minimum candidate count per prompt-group.
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
            return sampled_input_ids, sampled_labels, empty_groups, completion_lengths, sampled_completion_ids

        group_ids = torch.tensor(sampled_group_ids, dtype=torch.long, device=device)
        return sampled_input_ids, sampled_labels, group_ids, completion_lengths, sampled_completion_ids

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
        packed_cum_seqlens = torch.tensor(cum_seqlens, dtype=torch.long, device=device)
        return packed_input_ids, packed_labels, packed_cum_seqlens, completion_token_counts

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
            self.tokenizer.decode(completion_ids.tolist(), skip_special_tokens=True)
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
                f"Reward shape {tuple(rewards.shape)} must match policy score shape {tuple(policy_scores.shape)}"
            )

        return rewards.to(device=policy_scores.device, dtype=policy_scores.dtype).detach()

    def train_step_gspo(self, batch: Dict[str, torch.Tensor]) -> tuple:
        """Single online GSPO training step."""
        if self.ref_model is None:
            raise RuntimeError("Reference model is not initialized for GSPO mode.")

        device = self.accelerator.device
        prompt_input_ids = batch["prompt_input_ids"]
        prompt_metadata = batch.get("prompt_metadata", [{} for _ in prompt_input_ids])

        sampled_input_ids, sampled_labels, group_ids, completion_lengths, sampled_completion_ids = self._sample_gspo_online_candidates(
            prompt_input_ids
        )

        if group_ids.numel() == 0:
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            return zero, {
                "loss": zero.detach(),
                "gspo/loss": zero.detach(),
                "gspo/skipped_batch": torch.tensor(1.0, device=device),
                "gspo/num_prompts": torch.tensor(float(len(prompt_input_ids)), device=device),
                "gspo/num_candidates": torch.tensor(0.0, device=device),
            }

        packed_input_ids, packed_labels, packed_cum_seqlens, completion_token_counts = self._pack_gspo_candidates(
            sampled_input_ids, sampled_labels
        )

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
        policy_scores, score_logs = self._compute_block_scores_for_batch(policy_logits, wedlm_batch)

        with torch.no_grad():
            reference_logits = self._forward_wedlm_logits(self.ref_model, wedlm_batch)
            reference_scores, _ = self._compute_block_scores_for_batch(reference_logits, wedlm_batch)

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
            "gspo/num_prompts": torch.tensor(float(len(prompt_input_ids)), device=device),
            "gspo/num_candidates": torch.tensor(float(group_ids.numel()), device=device),
            "gspo/avg_completion_tokens": torch.tensor(
                float(sum(completion_lengths)) / max(len(completion_lengths), 1),
                device=device,
            ),
            "gspo/avg_supervised_tokens": torch.tensor(
                float(sum(completion_token_counts)) / max(len(completion_token_counts), 1),
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

    def train_step_dpo(self, batch: Dict[str, torch.Tensor]) -> tuple:
        """Single DPO training step using a low-memory backward strategy.

        To avoid OOM on long chosen/rejected pairs, this method:
        1) runs no-grad passes to estimate DPO coefficients;
        2) runs chosen and rejected gradient passes separately and backward immediately.
        """
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
                policy_chosen_logits_ng = self._forward_wedlm_logits(self.model, chosen_batch)
                policy_chosen_scores_ng, chosen_logs = self._compute_block_scores_for_batch(
                    policy_chosen_logits_ng, chosen_batch
                )
                del policy_chosen_logits_ng

                policy_rejected_logits_ng = self._forward_wedlm_logits(self.model, rejected_batch)
                policy_rejected_scores_ng, rejected_logs = self._compute_block_scores_for_batch(
                    policy_rejected_logits_ng, rejected_batch
                )
                del policy_rejected_logits_ng

                reference_chosen_logits = self._forward_wedlm_logits(self.ref_model, chosen_batch)
                reference_chosen_scores, _ = self._compute_block_scores_for_batch(reference_chosen_logits, chosen_batch)
                del reference_chosen_logits

                reference_rejected_logits = self._forward_wedlm_logits(self.ref_model, rejected_batch)
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

                # dL/ds_chosen = beta * (sigmoid(z) - 1) / batch_size
                batch_size = max(int(z.numel()), 1)
                coeff_chosen = beta * (torch.sigmoid(z) - 1.0) / float(batch_size)
                coeff_rejected = -coeff_chosen

            # Chosen branch backward (policy only)
            policy_chosen_logits = self._forward_wedlm_logits(self.model, chosen_batch)
            policy_chosen_scores, _ = self._compute_block_scores_for_batch(policy_chosen_logits, chosen_batch)
            chosen_objective = (
                (coeff_chosen.detach() * policy_chosen_scores).sum() * sample_scale * dpo_mix
            )

            ar_logs: Dict[str, torch.Tensor] = {}
            if ar_mix > 0:
                ar_loss, ar_logs = self._compute_ar_loss(policy_chosen_logits, chosen_labels, chosen_batch)
                ar_loss_sum = ar_loss_sum + ar_loss.detach()
                chosen_objective = chosen_objective + (ar_loss * sample_scale * ar_mix)

            self.accelerator.backward(chosen_objective)
            del policy_chosen_logits, policy_chosen_scores, chosen_objective

            # Rejected branch backward (policy only)
            policy_rejected_logits = self._forward_wedlm_logits(self.model, rejected_batch)
            policy_rejected_scores, _ = self._compute_block_scores_for_batch(policy_rejected_logits, rejected_batch)
            rejected_objective = (
                (coeff_rejected.detach() * policy_rejected_scores).sum() * sample_scale * dpo_mix
            )
            self.accelerator.backward(rejected_objective)
            del policy_rejected_logits, policy_rejected_scores, rejected_objective

            sample_logs: Dict[str, torch.Tensor] = {
                "dpo/policy_chosen_score": policy_chosen_scores_ng.mean().detach(),
                "dpo/policy_rejected_score": policy_rejected_scores_ng.mean().detach(),
                "dpo/reference_chosen_score": reference_chosen_scores.mean().detach(),
                "dpo/reference_rejected_score": reference_rejected_scores.mean().detach(),
                "dpo/policy_margin": (policy_chosen_scores_ng - policy_rejected_scores_ng).mean().detach(),
                "dpo/reference_margin": (reference_chosen_scores - reference_rejected_scores).mean().detach(),
                "dpo/chosen_num_masked_tokens": chosen_logs["score/num_masked_tokens"],
                "dpo/chosen_num_blocks": chosen_logs["score/num_blocks"],
                "dpo/rejected_num_masked_tokens": rejected_logs["score/num_masked_tokens"],
                "dpo/rejected_num_blocks": rejected_logs["score/num_blocks"],
                "dpo/loss": dpo_loss.detach(),
                "dpo/rewards_chosen": (beta * (policy_chosen_scores_ng - reference_chosen_scores)).mean().detach(),
                "dpo/rewards_rejected": (beta * (policy_rejected_scores_ng - reference_rejected_scores)).mean().detach(),
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

                total_logs[key] = total_logs.get(key, torch.zeros_like(value)) + value.detach()

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

    def train_step(self, batch: Dict[str, torch.Tensor]) -> tuple:
        """Dispatch training step by mode."""
        if self.config.training_mode == "dpo":
            return self.train_step_dpo(batch)

        if self.config.training_mode == "gspo":
            return self.train_step_gspo(batch)

        return self.train_step_sft(batch)
    
    def _compute_ar_loss(self, logits, packed_labels, batch: WeDLMBatch):
        """Extract x0 stream and compute AR loss."""
        device = logits.device
        bs = batch.base_cum_seqlens.numel() - 1
        
        x0_logits, x0_labels = [], []
        for si in range(bs):
            pst = batch.cum_seqlens[si].item()
            L = (batch.cum_seqlens[si + 1].item() - pst) // 2
            orig_st = batch.base_cum_seqlens[si].item()
            
            if L > 0:
                x0_logits.append(logits[pst:pst + L])
                x0_labels.append(packed_labels[orig_st:orig_st + L])
        
        if x0_logits:
            return compute_ar_loss(torch.cat(x0_logits), torch.cat(x0_labels))
        return torch.tensor(0.0, device=device), {}
    
    def train(self):
        """Main training loop."""
        logger.info(f"Starting training: {len(self.train_dataloader)} batches per GPU, {self.num_training_steps} total update steps")
        
        progress_bar = tqdm(total=self.num_training_steps, disable=not self.accelerator.is_local_main_process)
        
        for epoch in range(self.config.num_train_epochs):
            if self.config.training_mode == "dpo":
                # Disable dropout for deterministic two-phase DPO gradient estimation.
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
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                
                if self.accelerator.sync_gradients:
                    self.global_step += 1
                    progress_bar.update(1)
                    progress_bar.set_postfix(loss=f"{logs['loss'].item():.4f}")
                    
                    if self.global_step % self.config.logging_steps == 0:
                        self._log_metrics(logs, epoch)
                    if self.global_step % self.config.save_steps == 0:
                        self._save_checkpoint()
        
        progress_bar.close()
        self._save_checkpoint(final=True)
        if self.wandb:
            self.wandb.finish()
        logger.info("Training complete!")
    
    def _log_metrics(self, logs, epoch):
        if self.accelerator.is_main_process:
            log_str = f"Epoch {epoch} Step {self.global_step}: "
            log_str += ", ".join(f"{k}={v.item():.4f}" for k, v in logs.items() if isinstance(v, torch.Tensor))
            logger.info(log_str)
            
            if self.wandb:
                self.wandb.log({k: v.item() if hasattr(v, 'item') else v for k, v in logs.items()}, step=self.global_step)
    
    def _save_checkpoint(self, final=False):
        self.accelerator.wait_for_everyone()
        save_path = os.path.join(self.config.output_dir, "final" if final else f"checkpoint-{self.global_step}")
        
        if self.accelerator.is_main_process:
            os.makedirs(save_path, exist_ok=True)
            self.accelerator.unwrap_model(self.model).save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            logger.info(f"Saved checkpoint to {save_path}")

