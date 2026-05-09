# coding=utf-8
"""GSPO Trainer for online Group-level RL training on WeDLM.

The trainer orchestrates:
  1. Online generation (G completions per prompt, no_grad)
  2. Reward computation (rule-based or model-based)
  3. Multi-model block scoring (policy, old, ref)
  4. GRPO clipped-surrogate loss + optional KL penalty
  5. Old-policy periodic synchronisation
"""

from __future__ import annotations

import os
import re
import math
import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler

from config import GSPOConfig
from data import GSPOPromptDataset, gspo_collate_fn
from generator import (
    get_mask_token_id,
    GenerationParams,
    wedlm_generate,
)
from scorer import ScorerConfig, compute_gspo_scores, _get_mask_token_id as _scorer_mask_token
from loss import compute_grpo_loss, compute_rewards
from src.attention import check_backend_available, get_available_backend, get_attention_wrapper

logger = logging.getLogger(__name__)

# Lazy wandb.
_wandb = None


def _init_wandb(config: GSPOConfig, accelerator: Accelerator):
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
        project=config.wandb_project or "wedlm-gspo",
        entity=config.wandb_team,
        group=config.wandb_group,
        config={k: v for k, v in config.__dict__.items() if not k.startswith('_')},
    )
    return wandb


class GSPOTrainer:
    """Online GSPO trainer for WeDLM."""

    def __init__(self, config: GSPOConfig, accelerator: Accelerator):
        self.config = config
        self.accelerator = accelerator
        self.wandb = _init_wandb(config, accelerator)
        self._setup()
        self._prepare_training()

    # ── Setup ────────────────────────────────────────────────────────
    def _setup(self):
        if not check_backend_available(self.config.attention_backend):
            self.config.attention_backend = get_available_backend()
        backend = self.config.attention_backend
        logger.info("Attention backend: %s", backend)
        logger.info("Training mode: %s", self.config.training_mode)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path, trust_remote_code=self.config.trust_remote_code,
        )
        from src.data import get_im_end_token_id
        self.im_end_token_id = get_im_end_token_id(self.tokenizer)
        self.tokenizer.pad_token_id = self.im_end_token_id

        model_kwargs = dict(
            trust_remote_code=self.config.trust_remote_code,
            torch_dtype=torch.bfloat16 if self.config.bf16 else torch.float32,
            attn_implementation="eager",
        )
        if self.config.use_deepspeed and self.config.deepspeed_zero_stage == 3:
            model_kwargs["low_cpu_mem_usage"] = True

        # Policy model (trained).
        self.model = AutoModelForCausalLM.from_pretrained(self.config.model_path, **model_kwargs)
        logger.info("Policy model loaded")

        # Old-policy snapshot (frozen, periodically synced).
        old_path = self.config.dpo_ref_model_path or self.config.model_path
        self.old_model = AutoModelForCausalLM.from_pretrained(old_path, **model_kwargs)
        for p in self.old_model.parameters():
            p.requires_grad = False
        self.old_model.eval()
        logger.info("Old-policy model loaded from %s", old_path)

        # Reference model for KL (frozen; same as old_model initially).
        self.ref_model = AutoModelForCausalLM.from_pretrained(old_path, **model_kwargs)
        for p in self.ref_model.parameters():
            p.requires_grad = False
        self.ref_model.eval()
        logger.info("Reference model loaded from %s", old_path)

        # Attention wrapper (shared across models).
        head_dim = self.model.config.hidden_size // self.model.config.num_attention_heads
        self.attn_wrapper = get_attention_wrapper(backend, head_dim, deterministic=False)
        if hasattr(self.attn_wrapper, "to"):
            self.attn_wrapper = self.attn_wrapper.to(self.accelerator.device)
        self.backend = backend

        # Read mask token id from model config (not hardcoded).
        self.mask_token_id = get_mask_token_id(self.model)
        logger.info("Mask token ID: %d (vocab size: %d)",
                     self.mask_token_id, self.model.config.vocab_size)

        # Dataset.
        self.train_dataset = GSPOPromptDataset(
            data_path=self.config.train_data,
            tokenizer=self.tokenizer,
            max_seq_length=self.config.max_seq_length,
            num_learnable_im_end=self.config.num_learnable_im_end,
        )
        if len(self.train_dataset) == 0:
            raise RuntimeError("No valid prompt samples found.")

        # DataLoader.
        if self.accelerator.num_processes > 1:
            self.train_sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=self.accelerator.num_processes,
                rank=self.accelerator.process_index,
                shuffle=True,
                seed=self.config.seed,
            )
            shuffle = False
        else:
            self.train_sampler = None
            shuffle = True

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.per_device_train_batch_size,
            sampler=self.train_sampler,
            shuffle=shuffle,
            collate_fn=gspo_collate_fn,
            num_workers=0,  # prompt-only; workers add minimal benefit
            pin_memory=True,
        )

    def _prepare_training(self):
        steps_per_epoch = len(self.train_dataloader)
        num_update_steps = math.ceil(steps_per_epoch / self.config.gradient_accumulation_steps)
        self.num_training_steps = num_update_steps * self.config.num_train_epochs
        num_warmup_steps = int(self.num_training_steps * self.config.warmup_ratio)

        if self.accelerator.is_main_process:
            logger.info("=== GSPO Training Configuration ===")
            logger.info("GPUs: %d", self.accelerator.num_processes)
            logger.info("Batches per GPU per epoch: %d", steps_per_epoch)
            logger.info("Gradient accumulation: %d", self.config.gradient_accumulation_steps)
            logger.info("Update steps per epoch: %d", num_update_steps)
            logger.info("Total update steps: %d", self.num_training_steps)
            logger.info("Warmup steps: %d", num_warmup_steps)
            logger.info("G=%d (group size)", self.config.gspo_group_size)
            logger.info("ε=%.2f (clip)", self.config.gspo_clip_epsilon)
            logger.info("Old-model sync every %d steps", self.config.gspo_old_model_update_steps)
            logger.info("KL β=%.4f (%s)", self.config.gspo_kl_beta, self.config.gspo_kl_estimator)

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
            self.config.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=self.num_training_steps,
        )

        # Prepare with accelerator (only policy + optim + scheduler).
        self.model, self.optimizer, self.train_dataloader, self.lr_scheduler = \
            self.accelerator.prepare(self.model, self.optimizer, self.train_dataloader, self.lr_scheduler)

        # Prepare old/ref models (evaluation mode).
        try:
            self.old_model = self.accelerator.prepare_model(self.old_model, evaluation_mode=True)
            self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
        except Exception as err:
            logger.warning("Failed to prepare old/ref with Accelerator (%s), fallback to .to(device)", err)
            self.old_model = self.old_model.to(self.accelerator.device)
            self.ref_model = self.ref_model.to(self.accelerator.device)

        self.global_step = 0

    # ── Old-policy sync ──────────────────────────────────────────────
    def _sync_old_model(self):
        """Copy policy weights to old_model."""
        try:
            policy_state = self.accelerator.unwrap_model(self.model).state_dict()
            self.accelerator.unwrap_model(self.old_model).load_state_dict(policy_state)
        except Exception:
            # Fallback for when unwrap_model isn't available.
            self.old_model.load_state_dict(self.model.state_dict())
        logger.info("Synced old_model ← policy at step %d", self.global_step)

    # ── Generation helpers ───────────────────────────────────────────
    def _make_gen_params(self) -> GenerationParams:
        return GenerationParams(
            max_tokens=self.config.gspo_gen_max_tokens,
            temperature=self.config.gspo_gen_temperature,
            block_size=self.config.block_size,
            window_size=self.config.gspo_gen_window_size,
            mask_token_id=self.mask_token_id,
            entropy_threshold=self.config.gspo_gen_entropy_threshold,
            pos_penalty_factor=self.config.gspo_gen_pos_penalty_factor,
        )

    @staticmethod
    def _make_system_prompt() -> str:
        return (
            "You are a math problem solver. "
            "For each question, think step by step carefully and explain your reasoning. "
            "Show your work and calculations. "
            "At the end, you MUST write the final answer on its own line "
            "in exactly this format: Answer: <your answer>"
        )

    # ── Training step ────────────────────────────────────────────────
    def train_step(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """One GSPO training step (online generation + scoring + GRPO loss).

        Args:
            batch: Dict with keys ``prompt_text``, ``prompt_ids``, ``ground_truth``,
                   each a list of length B.

        Returns:
            total_loss: scalar differentiable loss.
            logs: dict of detached scalar metrics.
        """
        device = self.accelerator.device
        B = len(batch["prompt_text"])
        G = self.config.gspo_group_size
        gen_params = self._make_gen_params()
        system_prompt = self._make_system_prompt()

        # Unwrapped model for generation & scoring (avoid accelerate wrapping issues).
        policy_unwrapped = self.accelerator.unwrap_model(self.model)
        old_unwrapped = self.accelerator.unwrap_model(self.old_model)
        ref_unwrapped = self.accelerator.unwrap_model(self.ref_model)

        eos_id = self.tokenizer.eos_token_id
        if eos_id is None:
            eos_id = self.im_end_token_id

        # ── Phase 1: Generate G completions per prompt (no_grad) ─────
        all_completion_ids: List[List[int]] = []    # B×G
        all_completion_texts: List[str] = []         # B×G
        all_ground_truths: List[str] = []            # B×G (repeated G times per prompt)

        for i in range(B):
            prompt_text = batch["prompt_text"][i]
            gt = batch["ground_truth"][i]
            for g in range(G):
                seed = self.global_step * B * G + i * G + g
                completion_ids, _ = wedlm_generate(
                    model=policy_unwrapped,
                    tokenizer=self.tokenizer,
                    prompt=prompt_text,
                    params=gen_params,
                    attn_wrapper=self.attn_wrapper,
                    backend=self.backend,
                    eos_token_id=eos_id,
                    seed=seed,
                    system_prompt=system_prompt,
                )
                comp_text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
                all_completion_ids.append(completion_ids)
                all_completion_texts.append(comp_text)
                all_ground_truths.append(gt)

        # Debug: log first completion every N steps to verify format.
        if self.global_step % self.config.logging_steps == 0 and all_completion_texts:
            sample_comp = all_completion_texts[0]
            sample_gt = all_ground_truths[0]
            # Head + tail: show beginning of reasoning and the final answer portion.
            head = sample_comp[:200]
            tail = sample_comp[-200:] if len(sample_comp) > 200 else ""
            logger.info("Sample completion HEAD: %r", head)
            if tail:
                logger.info("Sample completion TAIL: %r", tail)
            logger.info("Ground truth (raw): %r", sample_gt)
            # Show whether key answer markers are present.
            from loss import _extract_deepmath_answer, _strip_latex_delimiters
            pred = _extract_deepmath_answer(sample_comp)
            gt_stripped = _strip_latex_delimiters(sample_gt)
            has_answer = bool(re.findall(r"(?i)answer\s*:", sample_comp))
            has_boxed = bool(re.findall(r"\\boxed\{", sample_comp))
            has_hash = "####" in sample_comp
            logger.info(
                "Reward preview: extracted=%r  gt_stripped=%r  "
                "markers: answer=%s boxed=%s hash=%s",
                pred, gt_stripped, has_answer, has_boxed, has_hash,
            )

        # ── Phase 2: Compute rewards ──────────────────────────────────
        rewards = compute_rewards(
            all_completion_texts, all_ground_truths,
            reward_type=self.config.gspo_reward_type,
        ).to(device)

        # ── Phase 3: Compute block scores ─────────────────────────────
        # Build prompt_ids_list (B prompts, each repeated G times is handled
        # inside scorer).
        prompt_ids_list = batch["prompt_ids"]  # length B

        scorer_cfg = ScorerConfig(
            block_size=self.config.block_size,
            mask_token_id=self.mask_token_id,
            max_seq_length=self.config.max_seq_length,
            num_learnable_im_end=self.config.num_learnable_im_end,
            mask_per_block=self.config.mask_per_block,
            loss_weighting_scheme=self.config.loss_weighting_scheme,
            block_reduce=self.config.dpo_block_reduce,
            seq_reduce=self.config.dpo_seq_reduce,
            mask_eps=self.config.mask_eps,
            num_mask_samples=self.config.gspo_num_mask_samples,
        )

        s_policy, s_old, s_ref = compute_gspo_scores(
            policy_model=policy_unwrapped,
            old_model=old_unwrapped,
            ref_model=ref_unwrapped,
            prompt_ids_list=prompt_ids_list,
            completion_ids_list=all_completion_ids,
            tokenizer=self.tokenizer,
            scorer_config=scorer_cfg,
            attn_wrapper=self.attn_wrapper,
            backend=self.backend,
        )

        # ── Phase 4: GRPO loss ────────────────────────────────────────
        total_loss, loss_logs = compute_grpo_loss(
            s_policy=s_policy,
            s_old=s_old,
            s_ref=s_ref,
            rewards=rewards,
            group_size=G,
            clip_epsilon=self.config.gspo_clip_epsilon,
            kl_beta=self.config.gspo_kl_beta,
            kl_estimator=self.config.gspo_kl_estimator,
        )

        # ── Phase 5: Logging extras ───────────────────────────────────
        logs: Dict[str, torch.Tensor] = {}
        logs["loss"] = total_loss.detach()
        logs.update({k: v.detach() if isinstance(v, torch.Tensor) else torch.tensor(float(v), device=device)
                      for k, v in loss_logs.items()})

        # Generation stats.
        avg_comp_len = sum(len(c) for c in all_completion_ids) / max(len(all_completion_ids), 1)
        logs["gen/avg_comp_len"] = torch.tensor(avg_comp_len, device=device)
        logs["gen/num_completions"] = torch.tensor(len(all_completion_ids), device=device)

        return total_loss, logs

    # ── Training loop ────────────────────────────────────────────────
    def train(self):
        logger.info("Starting GSPO training: %d batches/GPU, %d update steps",
                     len(self.train_dataloader), self.num_training_steps)

        progress_bar = tqdm(total=self.num_training_steps,
                            disable=not self.accelerator.is_local_main_process)
        # Policy in eval mode (like DPO; dropout disabled for deterministic scoring).
        self.model.eval()

        for epoch in range(self.config.num_train_epochs):
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)

            for batch in self.train_dataloader:
                with self.accelerator.accumulate(self.model):
                    try:
                        loss, logs = self.train_step(batch)
                    except Exception as e:
                        logger.error("Error at step %d: %s", self.global_step, e, exc_info=True)
                        raise
                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.model.parameters(), self.config.max_grad_norm,
                        )

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
                    if self.global_step % self.config.gspo_old_model_update_steps == 0:
                        self._sync_old_model()

        progress_bar.close()
        # Final sync + save.
        self._sync_old_model()
        self._save_checkpoint(final=True)
        if self.wandb:
            self.wandb.finish()
        logger.info("GSPO training complete!")

    # ── Logging & Saving ─────────────────────────────────────────────
    def _log_metrics(self, logs, epoch):
        if self.accelerator.is_main_process:
            parts = [f"Epoch {epoch} Step {self.global_step}:"]
            parts += [f"{k}={v.item():.4f}" for k, v in logs.items()
                      if isinstance(v, torch.Tensor)]
            logger.info("  ".join(parts))
            if self.wandb:
                wb_dict = {k: v.item() if hasattr(v, "item") else v for k, v in logs.items()}
                self.wandb.log(wb_dict, step=self.global_step)

    def _save_checkpoint(self, final=False):
        self.accelerator.wait_for_everyone()
        tag = "final" if final else f"checkpoint-{self.global_step}"
        save_path = os.path.join(self.config.output_dir, tag)
        if self.accelerator.is_main_process:
            os.makedirs(save_path, exist_ok=True)
            self.accelerator.unwrap_model(self.model).save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            self.config.save_yaml(os.path.join(save_path, "training_config.yaml"))
            logger.info("Saved checkpoint to %s", save_path)
