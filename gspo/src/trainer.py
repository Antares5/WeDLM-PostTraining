# coding=utf-8
"""GSPO On-Policy Trainer.

Core training loop for Group Sample Policy Optimization on WeDLM.

Phase 1 (no grad):  generate G responses per prompt using current policy
Phase 2 (with grad): compute block scores S_θ(y|x), REINFORCE loss, backward
Phase 3 (periodic):   sync training weights → generator
"""

import os
import logging
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler

from gspo.src.config import GSPOConfig
from gspo.src.batch import WeDLMBatch, build_wedlm_batch_from_response
from gspo.src.model import wedlm_forward
from gspo.src.loss import compute_block_scores, compute_gspo_loss
from gspo.src.attention import check_backend_available, get_available_backend, get_attention_wrapper
from gspo.src.data import GSPOPromptDataset, gspo_collate_fn
from gspo.src.generator import BaseGenerator, MockGenerator, WeDLMGenerator

logger = logging.getLogger(__name__)

MASK_TOKEN_ID = 151665


class GSPOTrainer:
    """GSPO on-policy trainer for block diffusion language models.

    Usage:
        config = GSPOConfig.from_yaml("config.yaml")
        trainer = GSPOTrainer(config)
        trainer.train()
    """

    def __init__(self, config: GSPOConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_step = 0

        # ── Backend ──
        if not check_backend_available(config.attention_backend):
            config.attention_backend = get_available_backend()
        logger.info(f"Attention backend: {config.attention_backend}")

        # ── Tokenizer ──
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_path, trust_remote_code=config.trust_remote_code
        )
        # Get im_end token ID for padding
        _im_end = getattr(self.tokenizer, "im_end_id", None)
        if _im_end is None:
            _im_end = getattr(self.tokenizer, "eos_token_id", 151645)
        self.tokenizer.pad_token_id = _im_end if _im_end is not None else self.tokenizer.eos_token_id

        # ── Model ──
        self._init_model()

        # ── Attention wrapper ──
        head_dim = self.model.config.hidden_size // self.model.config.num_attention_heads
        self.attn_wrapper = get_attention_wrapper(
            config.attention_backend, head_dim, deterministic=False
        )

        # ── Generator ──
        self._init_generator()

        # ── Optimizer & scheduler ──
        self._init_optimizer()

        # ── Dataset ──
        self._init_dataloader()

    # ═══════════════════════════════════════════════════════════
    # Initialization helpers
    # ═══════════════════════════════════════════════════════════

    def _init_model(self):
        """Load the training model (HF format)."""
        logger.info(f"Loading model from {self.config.model_path}")
        model_kwargs = {
            "trust_remote_code": self.config.trust_remote_code,
            "torch_dtype": torch.bfloat16 if self.config.bf16 else torch.float32,
            "attn_implementation": "eager",
        }
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path, **model_kwargs
        )
        self.model.to(self.device)
        self.model.train()
        # Disable dropout for stable score estimation
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.eval()

    def _init_generator(self):
        """Create the generator (real WeDLM engine or mock)."""
        try:
            self.generator: BaseGenerator = WeDLMGenerator(
                model_path=self.config.model_path,
                block_size=self.config.gspo_kvcache_block_size,
                window_size=self.config.gspo_window_size,
                gpu_memory_utilization=0.3,
                max_num_seqs=16,
                max_model_len=self.config.max_seq_length,
            )
            logger.info("WeDLMGenerator initialized (real engine)")
        except Exception as e:
            logger.warning(f"WeDLMGenerator init failed ({e}), falling back to MockGenerator")
            self.generator = MockGenerator(fixed_length=64)

    def _init_optimizer(self):
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_groups = [
            {
                "params": [p for n, p in self.model.named_parameters()
                           if not any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters()
                           if any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = torch.optim.AdamW(
            optimizer_groups, lr=self.config.learning_rate
        )

        # Estimate total steps for scheduler
        steps_per_epoch = max(len(self.train_dataloader) if hasattr(self, 'train_dataloader') else 100, 1)
        num_update_steps = math.ceil(steps_per_epoch / self.config.gradient_accumulation_steps)
        self.num_training_steps = num_update_steps * self.config.num_train_epochs
        num_warmup = int(self.num_training_steps * self.config.warmup_ratio)

        self.lr_scheduler = get_scheduler(
            self.config.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup,
            num_training_steps=self.num_training_steps,
        )

        self.scaler = torch.amp.GradScaler("cuda") if self.config.bf16 else None

    def _init_dataloader(self):
        self.train_dataset = GSPOPromptDataset(
            data_path=self.config.train_data,
            tokenizer=self.tokenizer,
            max_prompt_length=self.config.max_seq_length,
        )
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.per_device_train_batch_size,
            shuffle=True,
            collate_fn=gspo_collate_fn,
            num_workers=0,
        )
        logger.info(f"GSPO dataloader: {len(self.train_dataset)} prompts, "
                     f"batch_size={self.config.per_device_train_batch_size}")

    # ═══════════════════════════════════════════════════════════
    # Reward function (plug-in point)
    # ═══════════════════════════════════════════════════════════

    @torch.no_grad()
    def _compute_rewards(
        self,
        prompts: List[List[int]],
        responses: List[List[int]],
    ) -> List[float]:
        """Compute reward for each (prompt, response) pair.

        Default implementation: simple length-based heuristic.
        Override this method to plug in a real reward model.

        Args:
            prompts: Tokenized prompt IDs.
            responses: Generated completion token IDs.

        Returns:
            List of scalar reward values.
        """
        # Simple heuristic: prefer medium-length responses (20-200 tokens)
        rewards = []
        for resp in responses:
            L = len(resp)
            if L < 5:
                r = -1.0
            elif L < 20:
                r = 0.0
            elif L <= 200:
                r = 1.0
            else:
                r = max(0.0, 1.0 - 0.01 * (L - 200))
            rewards.append(r)
        return rewards

    # ═══════════════════════════════════════════════════════════
    # Score computation (with grad)
    # ═══════════════════════════════════════════════════════════

    def _score_response(
        self,
        response_ids: torch.Tensor,
        prompt_len: int,
    ) -> torch.Tensor:
        """Compute block-level score S_θ(y|x) for a single response.

        K Monte Carlo masking samples are averaged to reduce variance.

        Args:
            response_ids: [L] tensor = prompt_ids + completion_ids.
            prompt_len: Number of prompt tokens (these are never masked).

        Returns:
            Scalar score tensor (retains grad).
        """
        K = max(int(self.config.gspo_num_mask_samples), 1)
        score_sum = torch.tensor(0.0, device=self.device)

        for _ in range(K):
            batch = build_wedlm_batch_from_response(
                response_ids=response_ids,
                prompt_len=prompt_len,
                block_size=self.config.block_size,
                mask_token_id=MASK_TOKEN_ID,
                backend=self.config.attention_backend,
                mask_per_block=self.config.mask_per_block,
                eps=self.config.mask_eps,
            )

            logits = wedlm_forward(
                self.model, batch, self.attn_wrapper, self.config.attention_backend
            )

            scores, _ = compute_block_scores(
                logits=logits,
                targets=batch.original_ids,
                masked_indices=batch.masked_indices,
                p_mask=batch.p_mask,
                logical_positions=batch.logical_positions,
                cum_seqlens=batch.cum_seqlens,
                block_size=self.config.block_size,
                weighting_scheme=self.config.loss_weighting_scheme,
                block_reduce="mean",
                seq_reduce="mean",
                eps=self.config.mask_eps,
            )
            # scores is [1] for a single response
            score_sum = score_sum + scores[0] / K

        return score_sum

    # ═══════════════════════════════════════════════════════════
    # Single training step
    # ═══════════════════════════════════════════════════════════

    def train_step(
        self,
        prompts_batch: List[List[int]],
        prompt_texts: List[str],
    ) -> Dict[str, float]:
        """Execute one full GSPO training step.

        Phase 1: Generate G responses per prompt (no grad).
        Phase 2: Compute rewards, block scores, REINFORCE loss (with grad).
        Phase 3: Backward + optimizer step.
        """
        G = self.config.gspo_group_size
        B = len(prompts_batch)
        device = self.device

        # ═══════════════ Phase 1: Generation ═══════════════
        # Expand each prompt G times
        all_prompts = []
        all_prompt_indices = []
        for b in range(B):
            for g in range(G):
                all_prompts.append(prompts_batch[b])
                all_prompt_indices.append(b)

        with torch.no_grad():
            from wedlm.sampling_params import SamplingParams
            sp = SamplingParams(
                temperature=self.config.gspo_temperature,
                max_tokens=self.config.gspo_max_new_tokens,
                top_p=0.95,
                top_k=50,
                wedlm_entropy_threshold=self.config.gspo_entropy_threshold,
                wedlm_pos_penalty_factor=self.config.gspo_pos_penalty_factor,
            )
            all_responses = self.generator.generate(all_prompts, sp)

        # Release generator memory before heavy training computation
        self.generator.release_memory()

        # Compute rewards
        all_rewards = self._compute_rewards(all_prompts, all_responses)

        # ═══════════════ Phase 2: Score + Loss ═══════════════
        N = len(all_responses)
        all_scores = []
        prompt_idx_t = torch.empty(N, dtype=torch.long, device=device)

        for i in range(N):
            prompt_ids = all_prompts[i]
            response_ids_only = all_responses[i]
            # Full sequence = prompt + completion
            full_ids = prompt_ids + response_ids_only
            full_t = torch.tensor(full_ids, dtype=torch.long, device=device)
            prompt_len = len(prompt_ids)

            score_i = self._score_response(full_t, prompt_len)
            all_scores.append(score_i)
            prompt_idx_t[i] = all_prompt_indices[i]

        scores_t = torch.stack(all_scores)
        rewards_t = torch.tensor(all_rewards, device=device, dtype=scores_t.dtype)

        loss, logs = compute_gspo_loss(scores_t, rewards_t, prompt_idx_t)

        # ═══════════════ Phase 3: Backward ═══════════════
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            if loss.requires_grad:
                loss.backward()

        # Convert tensor logs to float
        log_dict = {k: float(v.detach().cpu()) if isinstance(v, torch.Tensor) else float(v)
                    for k, v in logs.items()}
        log_dict["loss"] = float(loss.detach().cpu())
        log_dict["lr"] = self.lr_scheduler.get_last_lr()[0]

        return log_dict

    # ═══════════════════════════════════════════════════════════
    # Weight sync (training model → generator)
    # ═══════════════════════════════════════════════════════════

    def _sync_weights_to_generator(self):
        """Save training model to temp dir, reload into generator."""
        import tempfile
        sync_dir = tempfile.mkdtemp(prefix="gspo_sync_")
        logger.info(f"Syncing weights → generator (step {self.global_step})")
        self.model.save_pretrained(sync_dir)
        self.tokenizer.save_pretrained(sync_dir)
        self.generator.update_weights(sync_dir)
        # Clean up temp dir is handled by generator.release_memory → engine.exit

    # ═══════════════════════════════════════════════════════════
    # Main training loop
    # ═══════════════════════════════════════════════════════════

    def train(self):
        """Main GSPO training loop."""
        logger.info(f"Starting GSPO training: {self.num_training_steps} update steps")

        for epoch in range(self.config.num_train_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.num_train_epochs}")

            for step, prompts_batch in enumerate(self.train_dataloader):
                # prompts_batch is a list of token-id lists
                prompt_texts = []  # can be filled from dataset if needed

                # GSPO step
                logs = self.train_step(prompts_batch, prompt_texts)

                # Gradient accumulation
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                    if self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

                    # Logging
                    if self.global_step % self.config.logging_steps == 0:
                        log_str = f"Step {self.global_step}: "
                        log_str += ", ".join(f"{k}={v:.4f}" for k, v in logs.items())
                        logger.info(log_str)

                    # Periodic weight sync
                    if self.global_step % self.config.gspo_sync_every_n_steps == 0:
                        self._sync_weights_to_generator()

                    # Save checkpoint
                    if self.global_step % self.config.save_steps == 0:
                        self._save_checkpoint()

            logger.info(f"Epoch {epoch + 1} complete")

        # Final save
        self._save_checkpoint(final=True)
        logger.info("GSPO training complete!")

    def _save_checkpoint(self, final: bool = False):
        save_path = os.path.join(
            self.config.output_dir,
            "final" if final else f"checkpoint-{self.global_step}",
        )
        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        logger.info(f"Saved checkpoint to {save_path}")
