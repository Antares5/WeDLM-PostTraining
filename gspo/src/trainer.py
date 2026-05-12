# coding=utf-8
"""GSPO Trainer for on-policy group sampling policy optimization."""

import os
import math
import logging
from typing import Dict, List, Tuple, Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler

from src.config import GSPOTrainingConfig
from src.data import GSPOPromptDataset, GSPOCollateFunction, get_im_end_token_id
from src.batch import WeDLMBatch, build_wedlm_batch
from src.model import wedlm_forward
from src.loss import compute_ar_loss, compute_block_scores, compute_gspo_coefficients, compute_gspo_loss
from src.attention import check_backend_available, get_available_backend, get_attention_wrapper
from src.generator import WeDLMGenerator
from src.reward import MathReward

logger = logging.getLogger(__name__)

MASK_TOKEN_ID = 151665

_wandb = None


class _NoOpContext:
    """A trivial context manager (no-op) used when zero.Init is not needed."""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass


def _init_wandb(config: GSPOTrainingConfig, accelerator: Accelerator):
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
        project=config.wandb_project or "wedlm-gspo",
        entity=config.wandb_team,
        group=config.wandb_group,
        config={k: v for k, v in config.__dict__.items() if not k.startswith('_')},
    )
    return wandb


class GSPOTrainer:
    """Trainer for GSPO on-policy training with WeDLM block diffusion."""

    def __init__(self, config: GSPOTrainingConfig, accelerator: Accelerator):
        self.config = config
        self.accelerator = accelerator
        self.wandb = _init_wandb(config, accelerator)
        self._setup()
        self._prepare_training()

    def _setup(self):
        """Initialize model, tokenizer, dataset, generator, reward."""
        if not check_backend_available(self.config.attention_backend):
            self.config.attention_backend = get_available_backend()
        logger.info(f"Attention backend: {self.config.attention_backend}")
        logger.info(f"GSPO training mode: K={self.config.gspo_num_samples}, beta={self.config.gspo_beta}")

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path, trust_remote_code=self.config.trust_remote_code
        )
        self.im_end_token_id = get_im_end_token_id(self.tokenizer)
        self.tokenizer.pad_token_id = self.im_end_token_id

        # Model loading kwargs
        model_kwargs = {
            "trust_remote_code": self.config.trust_remote_code,
            "torch_dtype": torch.bfloat16 if self.config.bf16 else torch.float32,
            "attn_implementation": "eager",
        }
        if self.config.use_deepspeed and self.config.deepspeed_zero_stage == 3:
            model_kwargs["low_cpu_mem_usage"] = True
            import deepspeed
            self._ds_zero3_ctx = deepspeed.zero.Init()
        else:
            self._ds_zero3_ctx = None

        # Policy model
        with (self._ds_zero3_ctx or _NoOpContext()):
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path, **model_kwargs
            )

        # Reference model
        ref_model_path = self.config.gspo_ref_model_path or self.config.model_path
        logger.info(f"Loading reference model from {ref_model_path}")
        with (self._ds_zero3_ctx or _NoOpContext()):
            self.ref_model = AutoModelForCausalLM.from_pretrained(
                ref_model_path, **model_kwargs
            )
        for param in self.ref_model.parameters():
            param.requires_grad = False
        self.ref_model.eval()

        # Attention wrapper
        head_dim = self.model.config.hidden_size // self.model.config.num_attention_heads
        self.attn_wrapper = get_attention_wrapper(
            self.config.attention_backend,
            head_dim,
            deterministic=False,
        )
        if hasattr(self.attn_wrapper, 'to'):
            self.attn_wrapper = self.attn_wrapper.to(self.accelerator.device)

        # Dataset
        logger.info(f"Loading prompt data from {self.config.gspo_prompt_data}")
        self.train_dataset = GSPOPromptDataset(
            data_path=self.config.gspo_prompt_data,
            tokenizer=self.tokenizer,
            max_seq_length=self.config.max_seq_length,
            prompt_format=self.config.gspo_prompt_format,
            num_learnable_im_end=self.config.num_learnable_im_end,
        )
        if len(self.train_dataset) == 0:
            raise RuntimeError("No valid prompt samples found for GSPO training.")

        logger.info(f"Loaded {len(self.train_dataset)} prompt samples")

        # DataLoader
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

        collate_fn = GSPOCollateFunction(pad_token_id=self.im_end_token_id)

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.per_device_train_batch_size,
            sampler=self.train_sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=0,  # avoid issues with LLMEngine multiprocessing
            pin_memory=False,  # accelerate handles device placement; True risks
                               # CUDA-tensor pin crash if default device leaks
        )

        # Generator (initialized after model is on device, during _prepare_training)
        self.generator = None

        # Math reward
        self.math_reward = MathReward(
            reward_type=self.config.gspo_reward_type,
            tokenizer=self.tokenizer,
        )

    def _prepare_training(self):
        """Prepare optimizer, scheduler, generator, and accelerator."""
        steps_per_epoch = len(self.train_dataloader)
        num_update_steps_per_epoch = math.ceil(
            steps_per_epoch / self.config.gradient_accumulation_steps
        )
        self.num_training_steps = num_update_steps_per_epoch * self.config.num_train_epochs
        num_warmup_steps = int(self.num_training_steps * self.config.warmup_ratio)

        if self.accelerator.is_main_process:
            logger.info(f"=== GSPO Training Configuration ===")
            logger.info(f"Number of GPUs: {self.accelerator.num_processes}")
            logger.info(f"Batches per GPU per epoch: {steps_per_epoch}")
            logger.info(f"Gradient accumulation steps: {self.config.gradient_accumulation_steps}")
            logger.info(f"Update steps per epoch: {num_update_steps_per_epoch}")
            logger.info(f"Total training steps: {self.num_training_steps}")
            logger.info(f"Warmup steps: {num_warmup_steps}")

        # Optimizer
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_groups = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = torch.optim.AdamW(optimizer_groups, lr=self.config.learning_rate)

        # Scheduler
        self.lr_scheduler = get_scheduler(
            self.config.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=self.num_training_steps,
        )

        # Prepare with accelerator
        self.model, self.optimizer, self.train_dataloader, self.lr_scheduler = \
            self.accelerator.prepare(
                self.model, self.optimizer, self.train_dataloader, self.lr_scheduler
            )

        # Prepare ref model
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

        # Initialize generator with unwrapped model
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        generation_config = self.config.get_generation_config()
        self.generator = WeDLMGenerator(
            model=unwrapped_model,
            tokenizer=self.tokenizer,
            generation_config=generation_config,
            device=self.accelerator.device,
            model_path=self.config.model_path,
        )

        self.global_step = 0

    # ========== Forward helpers ==========

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
        seq_reduce = "mean" if self.config.gspo_length_norm else self.config.gspo_seq_reduce
        return compute_block_scores(
            logits=logits,
            targets=batch.original_ids,
            masked_indices=batch.masked_indices,
            p_mask=batch.p_mask,
            logical_positions=batch.logical_positions,
            cum_seqlens=batch.cum_seqlens,
            block_size=self.config.block_size,
            weighting_scheme=self.config.loss_weighting_scheme,
            block_reduce=self.config.gspo_block_reduce,
            seq_reduce=seq_reduce,
            eps=self.config.mask_eps,
        )

    def _build_wedlm_batch_for_response(
        self, input_ids: torch.Tensor, labels: torch.Tensor, device: torch.device
    ) -> WeDLMBatch:
        """Build a WeDLMBatch for a single response (bs=1)."""
        cum_seqlens = torch.tensor([0, input_ids.size(0)], dtype=torch.long, device=device)
        return build_wedlm_batch(
            packed_input_ids=input_ids.to(device),
            packed_labels=labels.to(device),
            cum_seqlens=cum_seqlens,
            block_size=self.config.block_size,
            mask_token_id=MASK_TOKEN_ID,
            mask_per_block=self.config.mask_per_block,
            backend=self.config.attention_backend,
            eps=self.config.mask_eps,
        )

    def _score_response(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Score a single response: build batch → forward → block scores."""
        wedlm_batch = self._build_wedlm_batch_for_response(input_ids, labels, device)
        logits = self._forward_wedlm_logits(model, wedlm_batch)
        scores, logs = self._compute_block_scores_for_batch(logits, wedlm_batch)
        return scores.squeeze(0), logs  # scalar score per response

    def _get_prompt_text(self, messages: List[Dict[str, str]]) -> str:
        """Format messages into prompt text for generation."""
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            text = ""
            for msg in messages:
                text += f"{msg.get('role', 'user')}: {msg.get('content', '')}\n"
            text += "assistant: "
            return text

    # ========== GSPO Training Step ==========

    def train_step_gspo(self, batch: Dict[str, Any]) -> tuple:
        """Single GSPO training step with 4-phase flow.

        Phases:
        1. Generation (no_grad): generate K responses per prompt
        2. Reward scoring (no_grad): math rule-based binary reward
        3. Reference scoring (no_grad): ref model block scores
        4. Policy scoring + backward (grad): two-pass per-branch backward
        """
        device = self.accelerator.device
        K = self.config.gspo_num_samples
        beta = float(self.config.gspo_beta)
        num_mask_samples = max(int(self.config.gspo_num_mask_samples), 1)
        sample_scale = 1.0 / float(num_mask_samples)

        # Extract batch data
        prompt_messages_list = batch["messages"]  # List[List[Dict]]
        ground_truths = batch["ground_truths"]  # List[str]
        bs = len(prompt_messages_list)

        # Get properly formatted prompt texts
        prompt_texts = [self._get_prompt_text(msgs) for msgs in prompt_messages_list]

        # ===== Phase 1: Generation (no_grad) =====
        self.model.eval()
        all_responses = []  # [(input_ids, labels, text) × K] per prompt

        with torch.no_grad():
            for prompt_text in prompt_texts:
                prompt_responses = []
                for k in range(K):
                    seed = self.config.seed + k
                    try:
                        response = self.generator.generate(
                            prompt_text, seed=seed
                        )
                        prompt_responses.append(response)
                    except Exception as e:
                        logger.warning(f"Generation failed for k={k}: {e}")
                        # Create empty fallback response
                        prompt_ids = self.tokenizer.encode(
                            prompt_text, add_special_tokens=False
                        )
                        prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long)
                        response = {
                            "input_ids": prompt_tensor,
                            "labels": torch.full_like(prompt_tensor, -100),
                            "text": "",
                        }
                        prompt_responses.append(response)
                all_responses.append(prompt_responses)

        self.model.train()

        # ===== Phase 2: Math Reward Scoring (no_grad) =====
        with torch.no_grad():
            rewards = []  # [bs, K]
            for prompt_text, responses, gt in zip(
                prompt_texts, all_responses, ground_truths
            ):
                response_texts = [r["text"] for r in responses]
                prompt_rewards = self.math_reward.compute_rewards(
                    [prompt_text] * K, response_texts, [gt] * K
                )
                # Normalize rewards within group
                if prompt_rewards.std() > 1e-8 and prompt_rewards.numel() > 1:
                    prompt_rewards = (prompt_rewards - prompt_rewards.mean()) / (
                        prompt_rewards.std() + 1e-8
                    )
                rewards.append(prompt_rewards)
            rewards = torch.stack(rewards, dim=0)  # [bs, K]

        # ===== Phase 3: Reference Scoring (no_grad) =====
        ref_scores = []  # [bs, K]
        with torch.no_grad():
            for responses in all_responses:
                prompt_ref_scores = []
                for resp in responses:
                    score, _ = self._score_response(
                        self.ref_model,
                        resp["input_ids"],
                        resp["labels"],
                        device,
                    )
                    prompt_ref_scores.append(score)
                ref_scores.append(torch.stack(prompt_ref_scores))
            ref_scores = torch.stack(ref_scores, dim=0)  # [bs, K]

        # ===== Phase 4: Policy Scoring + Backward (grad) =====
        total_dpo_loss = torch.tensor(0.0, device=device)
        total_logs: Dict[str, torch.Tensor] = {}

        for sample_idx in range(bs):
            sample_responses = all_responses[sample_idx]
            sample_rewards = rewards[sample_idx]  # [K]
            sample_ref = ref_scores[sample_idx]  # [K]

            for _ in range(num_mask_samples):
                # 4a. No-grad pass: get all K policy scores for coefficient computation
                with torch.no_grad():
                    policy_scores_ng = []
                    for resp in sample_responses:
                        score, _ = self._score_response(
                            self.model,
                            resp["input_ids"],
                            resp["labels"],
                            device,
                        )
                        policy_scores_ng.append(score)
                    policy_scores_ng = torch.stack(policy_scores_ng)  # [K]

                # 4b. Compute GSPO coefficients
                coeffs = compute_gspo_coefficients(
                    policy_scores_ng, sample_ref, sample_rewards, beta
                )  # [K]

                # 4c. Compute GSPO loss for logging
                _, gspo_logs = compute_gspo_loss(
                    policy_scores_ng, sample_ref, sample_rewards, beta
                )

                # 4d. Per-branch backward
                for k in range(K):
                    if abs(coeffs[k].item()) < 1e-10:
                        continue

                    resp = sample_responses[k]
                    policy_score, _ = self._score_response(
                        self.model,
                        resp["input_ids"],
                        resp["labels"],
                        device,
                    )
                    branch_loss = (
                        coeffs[k].detach() * policy_score * sample_scale
                    )
                    self.accelerator.backward(branch_loss)
                    del policy_score, branch_loss

                # Accumulate logs
                for key, value in gspo_logs.items():
                    if isinstance(value, torch.Tensor):
                        total_logs[key] = total_logs.get(
                            key, torch.tensor(0.0, device=device)
                        ) + value.detach()

        # Average logs over batch
        denom = float(bs * num_mask_samples)
        avg_logs = {key: value / denom for key, value in total_logs.items()}
        avg_logs["loss"] = avg_logs.get("gspo/loss", torch.tensor(0.0, device=device))

        # Compute a dummy loss for accelerator tracking (actual gradients already accumulated)
        dummy_loss = avg_logs["loss"].clone().detach().requires_grad_(True)
        return dummy_loss, avg_logs

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

    # ========== Main Training Loop ==========

    def train(self):
        """Main training loop for GSPO."""
        logger.info(
            f"Starting GSPO training: {len(self.train_dataloader)} batches per GPU, "
            f"{self.num_training_steps} total update steps"
        )

        progress_bar = tqdm(
            total=self.num_training_steps,
            disable=not self.accelerator.is_local_main_process,
        )

        for epoch in range(self.config.num_train_epochs):
            # Use eval mode for deterministic two-phase gradient estimation
            self.model.eval()

            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)

            for batch in self.train_dataloader:
                with self.accelerator.accumulate(self.model):
                    loss, logs = self.train_step_gspo(batch)

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
                    loss_val = logs.get("loss", torch.tensor(0.0))
                    progress_bar.set_postfix(
                        loss=f"{loss_val.item():.4f}"
                    )

                    if self.global_step % self.config.logging_steps == 0:
                        self._log_metrics(logs, epoch)
                    if self.global_step % self.config.save_steps == 0:
                        self._save_checkpoint()

        progress_bar.close()
        self._save_checkpoint(final=True)
        if self.wandb:
            self.wandb.finish()
        logger.info("GSPO training complete!")

    def _log_metrics(self, logs: Dict, epoch: int):
        """Log metrics to console and wandb."""
        if self.accelerator.is_main_process:
            log_str = f"Epoch {epoch} Step {self.global_step}: "
            log_str += ", ".join(
                f"{k}={v.item():.4f}"
                for k, v in logs.items()
                if isinstance(v, torch.Tensor) and v.numel() == 1
            )
            logger.info(log_str)

            if self.wandb:
                self.wandb.log(
                    {
                        k: v.item() if hasattr(v, 'item') else v
                        for k, v in logs.items()
                    },
                    step=self.global_step,
                )

    def _save_checkpoint(self, final: bool = False):
        """Save model checkpoint."""
        self.accelerator.wait_for_everyone()
        save_path = os.path.join(
            self.config.output_dir,
            "final" if final else f"checkpoint-{self.global_step}",
        )

        if self.accelerator.is_main_process:
            os.makedirs(save_path, exist_ok=True)
            self.accelerator.unwrap_model(self.model).save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            logger.info(f"Saved checkpoint to {save_path}")
