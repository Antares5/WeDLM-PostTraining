# coding=utf-8
"""GSPO Trainer with DeepSpeed multi-GPU support.

Architecture
------------
The GSPO trainer follows the same Accelerate + DeepSpeed pattern as the DPO
trainer in dpo/src/trainer.py, but implements the 3-phase GSPO loop:

  Phase 1 (external):  Generate responses from current policy → token ids + rewards
  Phase 2 (train step):  wedlm_forward → compute_block_scores → compute_gspo_loss → backward
  Phase 3 (periodic):    Sync weights to generator (handled externally, every K steps)

For the MVP (Step 3), Phase 1 uses mock/offline responses. Real on-policy
generation will be integrated in Step 4.
"""

import os
import math
import logging
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler

from gspo.src.config import GSPOConfig
from gspo.src.data import GSPOPromptDataset, gspo_collate_fn
from gspo.src.batch import WeDLMBatch, build_wedlm_batch_from_response
from gspo.src.model import wedlm_forward
from gspo.src.loss import compute_block_scores, compute_gspo_loss
from gspo.src.attention import check_backend_available, get_available_backend, get_attention_wrapper

logger = logging.getLogger(__name__)

MASK_TOKEN_ID = 151665

# Lazy import wandb
_wandb = None


def _init_wandb(config: GSPOConfig, accelerator: Accelerator):
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
        config={k: v for k, v in config.__dict__.items() if not k.startswith("_")},
    )
    return wandb


# ── Mock dataset for smoke testing ──

class GSPOMockResponseDataset(Dataset):
    """Mock dataset that returns pre-grouped responses per prompt.

    Each __getitem__ returns a list of G dicts for ONE prompt, so the
    DataLoader naturally batches complete groups (all G responses for
    one prompt). This is essential because compute_gspo_loss needs
    group_size >= 2 within each batch entry to compute advantages.

    In production, each group comes from the WeDLM generator + reward model.
    """

    def __init__(
        self,
        prompts: List[List[int]],
        gspo_group_size: int,
        max_response_len: int = 512,
        seed: int = 42,
    ):
        self.gspo_group_size = gspo_group_size
        self.max_response_len = max_response_len
        self.num_prompts = len(prompts)
        rng = torch.Generator().manual_seed(seed)

        # Pre-build one group per prompt
        self.groups: List[List[Dict]] = []
        for b, prompt_ids in enumerate(prompts):
            prompt_list = list(prompt_ids)
            group_samples: List[Dict] = []
            for g in range(gspo_group_size):
                extra_len = int(torch.randint(16, max_response_len + 1, (1,), generator=rng).item())
                continuation = torch.randint(0, 50000, (extra_len,), generator=rng).tolist()
                response_ids = prompt_list + continuation
                reward = float(len(continuation)) / max_response_len + torch.rand(1, generator=rng).item() * 0.5
                group_samples.append({
                    "response_ids": response_ids,
                    "prompt_len": len(prompt_list),
                    "reward": reward,
                    "prompt_idx": b,
                })
            self.groups.append(group_samples)

    def __len__(self):
        return self.num_prompts

    def __getitem__(self, idx):
        """Return all G responses for the idx-th prompt as a list."""
        return self.groups[idx]


def gspo_mock_collate_fn(batch: List[List[Dict]]) -> Dict:
    """Collate: batch is a list of groups, each with G samples.
    Flatten into a single list of samples for the train step.
    """
    flat = []
    for group in batch:
        flat.extend(group)
    return {"samples": flat}


# ═══════════════════════════════════════════════════════════════
# GSPO Trainer
# ═══════════════════════════════════════════════════════════════

class GSPOTrainer:
    """Trainer for GSPO on-policy RL training with DeepSpeed support.

    Usage:
        config = GSPOConfig.from_yaml("configs/example.yaml")
        accelerator = Accelerator(...)
        trainer = GSPOTrainer(config, accelerator)
        trainer.train()
    """

    def __init__(self, config: GSPOConfig, accelerator: Accelerator):
        self.config = config
        self.accelerator = accelerator
        self.wandb = _init_wandb(config, accelerator)
        self._setup()
        self._prepare_training()

    # ── Setup ──

    def _setup(self):
        """Initialize model, tokenizer, attention wrapper, and dataset."""
        if not check_backend_available(self.config.attention_backend):
            self.config.attention_backend = get_available_backend()
        logger.info(f"Attention backend: {self.config.attention_backend}")
        logger.info(f"Training mode: {self.config.training_mode}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path, trust_remote_code=self.config.trust_remote_code
        )
        # Fallback pad token
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id or MASK_TOKEN_ID

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

        head_dim = self.model.config.hidden_size // self.model.config.num_attention_heads
        self.attn_wrapper = get_attention_wrapper(
            self.config.attention_backend, head_dim, deterministic=False,
        )
        if hasattr(self.attn_wrapper, "to"):
            self.attn_wrapper = self.attn_wrapper.to(self.accelerator.device)

        # Dataset: load prompts from JSONL
        self.prompt_dataset = GSPOPromptDataset(
            data_path=self.config.train_data,
            tokenizer=self.tokenizer,
            max_prompt_length=self.config.max_seq_length,
        )
        if len(self.prompt_dataset) == 0:
            raise RuntimeError(f"No valid prompts found in {self.config.train_data}")

        logger.info(f"Loaded {len(self.prompt_dataset)} prompts for GSPO training")

        # Build mock response dataset (replaced by real generator in Step 4)
        self.train_dataset = GSPOMockResponseDataset(
            prompts=[self.prompt_dataset[i].tolist() for i in range(len(self.prompt_dataset))],
            gspo_group_size=self.config.gspo_group_size,
            max_response_len=min(self.config.gspo_max_new_tokens, 512),
            seed=self.config.seed,
        )
        logger.info(
            f"Built mock dataset with {len(self.train_dataset)} responses "
            f"(B={len(self.prompt_dataset)}, G={self.config.gspo_group_size})"
        )

        # Distributed sampler
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
            batch_size=self.config.per_device_train_batch_size,  # prompts per batch
            sampler=self.train_sampler,
            shuffle=shuffle,
            collate_fn=gspo_mock_collate_fn,
            num_workers=0,
            pin_memory=False,
        )

    def _prepare_training(self):
        """Prepare optimizer, scheduler, and Accelerate wrapper."""
        steps_per_epoch = len(self.train_dataloader)

        num_update_steps_per_epoch = math.ceil(
            steps_per_epoch / self.config.gradient_accumulation_steps
        )
        self.num_training_steps = num_update_steps_per_epoch * self.config.num_train_epochs
        num_warmup_steps = int(self.num_training_steps * self.config.warmup_ratio)

        if self.accelerator.is_main_process:
            logger.info(f"=== GSPO Training Configuration ===")
            logger.info(f"GPUs: {self.accelerator.num_processes}")
            logger.info(f"Responses per GPU per epoch: {steps_per_epoch}")
            logger.info(f"Gradient accumulation steps: {self.config.gradient_accumulation_steps}")
            logger.info(f"Update steps per epoch: {num_update_steps_per_epoch}")
            logger.info(f"Total training steps: {self.num_training_steps}")
            logger.info(f"Warmup steps: {num_warmup_steps}")
            logger.info(f"Group size G: {self.config.gspo_group_size}")
            logger.info(f"MC mask samples K: {self.config.gspo_num_mask_samples}")

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

        self.lr_scheduler = get_scheduler(
            self.config.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=self.num_training_steps,
        )

        self.model, self.optimizer, self.train_dataloader, self.lr_scheduler = (
            self.accelerator.prepare(
                self.model, self.optimizer, self.train_dataloader, self.lr_scheduler
            )
        )

        self.global_step = 0

    # ── Forward helpers ──

    def _forward_wedlm_logits(
        self, model: torch.nn.Module, batch: WeDLMBatch
    ) -> torch.Tensor:
        """Forward WeDLM model and return logits."""
        try:
            forward_model = self.accelerator.unwrap_model(model)
        except Exception:
            forward_model = model
        return wedlm_forward(
            forward_model, batch, self.attn_wrapper, self.config.attention_backend
        )

    # ── Train step ──

    def train_step_gspo(self, batch: Dict) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Single GSPO training step.

        Args:
            batch: Dict with key "samples" → list of dicts, each having:
                response_ids, prompt_len, reward, prompt_idx.

        Returns:
            loss: Scalar loss (for backward).
            logs: Dict of monitoring metrics.
        """
        device = self.accelerator.device
        samples: List[Dict] = batch["samples"]
        N = len(samples)
        K = max(int(self.config.gspo_num_mask_samples), 1)

        if N == 0:
            return torch.tensor(0.0, device=device, requires_grad=True), {"loss": torch.tensor(0.0, device=device)}

        all_scores: List[torch.Tensor] = []
        all_rewards: List[float] = []
        all_prompt_idx: List[int] = []

        # Score each response with K MC masking samples
        for sample in samples:
            response_ids = torch.tensor(sample["response_ids"], device=device, dtype=torch.long)
            prompt_len = sample["prompt_len"]

            score_sum = torch.tensor(0.0, device=device)
            for _ in range(K):
                batch_k = build_wedlm_batch_from_response(
                    response_ids=response_ids,
                    prompt_len=prompt_len,
                    block_size=self.config.block_size,
                    mask_token_id=MASK_TOKEN_ID,
                    backend=self.config.attention_backend,
                    mask_per_block=self.config.mask_per_block,
                    eps=self.config.mask_eps,
                )
                logits = self._forward_wedlm_logits(self.model, batch_k)
                s, _ = compute_block_scores(
                    logits=logits,
                    targets=batch_k.original_ids,
                    masked_indices=batch_k.masked_indices,
                    p_mask=batch_k.p_mask,
                    logical_positions=batch_k.logical_positions,
                    cum_seqlens=batch_k.cum_seqlens,
                    block_size=self.config.block_size,
                    weighting_scheme=self.config.loss_weighting_scheme,
                    block_reduce="mean",
                    seq_reduce="mean",
                    eps=self.config.mask_eps,
                )
                score_sum = score_sum + s[0] / K  # single-sequence score

            # Guard against NaN scores (can occur with extreme masking ratios)
            if not torch.isfinite(score_sum):
                score_sum = torch.tensor(0.0, device=device, requires_grad=True)
            all_scores.append(score_sum)
            all_rewards.append(sample["reward"])
            all_prompt_idx.append(sample["prompt_idx"])

        scores_t = torch.stack(all_scores)
        rewards_t = torch.tensor(all_rewards, device=device, dtype=scores_t.dtype)
        prompt_t = torch.tensor(all_prompt_idx, device=device, dtype=torch.long)

        loss, logs = compute_gspo_loss(scores_t, rewards_t, prompt_t)
        return loss, logs

    def train_step(self, batch: Dict) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Dispatch to GSPO train step."""
        return self.train_step_gspo(batch)

    # ── Main training loop ──

    def train(self):
        """Main GSPO training loop."""
        logger.info(
            f"Starting GSPO training: {len(self.train_dataloader)} batches/GPU, "
            f"{self.num_training_steps} total update steps"
        )

        progress_bar = tqdm(
            total=self.num_training_steps,
            disable=not self.accelerator.is_local_main_process,
        )

        for epoch in range(self.config.num_train_epochs):
            self.model.train()

            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)

            for batch in self.train_dataloader:
                with self.accelerator.accumulate(self.model):
                    loss, logs = self.train_step(batch)
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
                    progress_bar.set_postfix(
                        loss=f"{logs.get('loss', torch.tensor(0.0)).detach().item():.4f}"
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

    # ── Logging & Saving ──

    def _log_metrics(self, logs: Dict, epoch: int):
        if self.accelerator.is_main_process:
            log_str = f"Epoch {epoch} Step {self.global_step}: "
            log_str += ", ".join(
                f"{k}={v.item():.4f}"
                for k, v in logs.items()
                if isinstance(v, torch.Tensor) and v.dim() == 0
            )
            logger.info(log_str)
            if self.wandb:
                self.wandb.log(
                    {k: v.item() if hasattr(v, "item") else v for k, v in logs.items()},
                    step=self.global_step,
                )

    def _save_checkpoint(self, final: bool = False):
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
