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
from src.loss import compute_ar_loss, compute_block_scores, compute_gspo_coefficients, compute_gspo_coefficients_with_kl, compute_gspo_loss, compute_kl_penalty
from src.attention import check_backend_available, get_available_backend, get_attention_wrapper
from src.generator import WeDLMGenerator
from src.reward import MathReward, ModelReward
from src.buffer import RolloutBuffer

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
        self._model_kwargs = dict(model_kwargs)

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

        # Math reward (Phase 1) or Model reward (Phase 3)
        if self.config.gspo_reward_type == "model":
            logger.info(f"Loading Reward Model from {self.config.gspo_reward_model_path}")
            self.reward_model = ModelReward(
                model_path=self.config.gspo_reward_model_path,
                tokenizer=self.tokenizer,
                device=self.accelerator.device,
                model_type=self.config.gspo_reward_model_type,
                torch_dtype=torch.bfloat16 if self.config.bf16 else torch.float32,
                trust_remote_code=self.config.trust_remote_code,
            )
        else:
            self.reward_model = MathReward(
                reward_type=self.config.gspo_reward_type,
                tokenizer=self.tokenizer,
            )

        # Rollout buffer (Phase 2)
        buffer_size = max(self.config.gspo_buffer_size,
                          self.config.gen_every_n_steps * self.config.per_device_train_batch_size)
        self.rollout_buffer = RolloutBuffer(max_size=buffer_size)
        self._step_in_gen_cycle = 0  # counter for gen_every_n_steps
        self._ref_on_cpu = False

        # Phase 3: monitoring state
        self._sample_prompt_cache: Optional[str] = None  # cached prompt for periodic logging
        self._last_logged_step: int = -1

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

        # Prepare ref model (keep raw, NOT through DeepSpeed)
        # ZeRO-2 already replicates model params on all GPUs, so DeepSpeed
        # wrapping adds zero benefit and causes GPU-0 memory concentration
        # due to engine hooks + all-reduce coordinator overhead.
        try:
            self.ref_model = self.ref_model.to(self.accelerator.device)
        except Exception as err:
            logger.warning(f"Failed to move ref model to device ({err})")
        self.ref_model.eval()
        logger.info(f"Reference model on {self.accelerator.device} (raw, no DeepSpeed wrapper)")

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

        # Phase 3: Resume from checkpoint if specified
        if self.config.gspo_resume_from_checkpoint:
            self._load_checkpoint(self.config.gspo_resume_from_checkpoint)

        # Phase 3: Log KL penalty status
        if self.config.gspo_use_kl_penalty and self.config.gspo_kl_coef > 0:
            logger.info(f"KL penalty enabled: coef={self.config.gspo_kl_coef}")
        if self.config.gspo_log_samples_every_n_steps > 0:
            logger.info(
                f"Periodic sample logging enabled: every {self.config.gspo_log_samples_every_n_steps} steps"
            )

    # ========== Ref Model Offload (Phase 2) ==========

    def _offload_ref_to_cpu(self):
        """Offload reference model to CPU to free GPU memory during policy backward.

        Only acts when config.ref_model_offload is True and ref is on GPU.
        """
        if not self.config.ref_model_offload:
            return
        if self._ref_on_cpu:
            return
        logger.debug("Offloading ref model to CPU...")
        self.ref_model = self.ref_model.cpu()
        self._ref_on_cpu = True
        torch.cuda.empty_cache()

    def _load_ref_to_gpu(self):
        """Load reference model back to GPU for scoring.

        Only acts when config.ref_model_offload is True and ref is on CPU.
        """
        if not self.config.ref_model_offload:
            return
        if not self._ref_on_cpu:
            return
        logger.debug("Loading ref model to GPU...")
        self.ref_model = self.ref_model.to(self.accelerator.device)
        self._ref_on_cpu = False

    # ========== Forward helpers ==========

    def _forward_wedlm_logits(
        self, model: torch.nn.Module, batch: WeDLMBatch
    ) -> torch.Tensor:
        """Forward helper for WeDLM logits."""
        forward_model = model
        if not (self.config.use_deepspeed and self.config.deepspeed_zero_stage == 3):
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

        When gen_every_n_steps == 1 (Phase 1 behavior):
            Generate → Reward → Ref Score → Policy Backward (all inline)

        When gen_every_n_steps > 1 (Phase 2 behavior):
            Generate → Reward → Ref Score → Push to buffer → Train from buffer
        """
        gen_every_n = self.config.gen_every_n_steps

        if gen_every_n <= 1:
            return self._train_step_gspo_full(batch)
        else:
            return self._train_step_with_buffer(batch)

    def _train_step_gspo_full(self, batch: Dict[str, Any]) -> tuple:
        """Full 4-phase GSPO step: generate + score + ref + backward (Phase 1 path)."""
        device = self.accelerator.device
        K = self.config.gspo_num_samples
        beta = float(self.config.gspo_beta)
        num_mask_samples = max(int(self.config.gspo_num_mask_samples), 1)
        sample_scale = 1.0 / float(num_mask_samples)

        # Extract batch data
        prompt_messages_list = batch["messages"]
        ground_truths = batch["ground_truths"]
        bs = len(prompt_messages_list)

        prompt_texts = [self._get_prompt_text(msgs) for msgs in prompt_messages_list]

        # ===== Phase 1-3: Generate + Score + Ref =====
        all_responses, rewards, ref_scores = self._generate_and_score_batch(
            prompt_texts, prompt_messages_list, ground_truths, K
        )

        # ===== Phase 4: Policy Scoring + Backward =====
        return self._policy_train_on_responses(
            all_responses, rewards, ref_scores, bs, K, beta,
            num_mask_samples, sample_scale, device
        )

    def _train_step_with_buffer(self, batch: Dict[str, Any]) -> tuple:
        """Train with rollout buffer: generate+push every N steps, always train from buffer.

        The dataloader batch is only used for generation; training always
        samples from the buffer to ensure on-policy freshness decay.
        """
        device = self.accelerator.device
        K = self.config.gspo_num_samples
        beta = float(self.config.gspo_beta)
        num_mask_samples = max(int(self.config.gspo_num_mask_samples), 1)
        sample_scale = 1.0 / float(num_mask_samples)

        prompt_messages_list = batch["messages"]
        ground_truths = batch["ground_truths"]
        bs = len(prompt_messages_list)

        # Track generation cycle
        self._step_in_gen_cycle += 1
        gen_every_n = self.config.gen_every_n_steps

        if self._step_in_gen_cycle >= gen_every_n or len(self.rollout_buffer) == 0:
            self._step_in_gen_cycle = 0
            # === Generation step: generate K responses, score, push to buffer ===
            prompt_texts = [self._get_prompt_text(msgs) for msgs in prompt_messages_list]
            all_responses, rewards, ref_scores = self._generate_and_score_batch(
                prompt_texts, prompt_messages_list, ground_truths, K
            )
            self.rollout_buffer.push_batch(
                prompt_texts, prompt_messages_list, ground_truths,
                all_responses, rewards, ref_scores
            )
            if self.accelerator.is_main_process:
                logger.debug(
                    f"Step {self.global_step}: generated {bs} prompts, "
                    f"buffer size={len(self.rollout_buffer)}/{self.rollout_buffer.max_size}"
                )
        else:
            # Non-generation step: discard dataloader batch, train from buffer only
            pass

        # === Train from buffer ===
        try:
            buf_batch = self.rollout_buffer.sample_batch(bs)
        except (IndexError, RuntimeError) as e:
            logger.warning(f"Cannot sample from buffer: {e}, returning zero loss")
            device = self.accelerator.device
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            return zero, {"loss": torch.tensor(0.0, device=device)}

        return self._policy_train_on_responses(
            buf_batch["all_responses"], buf_batch["rewards"], buf_batch["ref_scores"],
            bs, K, beta, num_mask_samples, sample_scale, device
        )

    def _generate_and_score_batch(
        self,
        prompt_texts: List[str],
        prompt_messages_list: List[List[Dict]],
        ground_truths: List[str],
        K: int,
    ) -> Tuple[List[List[Dict]], torch.Tensor, torch.Tensor]:
        """Phases 1-3: Generate K responses per prompt, compute rewards and ref scores.

        Returns:
            all_responses: List[bs][K] of response dicts
            rewards: Tensor [bs, K]
            ref_scores: Tensor [bs, K]
        """
        device = self.accelerator.device

        # ===== Phase 1: Generation (no_grad) =====
        self.model.eval()
        self._gather_params_for_generation()  # ZeRO-3 safety: gather shards before reading weights
        all_responses = []

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
        torch.cuda.empty_cache()

        # ===== Phase 2: Reward Scoring (no_grad) =====
        with torch.no_grad():
            rewards = []
            for prompt_text, responses, gt in zip(
                prompt_texts, all_responses, ground_truths
            ):
                response_texts = [r["text"] for r in responses]
                prompt_rewards = self.reward_model.compute_rewards(
                    [prompt_text] * K, response_texts, [gt] * K
                )
                # Debug: log per-prompt reward summary before normalization
                logger.info(
                    f"Rewards raw: GT={gt!r} | values={prompt_rewards.tolist()} | "
                    f"mean={prompt_rewards.mean().item():.3f} std={prompt_rewards.std().item():.3f}"
                )
                # Normalize rewards within group (skip if all identical)
                if prompt_rewards.std() > 1e-8 and prompt_rewards.numel() > 1:
                    prompt_rewards = (prompt_rewards - prompt_rewards.mean()) / (
                        prompt_rewards.std() + 1e-8
                    )
                else:
                    logger.warning(
                        f"All rewards identical (std≈0) for GT={gt!r}, "
                        f"values={prompt_rewards.tolist()}. "
                        f"GSPO loss will be 0 for this prompt!"
                    )
                rewards.append(prompt_rewards)
            rewards = torch.stack(rewards, dim=0)  # [bs, K]

        # ===== Phase 3: Reference Scoring (no_grad) =====
        # Ensure ref model is on GPU
        self._load_ref_to_gpu()

        ref_scores = []
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

        # Offload ref to CPU after scoring to free GPU memory for policy backward
        self._offload_ref_to_cpu()
        torch.cuda.empty_cache()

        return all_responses, rewards, ref_scores

    def _policy_train_on_responses(
        self,
        all_responses: List[List[Dict]],
        rewards: torch.Tensor,
        ref_scores: torch.Tensor,
        bs: int,
        K: int,
        beta: float,
        num_mask_samples: int,
        sample_scale: float,
        device: torch.device,
    ) -> tuple:
        """Phase 4: Policy scoring + per-branch backward on pre-computed responses.

        Phase 3 additions:
        - KL penalty (when gspo_use_kl_penalty=True)
        - Generation quality metrics collection

        Args:
            all_responses: List[bs][K] of response dicts.
            rewards: Tensor [bs, K] of pre-computed rewards.
            ref_scores: Tensor [bs, K] of pre-computed ref scores.
            bs, K: Batch size and number of samples per prompt.
            beta: GSPO temperature.
            num_mask_samples: Number of MC mask samples.
            sample_scale: 1.0 / num_mask_samples.
            device: Target device.

        Returns:
            (dummy_loss, avg_logs) tuple.
        """
        total_dpo_loss = torch.tensor(0.0, device=device)
        total_logs: Dict[str, torch.Tensor] = {}
        total_quality_logs: Dict[str, List[float]] = {
            "gen/response_length": [],
            "gen/unique_token_ratio": [],
        }

        kl_coef = self.config.gspo_kl_coef if self.config.gspo_use_kl_penalty else 0.0

        for sample_idx in range(bs):
            sample_responses = all_responses[sample_idx]
            sample_rewards = rewards[sample_idx].to(device)  # [K]
            sample_ref = ref_scores[sample_idx].to(device)  # [K]

            # === Phase 3: Collect generation quality metrics ===
            for resp in sample_responses:
                resp_ids = resp.get("input_ids")
                labels = resp.get("labels")
                if resp_ids is not None and labels is not None:
                    # Response length (non -100 tokens in labels)
                    resp_mask = labels != -100
                    resp_len = int(resp_mask.sum().item())
                    total_quality_logs["gen/response_length"].append(float(resp_len))
                    # Unique token ratio in response
                    resp_tokens = labels[resp_mask]
                    if resp_tokens.numel() > 0:
                        unique_ratio = len(set(resp_tokens.tolist())) / max(resp_tokens.numel(), 1)
                        total_quality_logs["gen/unique_token_ratio"].append(float(unique_ratio))

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

                # 4b. Compute GSPO coefficients (with optional KL penalty)
                coeffs = compute_gspo_coefficients_with_kl(
                    policy_scores_ng, sample_ref, sample_rewards, beta, kl_coef
                )  # [K]

                # 4c. Compute GSPO loss for logging
                _, gspo_logs = compute_gspo_loss(
                    policy_scores_ng, sample_ref, sample_rewards, beta
                )

                # 4d. KL penalty logging (Phase 3)
                if kl_coef > 0:
                    kl_loss, kl_per_sample = compute_kl_penalty(
                        policy_scores_ng, sample_ref
                    )
                    gspo_logs["gspo/kl_penalty"] = kl_loss.detach()
                    gspo_logs["gspo/kl_penalty_max"] = kl_per_sample.max()

                # 4e. Per-branch backward
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

        # Aggregate quality metrics
        for key, values in total_quality_logs.items():
            if values:
                total_logs[key] = torch.tensor(
                    sum(values) / len(values), device=device
                )
        # Reward distribution stats
        if rewards.numel() > 0:
            total_logs["gen/reward_mean"] = rewards.float().mean().to(device)
            total_logs["gen/reward_std"] = rewards.float().std().to(device)
            total_logs["gen/reward_min"] = rewards.float().min().to(device)
            total_logs["gen/reward_max"] = rewards.float().max().to(device)

        del ref_scores, all_responses, rewards
        torch.cuda.empty_cache()

        # Average logs over batch
        denom = float(bs * num_mask_samples)
        # Quality metrics are per-sample averages, only divide by bs
        quality_keys = {"gen/response_length", "gen/unique_token_ratio",
                        "gen/reward_mean", "gen/reward_std",
                        "gen/reward_min", "gen/reward_max"}
        avg_logs = {}
        for key, value in total_logs.items():
            if key in quality_keys:
                avg_logs[key] = value / float(bs)
            else:
                avg_logs[key] = value / denom
        avg_logs["loss"] = avg_logs.get("gspo/loss", torch.tensor(0.0, device=device))

        # Dummy loss for accelerator tracking (actual gradients already accumulated)
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

    # ========== Phase 3: Periodic Generation Sample Logging ==========

    def _log_generation_samples(self):
        """Generate and log sample responses for quality monitoring.

        Designed to be called alongside checkpoint saves (not in the hot
        path of every training step) to avoid blocking distributed training.
        Only runs on main process.
        """
        if not self.accelerator.is_main_process:
            return

        log_every = self.config.gspo_log_samples_every_n_steps
        if log_every <= 0:
            return
        if self.global_step % log_every != 0:
            return
        if self._last_logged_step == self.global_step:
            return  # already logged at this step (dedup guard)

        # Cache a sample prompt on first call
        if self._sample_prompt_cache is None:
            try:
                sample = self.train_dataset[0]
                self._sample_prompt_cache = self._get_prompt_text(sample["messages"])
            except Exception as e:
                logger.warning(f"Cannot cache sample prompt: {e}")
                return

        prompt_text = self._sample_prompt_cache
        K = self.config.gspo_num_samples

        logger.info(f"[Step {self.global_step}] Generating {K} sample responses for monitoring...")

        try:
            self.model.eval()
            self._gather_params_for_generation()

            with torch.no_grad():
                responses = []
                for k in range(K):
                    seed = self.config.seed + k + self.global_step
                    resp = self.generator.generate(prompt_text, seed=seed)
                    responses.append(resp)

            self.model.train()

            # Log prompt (truncated)
            prompt_preview = prompt_text[:300] + "..." if len(prompt_text) > 300 else prompt_text
            logger.info(f"[Monitor] Prompt: {prompt_preview}")

            # Log each response with its length
            for k, resp in enumerate(responses):
                text = resp.get("text", "")
                labels = resp.get("labels")
                resp_len = int((labels != -100).sum().item()) if labels is not None else 0
                text_preview = text[:200] + "..." if len(text) > 200 else text
                logger.info(f"[Monitor] Response {k+1}/{K} (len={resp_len}): {text_preview}")

            # Log to wandb if enabled
            if self.wandb:
                sample_table = self.wandb.Table(
                    columns=["Response #", "Length", "Text"]
                )
                for k, resp in enumerate(responses):
                    text = resp.get("text", "")
                    labels = resp.get("labels")
                    resp_len = int((labels != -100).sum().item()) if labels is not None else 0
                    sample_table.add_data(k + 1, resp_len, text[:500])
                self.wandb.log(
                    {"monitoring/samples": sample_table},
                    step=self.global_step,
                )

            self._last_logged_step = self.global_step

        except Exception as e:
            logger.warning(f"Sample generation failed: {e}")
            self.model.train()

    def _calc_gpu_memory(self) -> str:
        """Return a compact per-GPU memory usage string for debugging."""
        if not torch.cuda.is_available():
            return "cuda=N/A"
        parts = []
        for i in range(torch.cuda.device_count()):
            alloc = torch.cuda.memory_allocated(i) / (1024**3)
            reserved = torch.cuda.memory_reserved(i) / (1024**3)
            parts.append(f"gpu{i}:{alloc:.1f}/{reserved:.1f}G")
        return " ".join(parts)

    # ========== Main Training Loop ==========

    def train(self):
        """Main training loop for GSPO.

        Two modes:
        - gen_every_n_steps == 1: generate + train every step (Phase 1)
        - gen_every_n_steps > 1: generate every N steps, train from buffer (Phase 2)
        """
        gen_every_n = self.config.gen_every_n_steps
        logger.info(
            f"Starting GSPO training: {len(self.train_dataloader)} batches per GPU, "
            f"{self.num_training_steps} total update steps, "
            f"gen_every_n={gen_every_n}"
        )

        progress_bar = tqdm(
            total=self.num_training_steps,
            initial=self.global_step,
            disable=not self.accelerator.is_local_main_process,
        )

        if self.accelerator.is_main_process:
            logger.info(f"Initial GPU memory: {self._calc_gpu_memory()}")
            if gen_every_n > 1:
                logger.info(
                    f"RolloutBuffer: size={self.rollout_buffer.max_size}, "
                    f"generating every {gen_every_n} steps"
                )

        # Phase 3: Log baseline generation samples before training starts
        self._log_generation_samples()

        for epoch in range(self.config.num_train_epochs):
            self.model.eval()

            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)

            for batch in self.train_dataloader:
                # Phase 2: multi-GPU sync before generation (ensure different prompts per GPU)
                if gen_every_n > 1 and self.accelerator.num_processes > 1:
                    self.accelerator.wait_for_everyone()

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
                        logs["gpu_memory"] = self._calc_gpu_memory()
                        if gen_every_n > 1:
                            logs["buffer/size"] = len(self.rollout_buffer)
                        self._log_metrics(logs, epoch)

                    if self.global_step % self.config.save_steps == 0:
                        self._save_checkpoint()

        progress_bar.close()
        # Phase 3: run generation sample logging at end of training (safe: all ranks alive)
        self._log_generation_samples()
        # Explicit barrier: ensure all ranks finish logging before final save
        self.accelerator.wait_for_everyone()
        self._save_checkpoint(final=True)
        if self.wandb:
            self.wandb.finish()
        logger.info("GSPO training complete!")

    def _log_metrics(self, logs: Dict, epoch: int):
        """Log metrics to console and wandb."""
        if self.accelerator.is_main_process:
            # Add current learning rate
            logs["lr"] = self.lr_scheduler.get_last_lr()[0]

            log_parts = [f"Epoch {epoch} Step {self.global_step}"]
            for k, v in logs.items():
                if isinstance(v, str):
                    log_parts.append(f"{k}={v}")
                elif isinstance(v, torch.Tensor) and v.numel() == 1:
                    log_parts.append(f"{k}={v.item():.4f}")
                elif isinstance(v, float):
                    log_parts.append(f"{k}={v:.4f}")
            logger.info(": ".join([log_parts[0], ", ".join(log_parts[1:])]))

            if self.wandb:
                self.wandb.log(
                    {
                        k: v.item() if hasattr(v, 'item') else v
                        for k, v in logs.items()
                    },
                    step=self.global_step,
                )

    # ========== Phase 3: Checkpoint Save / Resume ==========

    def _save_checkpoint(self, final: bool = False):
        """Save model checkpoint with full training state for resumption.

        For the final checkpoint, only model weights + tokenizer are saved
        (skipping optimizer state to avoid DeepSpeed all-gather hang when
        GPU memory is tight).

        For intermediate checkpoints, full trainer state is saved.
        """
        # Free GPU memory before saving to reduce OOM risk during all-gather
        torch.cuda.empty_cache()

        save_path = os.path.join(
            self.config.output_dir,
            "final" if final else f"checkpoint-{self.global_step}",
        )

        if self.accelerator.is_main_process:
            os.makedirs(save_path, exist_ok=True)

        # For final checkpoint: save model weights only (no all-gather needed)
        # This avoids the DeepSpeed ZeRO-2 optimizer state_dict hang
        if final:
            if self.accelerator.is_main_process:
                try:
                    self.accelerator.unwrap_model(self.model).save_pretrained(
                        save_path, safe_serialization=True
                    )
                    self.tokenizer.save_pretrained(save_path)
                    logger.info(f"Saved final model to {save_path}")
                except Exception as e:
                    logger.error(f"Failed to save final checkpoint: {e}")
            # No wait_for_everyone for final save — avoid hang if a rank died
            return

        # Intermediate checkpoint: full state (needs all processes alive)
        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            # Save model and tokenizer
            self.accelerator.unwrap_model(self.model).save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)

            # Save trainer state for resumption (DeepSpeed optimizer is skipped)
            trainer_state = {
                "global_step": self.global_step,
                "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
                "step_in_gen_cycle": self._step_in_gen_cycle,
            }
            # Only save optimizer state if NOT using DeepSpeed (DeepSpeed state_dict
            # requires all ranks to participate in all-gather, can hang if OOM)
            if not self.config.use_deepspeed:
                try:
                    trainer_state["optimizer_state_dict"] = self.optimizer.state_dict()
                except Exception as e:
                    logger.warning(f"Could not save optimizer state: {e}")

            # Save RNG states for reproducibility
            import random
            trainer_state["python_rng_state"] = random.getstate()
            trainer_state["torch_rng_state"] = torch.get_rng_state()
            if torch.cuda.is_available():
                trainer_state["cuda_rng_state"] = torch.cuda.get_rng_state()

            state_path = os.path.join(save_path, "trainer_state.pt")
            torch.save(trainer_state, state_path)
            logger.info(f"Saved checkpoint to {save_path} (step={self.global_step})")

    def _load_checkpoint(self, checkpoint_path: str):
        """Load model and training state from a checkpoint for resumption.

        Args:
            checkpoint_path: Path to the checkpoint directory containing
                             model weights, tokenizer, and trainer_state.pt.
        """
        if not os.path.isdir(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")

        state_path = os.path.join(checkpoint_path, "trainer_state.pt")
        if not os.path.isfile(state_path):
            raise FileNotFoundError(
                f"trainer_state.pt not found in {checkpoint_path}. "
                f"Only final checkpoints (saved with Phase 3+) support resumption."
            )

        logger.info(f"Resuming from checkpoint: {checkpoint_path}")

        # Load trainer state
        trainer_state = torch.load(state_path, map_location="cpu")

        # Restore global step
        self.global_step = trainer_state.get("global_step", 0)
        self._step_in_gen_cycle = trainer_state.get("step_in_gen_cycle", 0)

        # Restore optimizer state
        if "optimizer_state_dict" in trainer_state:
            try:
                self.optimizer.load_state_dict(trainer_state["optimizer_state_dict"])
                logger.info("Optimizer state restored")
            except Exception as e:
                logger.warning(f"Failed to restore optimizer state: {e}")

        # Restore scheduler state
        if "lr_scheduler_state_dict" in trainer_state:
            try:
                self.lr_scheduler.load_state_dict(trainer_state["lr_scheduler_state_dict"])
                logger.info("LR scheduler state restored")
            except Exception as e:
                logger.warning(f"Failed to restore scheduler state: {e}")

        # Restore RNG states
        if "python_rng_state" in trainer_state:
            import random
            random.setstate(trainer_state["python_rng_state"])
        if "torch_rng_state" in trainer_state:
            torch.set_rng_state(trainer_state["torch_rng_state"])
        if "cuda_rng_state" in trainer_state and torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state(trainer_state["cuda_rng_state"])
            except Exception as e:
                logger.warning(f"Failed to restore CUDA RNG state: {e}")

        logger.info(f"Resumed from step {self.global_step}")

    # ========== Phase 3: ZeRO-3 Safety ==========

    @torch.no_grad()
    def _gather_params_for_generation(self):
        """Ensure all model parameters are gathered before generation.

        Under ZeRO-3, parameters are sharded across GPUs. Before generation,
        we use DeepSpeed's GatheredParameters context to gather all params
        so the generator can read the full model weights.

        This should ONLY be called right before generation, NOT in the main
        training loop (barriers there cause distributed deadlocks).
        """
        if not (self.config.use_deepspeed and self.config.deepspeed_zero_stage == 3):
            return
        try:
            import deepspeed
            from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
            params_to_gather = [
                p for p in self.model.parameters()
                if hasattr(p, 'ds_status') and p.ds_status != ZeroParamStatus.NOT_AVAILABLE
            ]
            if params_to_gather:
                with deepspeed.zero.GatheredParameters(params_to_gather, modifier_rank=0):
                    pass  # params gathered within context; released on exit
        except Exception as e:
            logger.debug(f"ZeRO-3 param gather skipped: {e}")
