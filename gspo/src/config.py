# coding=utf-8
"""GSPO training configuration — self-contained, no external dependencies."""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import os
import yaml


@dataclass
class GSPOConfig:
    """Configuration for GSPO on-policy RL training on WeDLM.

    This is a self-contained config class; it does NOT depend on dpo/ or finetune/.
    """

    # ── Model ──
    model_path: str = "tencent/WeDLM-8B-Base"
    trust_remote_code: bool = True

    # ── Data ──
    train_data: str = "data/train.jsonl"
    max_seq_length: int = 2048

    # Training mode (always "gspo")
    training_mode: str = "gspo"

    # ── GSPO core ──
    gspo_group_size: int = 4             # G: responses sampled per prompt
    gspo_num_mask_samples: int = 4       # K: MC masking estimates per response

    # ── Generation ──
    gspo_temperature: float = 0.8         # sampling temperature during generation
    gspo_max_new_tokens: int = 512        # max tokens to generate per response
    gspo_entropy_threshold: float = 0.4   # WeDLM parallel-decoding entropy threshold
    gspo_pos_penalty_factor: float = 0.02 # WeDLM position penalty

    # ── Weight sync ──
    gspo_sync_every_n_steps: int = 10     # sync training weights → generator every N steps

    # ── Generator (engine) config ──
    gspo_window_size: int = 16            # WeDLM sliding window size for generation
    gspo_kvcache_block_size: int = 4096   # KV cache block size

    # ── Reward model ──
    reward_model_path: Optional[str] = None

    # ── WeDLM specific ──
    block_size: int = 32
    mask_per_block: bool = True
    loss_weighting_scheme: str = "weighted"  # "weighted" (1/γ) or "uniform"
    mask_eps: float = 1e-8
    num_learnable_im_end: int = 8

    # ── AR loss ──
    enable_ar_loss: bool = True
    ar_loss_weight: float = 1.0

    # ── Attention backend ──
    attention_backend: str = "magi"  # "magi" or "dense"

    # ── Training ──
    output_dir: str = "outputs"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 3e-6
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # ── Cache ──
    rebuild_cache: bool = False

    # ── DeepSpeed ──
    use_deepspeed: bool = False
    deepspeed_zero_stage: int = 2
    deepspeed_offload_optimizer: bool = False
    deepspeed_offload_param: bool = False
    deepspeed_offload_nvme: bool = False
    deepspeed_nvme_path: str = "/tmp/deepspeed_offload"
    deepspeed_pin_memory: bool = True
    deepspeed_overlap_comm: bool = True
    deepspeed_contiguous_gradients: bool = True
    deepspeed_reduce_bucket_size: int = 50000000
    deepspeed_stage3_prefetch_bucket_size: int = 50000000
    deepspeed_stage3_param_persistence_threshold: int = 100000
    deepspeed_stage3_max_live_parameters: int = 1000000000
    deepspeed_stage3_max_reuse_distance: int = 1000000000
    deepspeed_config_file: Optional[str] = None

    # ── Logging & Saving ──
    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 3

    # ── Device & Seed ──
    bf16: bool = True
    seed: int = 42

    # ── WandB (optional) ──
    use_wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_team: Optional[str] = None
    wandb_group: Optional[str] = None
    wandb_host: Optional[str] = None
    wandb_key: Optional[str] = None

    def __post_init__(self):
        if self.training_mode not in ("sft", "dpo", "gspo"):
            raise ValueError(f"Unknown training_mode: {self.training_mode}")

        if self.loss_weighting_scheme not in ("uniform", "weighted"):
            raise ValueError(f"Unknown loss_weighting_scheme: {self.loss_weighting_scheme}")

        if self.gspo_group_size < 2:
            raise ValueError(
                "gspo_group_size must be >= 2 (need at least 2 for group advantage)"
            )
        if self.gspo_num_mask_samples < 1:
            raise ValueError("gspo_num_mask_samples must be >= 1")
        if self.gspo_temperature < 0:
            raise ValueError("gspo_temperature must be >= 0")
        if self.attention_backend not in ("magi", "dense"):
            raise ValueError(f"Unknown attention_backend: {self.attention_backend}")

    # ── YAML I/O ──

    @classmethod
    def from_yaml(cls, path: str) -> "GSPOConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def save_yaml(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        save_dict = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(save_dict, f, default_flow_style=False)

    # ── DeepSpeed config generator ──

    def get_deepspeed_config(self) -> Optional[Dict[str, Any]]:
        if not self.use_deepspeed:
            return None
        if self.deepspeed_config_file and os.path.exists(self.deepspeed_config_file):
            import json
            with open(self.deepspeed_config_file, "r") as f:
                return json.load(f)

        ds_config = {
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "gradient_clipping": self.max_grad_norm,
            "steps_per_print": self.logging_steps,
            "wall_clock_breakdown": False,
        }
        ds_config["bf16" if self.bf16 else "fp16"] = {"enabled": True}

        zero_config = {
            "stage": self.deepspeed_zero_stage,
            "overlap_comm": self.deepspeed_overlap_comm,
            "contiguous_gradients": self.deepspeed_contiguous_gradients,
            "reduce_bucket_size": self.deepspeed_reduce_bucket_size,
            "allgather_bucket_size": self.deepspeed_reduce_bucket_size,
        }

        if self.deepspeed_offload_optimizer:
            device = "nvme" if self.deepspeed_offload_nvme else "cpu"
            zero_config["offload_optimizer"] = {
                "device": device,
                "pin_memory": self.deepspeed_pin_memory,
            }
            if self.deepspeed_offload_nvme:
                zero_config["offload_optimizer"]["nvme_path"] = self.deepspeed_nvme_path

        if self.deepspeed_zero_stage == 3:
            zero_config.update({
                "stage3_prefetch_bucket_size": self.deepspeed_stage3_prefetch_bucket_size,
                "stage3_param_persistence_threshold": self.deepspeed_stage3_param_persistence_threshold,
                "stage3_max_live_parameters": self.deepspeed_stage3_max_live_parameters,
                "stage3_max_reuse_distance": self.deepspeed_stage3_max_reuse_distance,
                "stage3_gather_16bit_weights_on_model_save": True,
            })
            if self.deepspeed_offload_param:
                device = "nvme" if self.deepspeed_offload_nvme else "cpu"
                zero_config["offload_param"] = {
                    "device": device,
                    "pin_memory": self.deepspeed_pin_memory,
                }

        ds_config["zero_optimization"] = zero_config
        return ds_config
