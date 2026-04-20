# coding=utf-8
"""Training configuration for WeDLM SFT."""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import yaml
import os


@dataclass
class WeDLMTrainingConfig:
    """Configuration for WeDLM SFT training."""
    
    # Model
    model_path: str = "tencent/WeDLM-8B-Instruct"
    trust_remote_code: bool = True
    
    # Data
    train_data: str = "data/train.jsonl"
    max_seq_length: int = 2048

    # Training mode
    training_mode: str = "sft"  # "sft", "dpo" or "gspo"

    # DPO scaffold (incremental phase 1/2)
    dpo_train_data: Optional[str] = None
    dpo_beta: float = 0.1
    dpo_ref_model_path: Optional[str] = None
    dpo_length_norm: bool = True
    dpo_block_reduce: str = "mean"  # "mean" or "sum"
    dpo_seq_reduce: str = "mean"  # "mean" or "sum"
    dpo_num_mask_samples: int = 1

    # Online GSPO
    gspo_train_data: Optional[str] = None
    gspo_ref_model_path: Optional[str] = None
    gspo_group_size: int = 4
    gspo_min_candidates_per_group: int = 2
    gspo_max_prompt_length: int = 1536
    gspo_max_new_tokens: int = 256
    gspo_rollout_temperature: float = 0.8
    gspo_rollout_top_p: float = 0.95
    gspo_rollout_top_k: int = 0
    gspo_score_temperature: float = 1.0
    gspo_reward_temperature: float = 1.0
    gspo_ref_alpha: float = 1.0
    gspo_kl_coef: float = 0.0
    gspo_reward_beta: float = 0.1
    gspo_reward_source: str = "policy_ref_margin"
    gspo_reward_length_penalty: float = 0.0
    gspo_reward_callable: Optional[str] = None
    gspo_reward_clip_min: Optional[float] = None
    gspo_reward_clip_max: Optional[float] = None
    gspo_deepmath_correct_bonus: float = 1.0
    gspo_deepmath_wrong_penalty: float = 0.0
    
    # WeDLM specific
    block_size: int = 32
    mask_per_block: bool = True
    loss_weighting_scheme: str = "weighted"  # "weighted" (1/γ) or "uniform"
    mask_eps: float = 1e-8
    num_learnable_im_end: int = 8
    
    # AR loss
    enable_ar_loss: bool = True
    ar_loss_weight: float = 1.0
    
    # Attention backend
    attention_backend: str = "magi"  # "magi" or "dense"
    
    # Training
    output_dir: str = "outputs"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 3e-6
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Cache
    rebuild_cache: bool = False
    
    # DeepSpeed
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
    
    # Logging & Saving
    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 3
    
    # Device & Seed
    bf16: bool = True
    seed: int = 42
    
    # WandB (optional)
    use_wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_team: Optional[str] = None  # entity
    wandb_group: Optional[str] = None
    wandb_host: Optional[str] = None  # for private deployment
    wandb_key: Optional[str] = None   # API key
    
    def __post_init__(self):
        if self.training_mode not in ["sft", "dpo", "gspo"]:
            raise ValueError(f"Unknown training_mode: {self.training_mode}")

        if self.loss_weighting_scheme not in ["uniform", "weighted"]:
            raise ValueError(f"Unknown loss_weighting_scheme: {self.loss_weighting_scheme}")

        if self.dpo_block_reduce not in ["mean", "sum"]:
            raise ValueError(f"Unknown dpo_block_reduce: {self.dpo_block_reduce}")

        if self.dpo_seq_reduce not in ["mean", "sum"]:
            raise ValueError(f"Unknown dpo_seq_reduce: {self.dpo_seq_reduce}")

        if self.dpo_num_mask_samples < 1:
            raise ValueError("dpo_num_mask_samples must be >= 1")

        if self.gspo_group_size < 2:
            raise ValueError("gspo_group_size must be >= 2")

        if self.gspo_min_candidates_per_group < 2:
            raise ValueError("gspo_min_candidates_per_group must be >= 2")

        if self.gspo_min_candidates_per_group > self.gspo_group_size:
            raise ValueError("gspo_min_candidates_per_group must be <= gspo_group_size")

        if self.gspo_max_prompt_length < 1:
            raise ValueError("gspo_max_prompt_length must be >= 1")

        if self.gspo_max_new_tokens < 1:
            raise ValueError("gspo_max_new_tokens must be >= 1")

        if self.gspo_rollout_temperature < 0:
            raise ValueError("gspo_rollout_temperature must be non-negative")

        if not (0.0 < self.gspo_rollout_top_p <= 1.0):
            raise ValueError("gspo_rollout_top_p must be in (0, 1]")

        if self.gspo_rollout_top_k < 0:
            raise ValueError("gspo_rollout_top_k must be non-negative")

        if self.gspo_score_temperature <= 0:
            raise ValueError("gspo_score_temperature must be positive")

        if self.gspo_reward_temperature <= 0:
            raise ValueError("gspo_reward_temperature must be positive")

        if self.gspo_kl_coef < 0:
            raise ValueError("gspo_kl_coef must be non-negative")

        if self.gspo_reward_beta <= 0:
            raise ValueError("gspo_reward_beta must be positive")

        if self.gspo_reward_source not in [
            "policy_ref_margin",
            "length_penalized_margin",
            "deepmath_correctness_margin",
            "callable",
        ]:
            raise ValueError(f"Unknown gspo_reward_source: {self.gspo_reward_source}")

        if self.gspo_reward_length_penalty < 0:
            raise ValueError("gspo_reward_length_penalty must be non-negative")

        if self.gspo_deepmath_correct_bonus < 0:
            raise ValueError("gspo_deepmath_correct_bonus must be non-negative")

        if self.gspo_deepmath_wrong_penalty < 0:
            raise ValueError("gspo_deepmath_wrong_penalty must be non-negative")

        if self.gspo_reward_source == "callable" and self.training_mode == "gspo":
            if not self.gspo_reward_callable:
                raise ValueError("gspo_reward_callable must be set when using callable reward source")

        if self.gspo_reward_clip_min is not None and self.gspo_reward_clip_max is not None:
            if self.gspo_reward_clip_min > self.gspo_reward_clip_max:
                raise ValueError("gspo_reward_clip_min must be <= gspo_reward_clip_max")

        if not self.mask_per_block:
            import warnings
            warnings.warn("mask_per_block=False does not match the paper's design.", UserWarning)
    
    @classmethod
    def from_yaml(cls, path: str) -> "WeDLMTrainingConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def save_yaml(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        save_dict = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(save_dict, f, default_flow_style=False)
    
    def get_batch_seq_length(self) -> int:
        return self.max_seq_length * self.per_device_train_batch_size
    
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
            zero_config["offload_optimizer"] = {"device": device, "pin_memory": self.deepspeed_pin_memory}
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
                zero_config["offload_param"] = {"device": device, "pin_memory": self.deepspeed_pin_memory}
        
        ds_config["zero_optimization"] = zero_config
        return ds_config

