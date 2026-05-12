# coding=utf-8
"""Training configuration for GSPO (Group Sampling Policy Optimization)."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import yaml
import os


@dataclass
class GSPOTrainingConfig:
    """Configuration for GSPO on-policy training with WeDLM block diffusion."""

    # ========== Model ==========
    model_path: str = "tencent/WeDLM-8B-Base"
    trust_remote_code: bool = True

    # ========== Data ==========
    gspo_prompt_data: str = "data/prompts.jsonl"
    gspo_prompt_format: str = "messages"  # "messages" or "deepmath"
    max_seq_length: int = 2048

    # ========== GSPO Core ==========
    gspo_num_samples: int = 4                # K: number of responses per prompt
    gspo_beta: float = 0.1                   # GSPO temperature coefficient
    gspo_block_reduce: str = "mean"           # block reduction: "mean" or "sum"
    gspo_seq_reduce: str = "mean"             # sequence reduction: "mean" or "sum"
    gspo_num_mask_samples: int = 1            # MC mask samples for score estimation
    gspo_length_norm: bool = True             # enable length normalization
    gspo_ref_model_path: Optional[str] = None # null = use model_path for ref

    # ========== Reward (Phase 1: rule-based math) ==========
    gspo_reward_type: str = "math_verify"     # "math_verify" (Phase 1) / "model" (Phase 3)
    gspo_reward_model_path: Optional[str] = None  # only for reward_type="model"

    # ========== Generation ==========
    gen_max_new_tokens: int = 512
    gen_temperature: float = 1.0
    gen_top_p: float = 1.0
    gen_top_k: int = 0
    gen_wedlm_entropy_threshold: float = 0.4
    gen_wedlm_pos_penalty_factor: float = 0.02

    # ========== WeDLM Structure ==========
    block_size: int = 32
    mask_per_block: bool = True
    loss_weighting_scheme: str = "weighted"   # "weighted" or "uniform"
    mask_eps: float = 1e-8
    num_learnable_im_end: int = 0

    # ========== AR Loss (optional) ==========
    enable_ar_loss: bool = False              # typically off for on-policy
    ar_loss_weight: float = 1.0

    # ========== Attention ==========
    attention_backend: str = "magi"           # "magi" or "dense"

    # ========== Training ==========
    output_dir: str = "outputs/gspo"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1.0e-6
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # ========== Cache ==========
    rebuild_cache: bool = False

    # ========== DeepSpeed ==========
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

    # ========== Logging & Saving ==========
    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 3

    # ========== Device & Seed ==========
    bf16: bool = True
    seed: int = 42

    # ========== WandB (optional) ==========
    use_wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_team: Optional[str] = None
    wandb_group: Optional[str] = None
    wandb_host: Optional[str] = None
    wandb_key: Optional[str] = None
    
    def __post_init__(self):
        if self.loss_weighting_scheme not in ["uniform", "weighted"]:
            raise ValueError(f"Unknown loss_weighting_scheme: {self.loss_weighting_scheme}")

        if self.gspo_block_reduce not in ["mean", "sum"]:
            raise ValueError(f"Unknown gspo_block_reduce: {self.gspo_block_reduce}")

        if self.gspo_seq_reduce not in ["mean", "sum"]:
            raise ValueError(f"Unknown gspo_seq_reduce: {self.gspo_seq_reduce}")

        if self.gspo_num_mask_samples < 1:
            raise ValueError("gspo_num_mask_samples must be >= 1")

        if self.gspo_num_samples < 2:
            raise ValueError("gspo_num_samples (K) must be >= 2")

        if self.gspo_reward_type not in ["math_verify", "model"]:
            raise ValueError(f"Unknown gspo_reward_type: {self.gspo_reward_type}")

        if self.gspo_prompt_format not in ["messages", "deepmath"]:
            raise ValueError(f"Unknown gspo_prompt_format: {self.gspo_prompt_format}")

        if not self.mask_per_block:
            import warnings
            warnings.warn("mask_per_block=False does not match the paper's design.", UserWarning)

    @classmethod
    def from_yaml(cls, path: str) -> "GSPOTrainingConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def save_yaml(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        save_dict = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(save_dict, f, default_flow_style=False)

    def get_generation_config(self) -> Dict[str, Any]:
        """Return generation parameters as a dictionary."""
        return {
            "max_new_tokens": self.gen_max_new_tokens,
            "temperature": self.gen_temperature,
            "top_p": self.gen_top_p,
            "top_k": self.gen_top_k,
            "wedlm_entropy_threshold": self.gen_wedlm_entropy_threshold,
            "wedlm_pos_penalty_factor": self.gen_wedlm_pos_penalty_factor,
        }

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

