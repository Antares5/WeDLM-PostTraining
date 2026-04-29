# coding=utf-8
"""Config registry: factory to load config from YAML by training_mode."""

import yaml
from .base import BaseTrainingConfig
from .sft import SFTConfig
from .dpo import DPOConfig
from .gspo import GSPOConfig


def from_yaml(path: str) -> BaseTrainingConfig:
    """Load a training config from a YAML file.

    Dispatches to the correct subclass based on the ``training_mode`` field.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    mode = data.get("training_mode", "sft")
    if mode == "dpo":
        return DPOConfig(**data)
    elif mode == "gspo":
        return GSPOConfig(**data)
    else:
        return SFTConfig(**data)
