# coding=utf-8
"""AR (auto-regressive) loss function.
Migrated from dpo/src/loss.py (compute_ar_loss)."""

from typing import Dict, Tuple
import torch
import torch.nn.functional as F


def compute_ar_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute standard autoregressive loss.

    Args:
        logits: Model output logits
        labels: Target labels (-100 for ignored positions)

    Returns:
        loss: Scalar loss tensor
        logs: Dictionary of logging metrics
    """
    device = logits.device

    shift_logits = logits[:-1]
    shift_labels = labels[1:]
    valid_mask = shift_labels != -100
    num_valid = valid_mask.sum()

    if num_valid == 0:
        zero = torch.tensor(0.0, device=device, requires_grad=True)
        return zero, {
            "ar/loss": zero.detach(),
            "ar/num_tokens": torch.tensor(0, device=device),
        }

    per_token_loss = F.cross_entropy(
        shift_logits, shift_labels, reduction="none", ignore_index=-100
    )
    loss = per_token_loss.sum() / num_valid

    return loss, {
        "ar/loss": loss.detach(),
        "ar/num_tokens": num_valid.detach(),
    }
