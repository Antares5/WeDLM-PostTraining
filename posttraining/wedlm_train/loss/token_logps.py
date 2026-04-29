# coding=utf-8
"""Compute masked token log-probabilities.
Migrated from dpo/src/loss.py (compute_masked_token_logps)."""

import torch
import torch.nn.functional as F


def compute_masked_token_logps(
    logits: torch.Tensor,
    targets: torch.Tensor,
    masked_indices: torch.Tensor,
) -> torch.Tensor:
    """Compute token log-probabilities on masked positions.

    Args:
        logits: Model logits of shape [T, V].
        targets: Token ids of shape [T].
        masked_indices: Bool mask of shape [T], True means score this token.

    Returns:
        A 1D tensor containing log-probabilities of masked tokens.
    """
    if logits.dim() != 2:
        raise ValueError(
            f"Expected logits to have shape [T, V], got {tuple(logits.shape)}"
        )

    if targets.dim() != 1:
        raise ValueError(
            f"Expected targets to have shape [T], got {tuple(targets.shape)}"
        )

    if masked_indices.dim() != 1:
        raise ValueError(
            f"Expected masked_indices to have shape [T], got {tuple(masked_indices.shape)}"
        )

    if logits.size(0) != targets.size(0) or logits.size(0) != masked_indices.size(0):
        raise ValueError(
            "logits, targets, and masked_indices must share the same token length."
        )

    safe_targets = targets.clone().long()
    safe_targets[safe_targets < 0] = 0
    token_nll = F.cross_entropy(logits, safe_targets, reduction="none")
    token_logps = -token_nll
    return token_logps[masked_indices]
