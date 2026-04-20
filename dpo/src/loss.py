# coding=utf-8
"""Loss functions for WeDLM training."""

from typing import Dict, Literal, Tuple
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
        raise ValueError(f"Expected logits to have shape [T, V], got {tuple(logits.shape)}")

    if targets.dim() != 1:
        raise ValueError(f"Expected targets to have shape [T], got {tuple(targets.shape)}")

    if masked_indices.dim() != 1:
        raise ValueError(f"Expected masked_indices to have shape [T], got {tuple(masked_indices.shape)}")

    if logits.size(0) != targets.size(0) or logits.size(0) != masked_indices.size(0):
        raise ValueError("logits, targets, and masked_indices must share the same token length.")

    safe_targets = targets.clone().long()
    safe_targets[safe_targets < 0] = 0
    token_nll = F.cross_entropy(logits, safe_targets, reduction="none")
    token_logps = -token_nll
    return token_logps[masked_indices]


def compute_block_scores(
    logits: torch.Tensor,
    targets: torch.Tensor,
    masked_indices: torch.Tensor,
    p_mask: torch.Tensor,
    logical_positions: torch.Tensor,
    cum_seqlens: torch.Tensor,
    block_size: int,
    weighting_scheme: Literal["uniform", "weighted"] = "weighted",
    block_reduce: Literal["mean", "sum"] = "mean",
    seq_reduce: Literal["mean", "sum"] = "mean",
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute block-level sequence scores from masked token log-probabilities.

    This is a differentiable score function for block-level preference learning.

    Args:
        logits: Model logits of shape [T, V].
        targets: Original token ids of shape [T].
        masked_indices: Bool mask of shape [T], True means the token belongs to masked xt stream.
        p_mask: Per-token masking ratio of shape [T].
        logical_positions: Original logical positions of shape [T].
        cum_seqlens: Sequence offsets [bs + 1] over packed token dimension.
        block_size: Block size used by WeDLM masking/reordering.
        weighting_scheme: "weighted" uses 1/(p_mask + eps), "uniform" uses equal weights.
        block_reduce: Reduction inside each block, "mean" or "sum".
        seq_reduce: Reduction across blocks in each sequence, "mean" or "sum".
        eps: Small value for numerical stability.

    Returns:
        sequence_scores: Tensor of shape [bs].
        logs: Scalar metrics for debugging/monitoring.
    """
    if block_size <= 0:
        raise ValueError("block_size must be a positive integer.")

    if weighting_scheme not in ["uniform", "weighted"]:
        raise ValueError(f"Unknown weighting_scheme: {weighting_scheme}")

    if block_reduce not in ["mean", "sum"]:
        raise ValueError(f"Unknown block_reduce: {block_reduce}")

    if seq_reduce not in ["mean", "sum"]:
        raise ValueError(f"Unknown seq_reduce: {seq_reduce}")

    if logits.dim() != 2:
        raise ValueError(f"Expected logits to have shape [T, V], got {tuple(logits.shape)}")

    token_len = logits.size(0)
    for name, tensor in {
        "targets": targets,
        "masked_indices": masked_indices,
        "p_mask": p_mask,
        "logical_positions": logical_positions,
    }.items():
        if tensor.dim() != 1 or tensor.size(0) != token_len:
            raise ValueError(f"{name} must have shape [T] and match logits token length.")

    if cum_seqlens.dim() != 1 or cum_seqlens.numel() < 1:
        raise ValueError("cum_seqlens must be a 1D tensor with at least one element.")

    device = logits.device
    dtype = logits.dtype
    batch_size = cum_seqlens.numel() - 1
    if batch_size <= 0:
        empty = torch.empty((0,), device=device, dtype=dtype)
        return empty, {
            "score/mean": torch.tensor(0.0, device=device),
            "score/num_masked_tokens": torch.tensor(0.0, device=device),
            "score/num_blocks": torch.tensor(0.0, device=device),
            "score/avg_blocks_per_seq": torch.tensor(0.0, device=device),
        }

    safe_targets = targets.clone().long()
    safe_targets[safe_targets < 0] = 0
    token_nll = F.cross_entropy(logits, safe_targets, reduction="none")
    token_logps = -token_nll.to(dtype)

    weights = torch.zeros_like(token_logps, dtype=dtype)
    if weighting_scheme == "weighted":
        masked_weights = 1.0 / (p_mask[masked_indices].to(dtype) + eps)
    else:
        num_masked = int(masked_indices.sum().item())
        masked_weights = torch.ones((num_masked,), device=device, dtype=dtype)

    if masked_weights.numel() > 0:
        weights[masked_indices] = masked_weights

    sequence_scores = []
    total_blocks = torch.tensor(0.0, device=device, dtype=dtype)
    total_masked_tokens = masked_indices.sum().to(dtype)

    for sample_idx in range(batch_size):
        seq_start = int(cum_seqlens[sample_idx].item())
        seq_end = int(cum_seqlens[sample_idx + 1].item())
        if seq_end <= seq_start:
            sequence_scores.append(token_logps.sum() * 0.0)
            continue

        seq_mask = masked_indices[seq_start:seq_end]
        if not torch.any(seq_mask):
            sequence_scores.append(token_logps[seq_start:seq_end].sum() * 0.0)
            continue

        seq_logps = token_logps[seq_start:seq_end][seq_mask]
        seq_weights = weights[seq_start:seq_end][seq_mask]
        seq_positions = logical_positions[seq_start:seq_end][seq_mask]
        seq_block_ids = torch.div(seq_positions, block_size, rounding_mode="floor")

        unique_blocks = torch.unique(seq_block_ids, sorted=True)
        total_blocks = total_blocks + unique_blocks.numel()
        block_scores = []

        for block_id in unique_blocks:
            block_mask = seq_block_ids == block_id
            block_logps = seq_logps[block_mask]
            block_weights = seq_weights[block_mask]

            if block_reduce == "sum":
                block_score = (block_logps * block_weights).sum()
            else:
                denom = block_weights.sum().clamp_min(eps)
                block_score = (block_logps * block_weights).sum() / denom

            block_scores.append(block_score)

        if len(block_scores) == 0:
            sequence_scores.append(seq_logps.sum() * 0.0)
            continue

        block_scores_t = torch.stack(block_scores)
        if seq_reduce == "sum":
            sequence_scores.append(block_scores_t.sum())
        else:
            sequence_scores.append(block_scores_t.mean())

    sequence_scores_t = torch.stack(sequence_scores)
    avg_blocks = total_blocks / max(batch_size, 1)
    return sequence_scores_t, {
        "score/mean": sequence_scores_t.mean().detach(),
        "score/num_masked_tokens": total_masked_tokens.detach(),
        "score/num_blocks": total_blocks.detach(),
        "score/avg_blocks_per_seq": avg_blocks.detach(),
    }


def compute_dpo_loss(
    policy_chosen_scores: torch.Tensor,
    policy_rejected_scores: torch.Tensor,
    reference_chosen_scores: torch.Tensor,
    reference_rejected_scores: torch.Tensor,
    beta: float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute DPO loss from sequence-level scores.

    Args:
        policy_chosen_scores: Policy scores of chosen responses, shape [bs].
        policy_rejected_scores: Policy scores of rejected responses, shape [bs].
        reference_chosen_scores: Reference scores of chosen responses, shape [bs].
        reference_rejected_scores: Reference scores of rejected responses, shape [bs].
        beta: DPO temperature coefficient.

    Returns:
        loss: Scalar DPO loss.
        logs: DPO metrics.
    """
    if beta <= 0:
        raise ValueError("beta must be positive.")

    shape = policy_chosen_scores.shape
    for name, tensor in {
        "policy_rejected_scores": policy_rejected_scores,
        "reference_chosen_scores": reference_chosen_scores,
        "reference_rejected_scores": reference_rejected_scores,
    }.items():
        if tensor.shape != shape:
            raise ValueError(f"{name} shape {tuple(tensor.shape)} must match {tuple(shape)}")

    pi_logratios = policy_chosen_scores - policy_rejected_scores
    ref_logratios = reference_chosen_scores - reference_rejected_scores
    logits = beta * (pi_logratios - ref_logratios)
    losses = -F.logsigmoid(logits)
    loss = losses.mean()

    chosen_rewards = beta * (policy_chosen_scores - reference_chosen_scores).detach()
    rejected_rewards = beta * (policy_rejected_scores - reference_rejected_scores).detach()
    reward_margins = chosen_rewards - rejected_rewards
    reward_accuracies = (reward_margins > 0).float()

    return loss, {
        "dpo/loss": loss.detach(),
        "dpo/rewards_chosen": chosen_rewards.mean(),
        "dpo/rewards_rejected": rejected_rewards.mean(),
        "dpo/rewards_margin": reward_margins.mean(),
        "dpo/rewards_accuracy": reward_accuracies.mean(),
        "dpo/logits": logits.mean().detach(),
    }


def compute_mlm_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    masked_indices: torch.Tensor,
    p_mask: torch.Tensor,
    weighting_scheme: str = "weighted",
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute MLM loss on masked positions.
    
    Args:
        logits: Model output logits
        targets: Original token ids (before masking)
        masked_indices: Bool tensor indicating masked positions
        p_mask: Per-token masking ratio
        weighting_scheme: "weighted" (1/γ weighting) or "uniform"
        eps: Small value to prevent division by zero
    
    Returns:
        loss: Scalar loss tensor
        logs: Dictionary of logging metrics
    """
    device = logits.device
    num_masked = masked_indices.sum()
    
    if num_masked == 0:
        zero = torch.tensor(0.0, device=device, requires_grad=True)
        return zero, {"mlm/loss": zero.detach(), "mlm/num_tokens": torch.tensor(0, device=device)}
    
    masked_logits = logits[masked_indices]
    masked_targets = targets[masked_indices]
    per_token_loss = F.cross_entropy(masked_logits, masked_targets, reduction="none")
    
    if weighting_scheme == "weighted":
        weights = 1.0 / (p_mask[masked_indices] + eps)
        weights = weights / weights.sum()
        loss = (per_token_loss * weights).sum()
    else:
        loss = per_token_loss.mean()
    
    return loss, {
        "mlm/loss": loss.detach(),
        "mlm/num_tokens": num_masked.detach(),
    }


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
        return zero, {"ar/loss": zero.detach(), "ar/num_tokens": torch.tensor(0, device=device)}
    
    per_token_loss = F.cross_entropy(shift_logits, shift_labels, reduction="none", ignore_index=-100)
    loss = per_token_loss.sum() / num_valid
    
    return loss, {
        "ar/loss": loss.detach(),
        "ar/num_tokens": num_valid.detach(),
    }

