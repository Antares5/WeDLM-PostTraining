# coding=utf-8
"""GSPO REINFORCE loss and block-diffusion score functions."""

from typing import Dict, Tuple, Literal
import torch
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════
# Block-level score function (pseudo-log-likelihood estimator)
# ═══════════════════════════════════════════════════════════════

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

    This is the differentiable score function S_θ(y|x) used as a proxy
    for log π_θ(y|x) in the REINFORCE estimator.

    Equivalent to dpo.src.loss.compute_block_scores — self-contained copy.
    """
    if block_size <= 0:
        raise ValueError("block_size must be a positive integer.")
    if weighting_scheme not in ["uniform", "weighted"]:
        raise ValueError(f"Unknown weighting_scheme: {weighting_scheme}")
    if block_reduce not in ["mean", "sum"]:
        raise ValueError(f"Unknown block_reduce: {block_reduce}")
    if seq_reduce not in ["mean", "sum"]:
        raise ValueError(f"Unknown seq_reduce: {seq_reduce}")

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
            block_weights_ = seq_weights[block_mask]

            if block_reduce == "sum":
                block_score = (block_logps * block_weights_).sum()
            else:
                denom = block_weights_.sum().clamp_min(eps)
                block_score = (block_logps * block_weights_).sum() / denom

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


# ═══════════════════════════════════════════════════════════════
# GSPO REINFORCE loss
# ═══════════════════════════════════════════════════════════════

def compute_gspo_loss(
    scores: torch.Tensor,
    rewards: torch.Tensor,
    prompt_indices: torch.Tensor,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """GSPO REINFORCE loss with group-relative advantage normalization.

    Args:
        scores: Block-level scores S_θ(y|x), shape [N] where N = B * G.
        rewards: Reward model scores r, shape [N].
        prompt_indices: Which prompt each response belongs to, shape [N],
            values in [0, B-1].
        eps: Small constant for std normalization stability.

    Returns:
        loss: Scalar GSPO loss (mean over prompts).
        logs: Dict of scalar monitoring metrics.

    Raises:
        ValueError: If input shapes are inconsistent or group_size < 2.
    """
    if scores.dim() != 1:
        raise ValueError(f"scores must be 1D, got shape {tuple(scores.shape)}")
    if rewards.dim() != 1:
        raise ValueError(f"rewards must be 1D, got shape {tuple(rewards.shape)}")
    if prompt_indices.dim() != 1:
        raise ValueError(f"prompt_indices must be 1D, got shape {tuple(prompt_indices.shape)}")

    N = scores.size(0)
    if rewards.size(0) != N or prompt_indices.size(0) != N:
        raise ValueError(
            f"Size mismatch: scores {N}, rewards {rewards.size(0)}, "
            f"prompt_indices {prompt_indices.size(0)}"
        )

    if N == 0:
        # Empty batch → zero loss (edge case, no-op)
        device = scores.device
        dtype = scores.dtype
        zero = torch.tensor(0.0, device=device, dtype=dtype)
        return zero, {
            "gspo/loss": zero,
            "gspo/adv_mean": zero,
            "gspo/adv_std": zero,
            "gspo/adv_max": zero,
            "gspo/adv_min": zero,
            "gspo/score_mean": zero,
            "gspo/score_std": zero,
            "gspo/reward_mean": zero,
            "gspo/reward_std": zero,
            "gspo/score_reward_acc": torch.tensor(0.0, device=device),
        }

    B = int(prompt_indices.max().item()) + 1
    if B <= 0:
        raise ValueError("No prompts found (prompt_indices is empty or zero-only)")

    device = scores.device
    dtype = scores.dtype

    # Cast rewards to match scores dtype
    rewards = rewards.to(dtype=dtype)

    total_loss = torch.tensor(0.0, device=device, dtype=dtype)
    all_advantages: list[torch.Tensor] = []

    for b in range(B):
        mask = prompt_indices == b
        group_rewards = rewards[mask]
        group_scores = scores[mask]
        group_size = mask.sum().item()

        if group_size < 2:
            # Edge case: single response per prompt — skip group normalization,
            # use reward itself as a scalar signal (no baseline subtraction).
            # This is suboptimal but keeps training from crashing.
            advantage = group_rewards - group_rewards  # zero advantage → no update
            all_advantages.append(advantage)
            continue

        # Group-normalized advantage
        mu = group_rewards.mean()
        sigma = group_rewards.std() + eps
        advantages = (group_rewards - mu) / sigma

        # REINFORCE: L_b = -mean(A_i * S_i)
        # The gradient of -A_i * S_i w.r.t. θ is -A_i * ∇S_i(θ).
        # Since A_i is detached (no grad through advantage), this is a valid
        # policy gradient estimator.
        group_loss = -(advantages.detach() * group_scores).mean()
        total_loss = total_loss + group_loss

        all_advantages.append(advantages)

    total_loss = total_loss / B

    # Monitoring metrics
    all_adv = torch.cat(all_advantages)
    reward_margin = torch.tensor(0.0, device=device)
    # Compute pair-wise reward accuracy within each group
    correct = 0
    total_pairs = 0
    for b in range(B):
        mask = prompt_indices == b
        group_r = rewards[mask]
        g = group_r.size(0)
        if g >= 2:
            for i in range(g):
                for j in range(i + 1, g):
                    # r_i > r_j should imply higher score (on average after training)
                    if (group_r[i] > group_r[j]) and (scores[mask][i] > scores[mask][j]):
                        correct += 1
                    elif (group_r[i] < group_r[j]) and (scores[mask][i] < scores[mask][j]):
                        correct += 1
                    total_pairs += 1
    score_acc = correct / max(total_pairs, 1)

    logs: Dict[str, torch.Tensor] = {
        "gspo/loss": total_loss.detach(),
        "gspo/adv_mean": all_adv.mean(),
        "gspo/adv_std": all_adv.std(),
        "gspo/adv_max": all_adv.max(),
        "gspo/adv_min": all_adv.min(),
        "gspo/score_mean": scores.mean().detach(),
        "gspo/score_std": scores.std().detach(),
        "gspo/reward_mean": rewards.mean(),
        "gspo/reward_std": rewards.std(),
        "gspo/score_reward_acc": torch.tensor(score_acc, device=device),
    }

    return total_loss, logs
