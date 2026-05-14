# coding=utf-8
"""Loss functions for GSPO on-policy training."""

from typing import Dict, Literal, Tuple, Optional
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


def compute_gspo_loss(
    policy_scores: torch.Tensor,
    reference_scores: torch.Tensor,
    rewards: torch.Tensor,
    beta: float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute GSPO (Group Sampling Policy Optimization) loss.

    Uses the best-of-K response as the "chosen" signal and the
    log-mean-exp of the remaining K-1 responses as the "rejected" signal.

    Loss form (Path A from the implementation plan):
        L = -log σ(β · [(s_θ(y_best) - s_old(y_best))
                        - log((1/(K-1)) · Σ_{j≠best} exp(s_θ(y_j) - s_old(y_j)))])

    Args:
        policy_scores: Policy model block scores, shape [K].
        reference_scores: Reference model block scores, shape [K].
        rewards: Binary/continuous rewards, shape [K].
        beta: GSPO temperature coefficient.

    Returns:
        loss: Scalar GSPO loss.
        logs: Dictionary of GSPO metrics.
    """
    if beta <= 0:
        raise ValueError("beta must be positive")

    K = policy_scores.size(0)
    if K < 2:
        # With K < 2, we cannot do group contrastive loss
        device = policy_scores.device
        zero = torch.tensor(0.0, device=device, requires_grad=True)
        return zero, {
            "gspo/loss": torch.tensor(0.0, device=device),
            "gspo/rewards_chosen": torch.tensor(0.0, device=device),
            "gspo/rewards_rejected": torch.tensor(0.0, device=device),
            "gspo/rewards_margin": torch.tensor(0.0, device=device),
            "gspo/rewards_accuracy": torch.tensor(0.0, device=device),
            "gspo/logits": torch.tensor(0.0, device=device),
            "gspo/num_correct": torch.tensor(0, device=device),
        }

    # Validate shapes
    if reference_scores.shape != policy_scores.shape:
        raise ValueError(
            f"Shape mismatch: policy_scores {tuple(policy_scores.shape)} vs "
            f"reference_scores {tuple(reference_scores.shape)}"
        )
    if rewards.shape != policy_scores.shape:
        raise ValueError(
            f"Shape mismatch: policy_scores {tuple(policy_scores.shape)} vs "
            f"rewards {tuple(rewards.shape)}"
        )

    device = policy_scores.device

    # Ensure rewards are on the same device (math rewards come from CPU)
    rewards = rewards.to(device)

    # Compute implied rewards from the difference between policy and reference scores
    pi_diff = policy_scores - reference_scores  # [K]

    # Find the best response (highest reward)
    best_idx = torch.argmax(rewards)
    best = pi_diff[best_idx]

    # Log-mean-exp of other responses
    others_mask = torch.ones(K, dtype=torch.bool, device=device)
    others_mask[best_idx] = False
    others = pi_diff[others_mask]  # [K-1]

    if others.numel() == 0:
        # All rewards equal, no contrast possible
        zero = torch.tensor(0.0, device=device, requires_grad=True)
        return zero, {
            "gspo/loss": torch.tensor(0.0, device=device),
            "gspo/rewards_chosen": best.detach(),
            "gspo/rewards_rejected": torch.tensor(0.0, device=device),
            "gspo/rewards_margin": torch.tensor(0.0, device=device),
            "gspo/rewards_accuracy": torch.tensor(0.0, device=device),
            "gspo/logits": torch.tensor(0.0, device=device),
            "gspo/num_correct": torch.tensor(int((rewards > 0.5).sum().item()), device=device),
        }

    # log-mean-exp: log(mean(exp(others)))
    others_logmeanexp = torch.logsumexp(others, dim=0) - torch.log(
        torch.tensor(others.numel(), dtype=others.dtype, device=device)
    )

    # GSPO logits
    logits = beta * (best - others_logmeanexp)
    loss = -F.logsigmoid(logits)

    # Logging metrics
    chosen_reward = rewards[best_idx].detach()
    rejected_rewards_mean = rewards[others_mask].mean().detach()
    reward_margin = chosen_reward - rejected_rewards_mean

    return loss, {
        "gspo/loss": loss.detach(),
        "gspo/rewards_chosen": chosen_reward,
        "gspo/rewards_rejected": rejected_rewards_mean,
        "gspo/rewards_margin": reward_margin,
        "gspo/rewards_accuracy": (reward_margin > 0).float(),
        "gspo/logits": logits.detach(),
        "gspo/num_correct": torch.tensor(int((rewards > 0.5).sum().item()), device=device),
        "gspo/best_score": policy_scores[best_idx].detach(),
        "gspo/best_ref_score": reference_scores[best_idx].detach(),
        "gspo/others_mean_score": policy_scores[others_mask].mean().detach(),
        "gspo/others_mean_ref_score": reference_scores[others_mask].mean().detach(),
    }


def compute_gspo_coefficients(
    policy_scores: torch.Tensor,
    reference_scores: torch.Tensor,
    rewards: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """Compute per-response gradient coefficients for GSPO loss.

    Instead of computing the full GSPO loss, this returns the coefficient
    ∂L/∂(policy_score_i) for each response i. These coefficients can be
    multiplied with the differentiable policy scores for per-branch backward.

    Args:
        policy_scores: [K] policy block scores (no-grad).
        reference_scores: [K] reference block scores.
        rewards: [K] binary/continuous rewards.
        beta: GSPO temperature coefficient.

    Returns:
        coefficients: [K] tensor, where coeff[i] = ∂L/∂(policy_score_i).
    """
    K = policy_scores.size(0)
    device = policy_scores.device

    if K < 2:
        return torch.zeros(K, device=device, dtype=policy_scores.dtype)

    # Ensure rewards are on the same device (math rewards come from CPU)
    rewards = rewards.to(device)

    pi_diff = policy_scores - reference_scores  # [K]

    best_idx = torch.argmax(rewards)
    others_mask = torch.ones(K, dtype=torch.bool, device=device)
    others_mask[best_idx] = False

    if others_mask.sum() == 0:
        return torch.zeros(K, device=device, dtype=pi_diff.dtype)

    best = pi_diff[best_idx]
    others = pi_diff[others_mask]  # [K-1]

    # log-mean-exp of others
    others_logmeanexp = torch.logsumexp(others, dim=0) - torch.log(
        torch.tensor(others.numel(), dtype=others.dtype, device=device)
    )

    # Use input dtype consistently to avoid float32 promotion from Python scalars
    beta_t = torch.tensor(beta, dtype=pi_diff.dtype, device=device)
    logits = beta_t * (best - others_logmeanexp)
    sigmoid_z = torch.sigmoid(logits)

    # dL/d(best) = -beta * (1 - sigmoid(logits))
    one = torch.tensor(1.0, dtype=pi_diff.dtype, device=device)
    coeff_best = -beta_t * (one - sigmoid_z)

    # dL/d(other_j) = beta * (1 - sigmoid(logits)) * softmax(other_j) / (K-1)
    # where softmax is over the others
    others_softmax = F.softmax(others, dim=0)
    coeff_others = beta_t * (one - sigmoid_z) * others_softmax

    # Build full coefficient tensor (match input dtype)
    coefficients = torch.zeros(K, device=device, dtype=pi_diff.dtype)
    coefficients[best_idx] = coeff_best
    coefficients[others_mask] = coeff_others

    return coefficients


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
    active_mask = shift_labels != -100
    active_logits = shift_logits[active_mask]
    active_labels = shift_labels[active_mask]

    if active_labels.numel() == 0:
        zero = torch.tensor(0.0, device=device, requires_grad=True)
        return zero, {"ar/loss": zero.detach(), "ar/num_tokens": torch.tensor(0, device=device)}

    loss = F.cross_entropy(active_logits, active_labels)
    return loss, {
        "ar/loss": loss.detach(),
        "ar/num_tokens": torch.tensor(active_labels.numel(), device=device),
    }


def compute_gspo_coefficients_with_kl(
    policy_scores: torch.Tensor,
    reference_scores: torch.Tensor,
    rewards: torch.Tensor,
    beta: float = 0.1,
    kl_coef: float = 0.0,
) -> torch.Tensor:
    """Compute per-response gradient coefficients for GSPO loss with optional KL penalty.

    Extends compute_gspo_coefficients with a KL divergence penalty between
    policy and reference score distributions (Path C from the implementation plan).

    The total coefficient for response i is:
        coeff[i] = coeff_gspo[i] + coeff_kl[i]

    where:
        coeff_gspo[i] = ∂L_gspo/∂(policy_score_i)   (same as compute_gspo_coefficients)
        coeff_kl[i]   = kl_coef * 2 * (policy_score_i - ref_score_i) / K
                         (gradient of MSE between policy and ref scores)

    Args:
        policy_scores: [K] policy block scores (no-grad).
        reference_scores: [K] reference block scores.
        rewards: [K] binary/continuous rewards.
        beta: GSPO temperature coefficient.
        kl_coef: KL penalty coefficient (α in Path C). 0 = disabled.

    Returns:
        coefficients: [K] tensor, where coeff[i] = ∂L_total/∂(policy_score_i).
    """
    # Get base GSPO coefficients
    coeffs = compute_gspo_coefficients(
        policy_scores, reference_scores, rewards, beta
    )

    # Add KL penalty coefficients if enabled
    if kl_coef > 0:
        K = policy_scores.size(0)
        device = policy_scores.device
        dtype = policy_scores.dtype

        # KL penalty: L_kl = kl_coef * mean((policy_scores - ref_scores)^2)
        # ∂L_kl/∂(policy_score_i) = kl_coef * 2 * (policy_score_i - ref_score_i) / K
        kl_grad = (2.0 * kl_coef / max(K, 1)) * (policy_scores - reference_scores.to(device))
        coeffs = coeffs + kl_grad

    return coeffs


def compute_kl_penalty(
    policy_scores: torch.Tensor,
    reference_scores: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute KL divergence penalty between policy and reference score distributions.

    Uses MSE as a practical proxy for KL divergence in block score space:
        L_kl = mean((policy_scores - reference_scores)^2)

    Args:
        policy_scores: [K] policy block scores.
        reference_scores: [K] reference block scores.

    Returns:
        kl_loss: Scalar KL penalty loss.
        kl_per_sample: [K] per-sample KL values for logging.
    """
    device = policy_scores.device
    ref = reference_scores.to(device)
    diff = policy_scores - ref
    kl_per_sample = diff * diff  # [K], squared difference per sample
    kl_loss = kl_per_sample.mean()
    return kl_loss, kl_per_sample.detach()
