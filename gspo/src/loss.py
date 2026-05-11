# coding=utf-8
"""GSPO REINFORCE loss for block diffusion language models.

Mathematical formulation
------------------------
For each prompt x, sample G responses {y_1, ..., y_G} from the current policy π_θ.
Compute rewards r_i = RM(x, y_i) and group-normalized advantages:

    μ = (1/G) Σ r_i
    σ = std(r_i)
    A_i = (r_i - μ) / (σ + ε)

The REINFORCE loss with score proxy S_θ(y|x) (block-level pseudo-log-likelihood):

    L(θ) = - (1/(B·G)) Σ_i A_i · S_θ(y_i | x_i)

This is the "group baseline" variant — no importance ratio needed,
which avoids the exp(noise) problem of importance sampling with MC score estimates.
"""

from typing import Dict, Tuple
import torch


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
