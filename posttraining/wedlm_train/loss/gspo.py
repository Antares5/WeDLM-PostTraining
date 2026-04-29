# coding=utf-8
"""GSPO loss function.
Migrated from dpo/src/loss.py (compute_gspo_loss)."""

from typing import Dict, Optional, Tuple
import torch
import torch.nn.functional as F


def compute_gspo_loss(
    policy_scores: torch.Tensor,
    reference_scores: torch.Tensor,
    group_ids: torch.Tensor,
    rewards: Optional[torch.Tensor] = None,
    score_temperature: float = 1.0,
    reward_temperature: float = 1.0,
    ref_alpha: float = 1.0,
    kl_coef: float = 0.0,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute Group Sequence Preference Optimization (GSPO) loss.

    This objective uses online group rollouts and matches the policy group
    distribution to a reward-induced target distribution:

        p_theta(i|x, G) = softmax((s_theta(i) - ref_alpha * s_ref(i)) / score_temperature)
        q(i|x, G)       = softmax(r(i) / reward_temperature)
        L_group         = -sum_i q_i * log p_i

    Optionally, a KL regularizer can be added between p_theta and a reference
    distribution derived from reference scores.

    Args:
        policy_scores: Policy sequence scores, shape [N].
        reference_scores: Reference sequence scores, shape [N].
        group_ids: Group id per sample, shape [N].
        rewards: Optional reward per sample, shape [N]. If None, uses
            reward = policy_scores - reference_scores (detached).
        score_temperature: Temperature for policy group distribution.
        reward_temperature: Temperature for target reward distribution.
        ref_alpha: Coefficient for reference-score adjustment in policy logits.
        kl_coef: Coefficient for KL(p_theta || p_ref).
        eps: Numerical stability constant.

    Returns:
        loss: Scalar GSPO loss.
        logs: Metrics for monitoring.
    """
    if policy_scores.dim() != 1:
        raise ValueError(
            f"policy_scores must have shape [N], got {tuple(policy_scores.shape)}"
        )

    if reference_scores.shape != policy_scores.shape:
        raise ValueError(
            f"reference_scores shape {tuple(reference_scores.shape)} must match {tuple(policy_scores.shape)}"
        )

    if group_ids.shape != policy_scores.shape:
        raise ValueError(
            f"group_ids shape {tuple(group_ids.shape)} must match {tuple(policy_scores.shape)}"
        )

    if rewards is not None and rewards.shape != policy_scores.shape:
        raise ValueError(
            f"rewards shape {tuple(rewards.shape)} must match {tuple(policy_scores.shape)}"
        )

    if score_temperature <= 0:
        raise ValueError("score_temperature must be positive")

    if reward_temperature <= 0:
        raise ValueError("reward_temperature must be positive")

    if kl_coef < 0:
        raise ValueError("kl_coef must be non-negative")

    device = policy_scores.device
    dtype = policy_scores.dtype
    unique_groups = torch.unique(group_ids, sorted=True)

    group_losses = []
    group_kls = []
    reward_means = []
    target_entropies = []
    policy_entropies = []
    top1_align = []
    group_sizes = []

    for gid in unique_groups:
        idx = torch.where(group_ids == gid)[0]
        group_size = int(idx.numel())
        if group_size < 2:
            continue

        group_policy_scores = policy_scores[idx]
        group_reference_scores = reference_scores[idx].detach()

        policy_logits = (
            group_policy_scores - ref_alpha * group_reference_scores
        ) / score_temperature
        log_policy_probs = F.log_softmax(policy_logits, dim=0)
        policy_probs = log_policy_probs.exp()

        if rewards is None:
            group_rewards = group_policy_scores.detach() - group_reference_scores
        else:
            group_rewards = rewards[idx].detach()

        target_logits = group_rewards / reward_temperature
        target_probs = F.softmax(target_logits, dim=0)

        ce_loss = -(target_probs * log_policy_probs).sum()
        loss = ce_loss

        if kl_coef > 0:
            ref_logits = group_reference_scores / score_temperature
            log_ref_probs = F.log_softmax(ref_logits, dim=0)
            kl = (policy_probs * (log_policy_probs - log_ref_probs)).sum()
            loss = loss + kl_coef * kl
            group_kls.append(kl.detach())

        group_losses.append(loss)
        reward_means.append(group_rewards.mean())

        target_entropy = -(target_probs * torch.log(target_probs.clamp_min(eps))).sum()
        policy_entropy = -(policy_probs * log_policy_probs).sum()
        target_entropies.append(target_entropy.detach())
        policy_entropies.append(policy_entropy.detach())

        align = (torch.argmax(policy_probs) == torch.argmax(target_probs)).to(dtype=torch.float32)
        top1_align.append(align)
        group_sizes.append(torch.tensor(float(group_size), device=device))

    if len(group_losses) == 0:
        zero = policy_scores.sum() * 0.0
        return zero, {
            "gspo/loss": zero.detach(),
            "gspo/num_valid_groups": torch.tensor(0.0, device=device),
            "gspo/avg_group_size": torch.tensor(0.0, device=device),
            "gspo/target_entropy": torch.tensor(0.0, device=device),
            "gspo/policy_entropy": torch.tensor(0.0, device=device),
            "gspo/top1_align": torch.tensor(0.0, device=device),
            "gspo/reward_mean": torch.tensor(0.0, device=device),
            "gspo/kl": torch.tensor(0.0, device=device),
        }

    loss = torch.stack(group_losses).mean()

    kl_value = (
        torch.stack(group_kls).mean()
        if len(group_kls) > 0
        else torch.tensor(0.0, device=device, dtype=dtype)
    )

    return loss, {
        "gspo/loss": loss.detach(),
        "gspo/num_valid_groups": torch.tensor(float(len(group_losses)), device=device),
        "gspo/avg_group_size": torch.stack(group_sizes).mean().detach(),
        "gspo/target_entropy": torch.stack(target_entropies).mean().detach(),
        "gspo/policy_entropy": torch.stack(policy_entropies).mean().detach(),
        "gspo/top1_align": torch.stack(top1_align).mean().detach(),
        "gspo/reward_mean": torch.stack(reward_means).mean().detach(),
        "gspo/kl": kl_value.detach(),
    }
