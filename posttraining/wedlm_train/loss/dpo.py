# coding=utf-8
"""DPO loss function.
Migrated from dpo/src/loss.py (compute_dpo_loss)."""

from typing import Dict, Tuple
import torch
import torch.nn.functional as F


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
            raise ValueError(
                f"{name} shape {tuple(tensor.shape)} must match {tuple(shape)}"
            )

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
