# coding=utf-8
"""GRPO loss functions and reward utilities for GSPO training.

Core loss::

    loss, logs = compute_grpo_loss(
        s_policy, s_old, s_ref, rewards, group_size, clip_epsilon, ...
    )

where ``s_policy`` carries gradient and ``s_old``/``s_ref`` are detached.
"""

from __future__ import annotations

import re
import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Group Advantage Normalisation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_group_advantage(
    rewards: torch.Tensor,          # [B*G]
    group_size: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Normalise rewards within each prompt group to produce advantages.

    For each group of ``group_size`` consecutive elements::

        A_j = (R_j - mean(R)) / (std(R) + eps)

    Args:
        rewards: Flat tensor of per-completion rewards, [B×G].
        group_size: G, number of completions per prompt.
        eps: Numerical stabiliser.

    Returns:
        advantages: [B×G] zero-mean, unit-variance within each group.
    """
    total = rewards.numel()
    if total % group_size != 0:
        raise ValueError(
            f"rewards size ({total}) must be divisible by group_size ({group_size})"
        )

    rewards_2d = rewards.view(-1, group_size)            # [B, G]
    mean_r = rewards_2d.mean(dim=1, keepdim=True)        # [B, 1]
    std_r = rewards_2d.std(dim=1, keepdim=True)          # [B, 1]
    advantages_2d = (rewards_2d - mean_r) / (std_r + eps)  # [B, G]
    return advantages_2d.view(-1)                         # [B*G]


# ═══════════════════════════════════════════════════════════════════════════════
# KL Estimators
# ═══════════════════════════════════════════════════════════════════════════════

def compute_k3_kl(
    s_policy: torch.Tensor,
    s_ref: torch.Tensor,
) -> torch.Tensor:
    """k3 estimator for KL(π_θ || π_ref).

    Uses the unbiased estimator::

        KL ≈ exp(s_ref - s_policy) - (s_ref - s_policy) - 1

    where s_policy and s_ref are block-score proxies for log π.

    Args:
        s_policy: [N] policy block scores.
        s_ref: [N] reference block scores (detached).

    Returns:
        kl: scalar mean KL estimate (non-negative in expectation).
    """
    diff = s_ref - s_policy
    return (torch.exp(diff) - diff - 1.0).mean()


def compute_reverse_kl(
    s_policy: torch.Tensor,
    s_ref: torch.Tensor,
) -> torch.Tensor:
    """Simple reverse-KL proxy: mean(s_ref - s_policy)."""
    return (s_ref - s_policy).mean()


# ═══════════════════════════════════════════════════════════════════════════════
# GRPO Clipped Surrogate Loss
# ═══════════════════════════════════════════════════════════════════════════════

def compute_grpo_loss(
    s_policy: torch.Tensor,         # [B*G] with grad
    s_old: torch.Tensor,            # [B*G] detached
    s_ref: Optional[torch.Tensor],  # [B*G] detached (or None)
    rewards: torch.Tensor,          # [B*G]
    group_size: int,                # G
    clip_epsilon: float = 0.2,      # ε
    kl_beta: float = 0.0,           # β_KL (0 = no KL penalty)
    kl_estimator: str = "k3",       # "k3" | "reverse_kl" | "none"
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute GRPO clipped surrogate loss with optional KL penalty.

    Formulae:

        ρ_j  = exp(s_policy_j - s_old_j)             [importance ratio]
        A_j  = group_advantage(R_j)                   [normalised]
        L_j  = -min(ρ_j·A_j, clip(ρ_j, 1-ε, 1+ε)·A_j)
        L    = mean(L_j) + β_KL · KL(s_policy, s_ref)

    Args:
        s_policy: Current policy block scores (grad-enabled).
        s_old: Old-policy block scores (detached).
        s_ref: Reference block scores (detached).  May be None if KL not used.
        rewards: Per-completion scalar rewards.
        group_size: Number of completions per prompt group.
        clip_epsilon: PPO clipping range.
        kl_beta: KL penalty coefficient.  Set to 0 to disable.
        kl_estimator: Which KL estimator ("k3", "reverse_kl", "none").

    Returns:
        loss: Scalar GSPO loss (differentiable).
        logs: Dict of detached scalar metrics for monitoring.
    """
    if clip_epsilon <= 0:
        raise ValueError("clip_epsilon must be positive")

    BxG = s_policy.numel()
    for name, t in [("s_old", s_old), ("rewards", rewards)]:
        if t.numel() != BxG:
            raise ValueError(f"{name}.numel()={t.numel()} != {BxG}")
    if s_ref is not None and s_ref.numel() != BxG:
        raise ValueError(f"s_ref.numel()={s_ref.numel()} != {BxG}")
    if BxG % group_size != 0:
        raise ValueError(f"B×G={BxG} not divisible by group_size={group_size}")

    device = s_policy.device
    dtype = s_policy.dtype

    # ── 1. Importance ratio ρ ───────────────────────────────────────
    log_ratio = s_policy - s_old                      # [B*G]
    rho = torch.exp(log_ratio)                        # [B*G]

    # ── 2. Group advantage ──────────────────────────────────────────
    advantages = compute_group_advantage(rewards, group_size)  # [B*G]

    # ── 3. Clipped surrogate ────────────────────────────────────────
    surr1 = rho * advantages                           # unclipped
    surr2 = torch.clamp(rho, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    per_sample_loss = -torch.min(surr1, surr2)        # [B*G]
    grpo_loss = per_sample_loss.mean()

    # ── 4. KL penalty (optional) ────────────────────────────────────
    kl_loss = torch.tensor(0.0, device=device, dtype=dtype)
    if kl_beta > 0 and s_ref is not None:
        if kl_estimator == "k3":
            kl_loss = compute_k3_kl(s_policy, s_ref)
        elif kl_estimator == "reverse_kl":
            kl_loss = compute_reverse_kl(s_policy, s_ref)
        elif kl_estimator != "none":
            raise ValueError(f"Unknown kl_estimator: {kl_estimator}")

    total_loss = grpo_loss + kl_beta * kl_loss

    # ── 5. Metrics ──────────────────────────────────────────────────
    with torch.no_grad():
        clip_mask_low = rho < (1.0 - clip_epsilon)
        clip_mask_high = rho > (1.0 + clip_epsilon)
        clip_frac = (clip_mask_low | clip_mask_high).float().mean()

        # Approximate KL from importance ratio: rho - log(rho) - 1
        approx_kl = (rho - log_ratio - 1.0).mean()

        logs: Dict[str, torch.Tensor] = {
            "grpo/loss": grpo_loss.detach(),
            "grpo/kl_loss": kl_loss.detach() if torch.is_tensor(kl_loss) else kl_loss,
            "grpo/total_loss": total_loss.detach(),
            "grpo/approx_kl": approx_kl.detach(),
            "grpo/clip_frac": clip_frac.detach(),
            "grpo/mean_rho": rho.mean().detach(),
            "grpo/mean_log_ratio": log_ratio.mean().detach(),
            "grpo/mean_advantage": advantages.mean().detach(),
            "grpo/std_advantage": advantages.std().detach(),
            "grpo/reward_mean": rewards.mean().detach(),
            "grpo/reward_std": rewards.std().detach(),
            "grpo/reward_min": rewards.min().detach(),
            "grpo/reward_max": rewards.max().detach(),
        }

    return total_loss, logs


# ═══════════════════════════════════════════════════════════════════════════════
# Reward Functions
# ═══════════════════════════════════════════════════════════════════════════════

def _strip_latex_delimiters(s: str) -> str:
    """Strip LaTeX math-mode delimiters from a ground-truth string.

    Handles:
      - ``$...$``  (inline math)
      - ``$$...$$`` (display math)
      - ``\\(...\\)`` (LaTeX inline)
      - ``\\[...\\]`` (LaTeX display)

    Returns the inner content with outer whitespace trimmed.
    """
    s = s.strip()
    # $$ ... $$
    if s.startswith("$$") and s.endswith("$$"):
        s = s[2:-2].strip()
    # \[ ... \]
    elif s.startswith("\\[") and s.endswith("\\]"):
        s = s[2:-2].strip()
    # $ ... $ (but not $$$)
    elif s.startswith("$") and s.endswith("$") and not s.startswith("$$"):
        s = s[1:-1].strip()
    # \( ... \)
    elif s.startswith("\\(") and s.endswith("\\)"):
        s = s[2:-2].strip()
    return s


def extract_gsm8k_answer(text: str) -> Optional[str]:
    """Extract the final answer from a GSM8K-style completion.

    Looks for ``#### <number>`` at the end of the text.  Returns the
    numeric string after the last ``####``.
    """
    match = re.findall(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if match:
        return match[-1].replace(",", "")
    return None


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract answer from \\boxed{...} (common in MATH dataset)."""
    match = re.findall(r"\\boxed\{([^}]+)\}", text)
    if match:
        return match[-1].strip()
    return None


def normalize_number(s: str) -> Optional[float]:
    """Try to parse a string as a number, returning float or None."""
    try:
        return float(s.replace(",", "").replace(" ", ""))
    except (ValueError, TypeError):
        return None


def math_reward(
    completion_text: str,
    ground_truth: str,
    tolerance: float = 1e-3,
) -> float:
    """Rule-based reward for math completions.

    Extraction priority:
    1. ``####`` (GSM8K)
    2. ``\\boxed{...}`` (MATH)
    3. Last number in the text

    Returns 1.0 if the extracted answer matches ground truth (numeric
    comparison with tolerance), 0.0 otherwise.

    Args:
        completion_text: Decoded completion string.
        ground_truth: Expected answer string.
        tolerance: Relative tolerance for float comparison.
    """
    # ── Normalise ground truth ──────────────────────────────────
    gt_stripped = _strip_latex_delimiters(ground_truth)

    # Try GSM8K format first.
    pred = extract_gsm8k_answer(completion_text)
    if pred is None:
        pred = extract_boxed_answer(completion_text)
    if pred is None:
        # Fallback: last numeric token in the text.
        nums = re.findall(r"-?[\d,]+(?:\.\d+)?", completion_text)
        pred = nums[-1] if nums else None

    if pred is None:
        return 0.0

    pred_num = normalize_number(pred)
    gt_num = normalize_number(gt_stripped)

    if pred_num is not None and gt_num is not None:
        if abs(gt_num) < 1e-12:
            return 1.0 if abs(pred_num) < tolerance else 0.0
        rel_err = abs(pred_num - gt_num) / (abs(gt_num) + 1e-8)
        return 1.0 if rel_err < tolerance else 0.0

    # String comparison fallback.
    pred_clean = pred.strip().lower().replace(",", "")
    gt_clean = gt_stripped.strip().lower().replace(",", "")
    return 1.0 if pred_clean == gt_clean else 0.0


def deepmath_reward(
    completion_text: str,
    ground_truth: str,
    tolerance: float = 1e-3,
) -> float:
    """Reward function for DeepMath-style completions.

    DeepMath questions have diverse answer types:
    - Yes/No binary questions
    - Numeric computation results
    - Short text answers

    This function:
    1. Extracts the **final answer** from the completion (looking for
       answer markers like "Answer:", "Therefore", etc.)
    2. Compares case-insensitively for yes/no, numerically for numbers.
    """
    # ── Normalise ground truth ──────────────────────────────────
    gt_stripped = _strip_latex_delimiters(ground_truth)

    # ── 1. Extract final answer from completion ──────────────────
    pred = _extract_deepmath_answer(completion_text)
    if pred is None:
        return 0.0

    # ── 2. Clean & compare ──────────────────────────────────────
    pred_clean = pred.strip().lower()
    gt_clean = gt_stripped.strip().lower()

    # Exact match after cleaning.
    if pred_clean == gt_clean:
        return 1.0

    # Yes/No variants.
    yes_variants = {"yes", "y", "true", "t", "1"}
    no_variants = {"no", "n", "false", "f", "0"}
    if gt_clean in yes_variants and pred_clean in yes_variants:
        return 1.0
    if gt_clean in no_variants and pred_clean in no_variants:
        return 1.0

    # Numeric comparison (with comma removal, unit stripping).
    pred_num = normalize_number(pred_clean.replace(",", ""))
    gt_num = normalize_number(gt_clean.replace(",", ""))
    if pred_num is not None and gt_num is not None:
        if abs(gt_num) < 1e-12:
            return 1.0 if abs(pred_num) < tolerance else 0.0
        rel_err = abs(pred_num - gt_num) / (abs(gt_num) + 1e-8)
        return 1.0 if rel_err < tolerance else 0.0

    return 0.0


def _extract_deepmath_answer(text: str) -> Optional[str]:
    """Extract the final answer from a DeepMath-style completion.

    Strategy (ordered by priority):
    1. Explicit markers: ``####``, ``\\boxed{}``, ``Answer:``, ``answer is``
    2. "the answer is/must be/would be X" (multi-word)
    3. Last standalone Yes/No or number in the text (fallback)
    4. Last non-empty line (final fallback)
    """
    # 1. #### marker.
    m = re.findall(r"####\s*(.+)", text)
    if m:
        return m[-1].strip()

    # 2. \boxed{...}
    m = re.findall(r"\\boxed\{([^}]+)\}", text)
    if m:
        return m[-1].strip()

    # 3. Answer: / answer: — capture everything after colon until EOL.
    m = re.findall(r"(?i)answer\s*:\s*([^\n]+)", text)
    if m:
        return m[-1].strip().rstrip(".")

    # 4. "the answer is/must be/would be/should be X" — multi-word aware.
    m = re.findall(
        r"(?i)the\s+answer\s+(?:is|must\s+be|would\s+be|should\s+be)\s+"
        r"(.+?)(?:\.\s*(?:$|\n)|$)",
        text,
    )
    if m:
        return m[-1].strip()

    # 5. "Therefore/Thus/Hence the answer ..." patterns.
    m = re.findall(
        r"(?i)(?:therefore|thus|hence|so|finally)[,.]?\s+(?:the\s+)?(?:answer|result|value)\s+(?:is|=)\s+"
        r"(.+?)(?:\.\s*(?:$|\n)|$)",
        text,
    )
    if m:
        return m[-1].strip()

    # 6. Fallback: last standalone Yes/No/true/false or number.
    yn = re.findall(r"\b(yes|no|true|false)\b", text, re.IGNORECASE)
    nums = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    if yn:
        return yn[-1].strip()
    if nums:
        return nums[-1].strip()

    # 7. Last non-empty line (skip LaTeX structural lines).
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    latex_struct = {r"\begin{cases}", r"\end{cases}", r"\begin{array}", r"\end{array}", r"\\"}
    for line in reversed(lines):
        if line not in latex_struct and len(line) < 200:
            return line

    return None


def compute_rewards(
    completion_texts: List[str],
    ground_truths: List[str],
    reward_type: str = "math_verify",
) -> torch.Tensor:
    """Compute scalar rewards for a batch of completions.

    Args:
        completion_texts: List of decoded completion strings.
        ground_truths: List of ground-truth answer strings (same length).
        reward_type: ``"math_verify"``, ``"deepmath"``, or ``"string_match"``.

    Returns:
        rewards: Float tensor [N].
    """
    rewards = []
    for idx, (comp, gt) in enumerate(zip(completion_texts, ground_truths)):
        if reward_type == "math_verify":
            r = math_reward(comp, gt)
        elif reward_type == "deepmath":
            r = deepmath_reward(comp, gt)
        elif reward_type == "string_match":
            r = 1.0 if comp.strip() == gt.strip() else 0.0
        else:
            raise ValueError(f"Unknown reward_type: {reward_type}")
        rewards.append(r)

        # ── Debug: log extraction details for first 2 completions ──
        if idx < 2:
            gt_stripped = _strip_latex_delimiters(gt)
            pred = _extract_deepmath_answer(comp) if reward_type == "deepmath" else None
            if pred is None and reward_type == "math_verify":
                pred = extract_gsm8k_answer(comp) or extract_boxed_answer(comp)
            has_answer_marker = bool(re.findall(r"(?i)answer\s*:", comp))
            has_boxed = bool(re.findall(r"\\boxed\{", comp))
            comp_tail = comp[-120:] if len(comp) > 120 else comp
            logger.debug(
                "Reward[%d]: r=%.1f | pred=%r | gt_raw=%r | gt_stripped=%r | "
                "has_answer_marker=%s has_boxed=%s | comp_tail=%r",
                idx, r, pred, gt, gt_stripped, has_answer_marker, has_boxed, comp_tail,
            )
    return torch.tensor(rewards, dtype=torch.float32)
