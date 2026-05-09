# coding=utf-8
"""Batch block-score computation for GSPO.

This module computes sequence-level block scores for policy, old-policy,
and reference models across B×G prompt+completion pairs.  It reuses
``build_wedlm_batch`` (random-masking dual stream) and
``compute_block_scores`` from dpo/src, and supports multi-mask-sample
averaging for variance reduction.

Typical usage::

    s_policy, s_old, s_ref = compute_gspo_scores(
        model, old_model, ref_model,
        prompt_ids_list, completion_ids_list,
        tokenizer, scorer_config, attn_wrapper, backend,
    )
    # s_policy: [B*G] with grad
    # s_old:    [B*G] detached
    # s_ref:    [B*G] detached (or None)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from src.batch import WeDLMBatch, build_wedlm_batch
from src.loss import compute_block_scores
from src.model import wedlm_forward
from src.data import get_im_end_token_id

logger = logging.getLogger(__name__)

MASK_TOKEN_ID = 151665  # default fallback; prefer model.config.mask_token_id


def _get_mask_token_id(model) -> int:
    """Read mask token id from a WeDLM model HF config."""
    if hasattr(model, "config") and hasattr(model.config, "mask_token_id"):
        mid = model.config.mask_token_id
        if mid is not None:
            return int(mid)
    return MASK_TOKEN_ID


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ScorerConfig:
    """Parameters for multi-model block-score computation.

    Attributes:
        block_size: WeDLM block size.
        mask_token_id: Token id used for mask placeholder.
        max_seq_length: Maximum token length per sequence.
        num_learnable_im_end: Number of learnable <|im_end|> tokens to append.
        mask_per_block: Whether to sample independent ratios per block.
        loss_weighting_scheme: ``"weighted"`` (1/γ) or ``"uniform"``.
        block_reduce: Reduction inside each block (``"mean"`` or ``"sum"``).
        seq_reduce: Reduction across blocks (``"mean"`` or ``"sum"``).
        mask_eps: Small constant for division stability.
        num_mask_samples: Number of random mask MC samples for score averaging.
    """
    block_size: int = 32
    mask_token_id: int = MASK_TOKEN_ID
    max_seq_length: int = 2048
    num_learnable_im_end: int = 0
    mask_per_block: bool = True
    loss_weighting_scheme: str = "weighted"
    block_reduce: str = "mean"
    seq_reduce: str = "mean"
    mask_eps: float = 1e-8
    num_mask_samples: int = 1


# ──────────────────────────────────────────────────────────────────────────────
# Tokenisation helpers
# ──────────────────────────────────────────────────────────────────────────────

def _tokenize_prompt_completion(
    tokenizer,
    prompt_ids: List[int],
    completion_ids: List[int],
    scorer_config: ScorerConfig,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, int]]:
    """Build (input_ids, labels, original_length) for one pair.

    ``labels`` are -100 for prompt portion and equal to input_ids for
    the completion portion.  This mirrors the training convention so
    that ``build_wedlm_batch`` only masks completion tokens.

    Returns:
        (input_ids, labels, original_length) or None if too short.
    """
    num_reserved = max(0, scorer_config.num_learnable_im_end - 1)
    effective_max = scorer_config.max_seq_length - num_reserved

    full_ids = prompt_ids + completion_ids
    prompt_len = len(prompt_ids)

    if len(full_ids) > effective_max:
        # Truncate completion to fit.
        max_comp_len = effective_max - prompt_len
        if max_comp_len <= 0:
            return None
        full_ids = prompt_ids + completion_ids[:max_comp_len]

    if num_reserved > 0:
        im_end_id = get_im_end_token_id(tokenizer)
        full_ids = full_ids + [im_end_id] * num_reserved

    input_ids = torch.tensor(full_ids, dtype=torch.long)
    labels = input_ids.clone()
    labels[:prompt_len] = -100

    return input_ids, labels, len(full_ids)


def _pack_prompt_completion_pairs(
    tokenizer,
    prompt_ids_list: List[List[int]],
    completion_ids_list: List[List[int]],
    scorer_config: ScorerConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack B×G prompt+completion pairs into flat tensors.

    Args:
        tokenizer: HuggingFace tokenizer.
        prompt_ids_list: B prompts, each a list of token ids.
        completion_ids_list: B×G completions, each a list of token ids.
        scorer_config: Score configuration.

    Returns:
        packed_input_ids: [T] flat token ids.
        packed_labels: [T] labels (-100 for prompt).
        cum_seqlens: [B×G + 1] sequence boundaries.
    """
    all_input_ids: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    seqlens = [0]

    B = len(prompt_ids_list)
    total_pairs = len(completion_ids_list)
    if total_pairs % B != 0:
        raise ValueError(
            f"completion count ({total_pairs}) must be a multiple of prompt count ({B})"
        )

    for i in range(total_pairs):
        prompt_idx = i // (total_pairs // B)
        result = _tokenize_prompt_completion(
            tokenizer,
            prompt_ids_list[prompt_idx],
            completion_ids_list[i],
            scorer_config,
        )
        if result is None:
            raise RuntimeError(
                f"Pair {i}: prompt+completion exceeds max_seq_length "
                f"({scorer_config.max_seq_length})"
            )

        ids, lbls, _ = result
        all_input_ids.append(ids)
        all_labels.append(lbls)
        seqlens.append(seqlens[-1] + ids.size(0))

    packed_input_ids = torch.cat(all_input_ids)
    packed_labels = torch.cat(all_labels)
    cum_seqlens = torch.tensor(seqlens, dtype=torch.long)

    return packed_input_ids, packed_labels, cum_seqlens


# ──────────────────────────────────────────────────────────────────────────────
# Model forward helpers
# ──────────────────────────────────────────────────────────────────────────────

def _try_unwrap(model: nn.Module) -> nn.Module:
    """Attempt to unwrap an Accelerator-wrapped model.

    If the model has an ``unwrap_model`` method (HuggingFace Accelerate
    wraps models this way), use it.  Otherwise return the model as-is.
    """
    # Accelerate wraps models and attaches an _orig_mod or similar.
    # The cleanest check: try to access accelerator.unwrap_model
    # but that requires the accelerator instance.  Instead we look for
    # the module attribute that accelerate sets.
    if hasattr(model, "module"):
        # DataParallel / DistributedDataParallel style.
        return model.module
    # Accelerate's internal wrapper stores original model in _orig_mod.
    if hasattr(model, "_orig_mod"):
        return model._orig_mod
    return model


def _single_model_forward(
    model: nn.Module,
    batch: WeDLMBatch,
    attn_wrapper: nn.Module,
    backend: str,
) -> torch.Tensor:
    """Run wedlm_forward, automatically unwrapping if needed."""
    m = _try_unwrap(model)
    return wedlm_forward(m, batch, attn_wrapper, backend)


# ──────────────────────────────────────────────────────────────────────────────
# Main API
# ──────────────────────────────────────────────────────────────────────────────

def compute_gspo_scores(
    policy_model: nn.Module,
    old_model: nn.Module,
    ref_model: Optional[nn.Module],
    prompt_ids_list: List[List[int]],
    completion_ids_list: List[List[int]],
    tokenizer,
    scorer_config: ScorerConfig,
    attn_wrapper: nn.Module,
    backend: str,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Compute block scores for policy, old-policy, and reference models.

    This is the **core scoring primitive** for GSPO.  It packs all B×G
    prompt+completion pairs into a single packed batch, runs K random-mask
    Monte Carlo forward passes on each model, and returns per-completion
    averaged block scores.

    **Memory strategy**: the policy forward is run with gradients enabled
    **only on the policy model**.  Old and reference forwards run under
    ``torch.no_grad()``.  References can be offloaded to CPU before this
    call to save VRAM; this function does not move models between devices.

    Args:
        policy_model: Current policy (grad-enabled forward).
        old_model: Old-policy snapshot (no-grad forward).
        ref_model: Reference model for KL (no-grad forward; may be None).
        prompt_ids_list: B prompts as token-id lists.
        completion_ids_list: B×G completions as token-id lists.
        tokenizer: HuggingFace tokenizer.
        scorer_config: Scoring hyper-parameters.
        attn_wrapper: Attention wrapper from ``dpo.src.attention``.
        backend: ``"dense"`` or ``"magi"``.

    Returns:
        s_policy: [B×G] policy block scores (with grad).
        s_old:    [B×G] old-policy block scores (detached).
        s_ref:    [B×G] reference block scores (detached), or None.
    """
    K = max(int(scorer_config.num_mask_samples), 1)
    B = len(prompt_ids_list)
    total = len(completion_ids_list)
    if total % B != 0:
        raise ValueError(f"completion count {total} not a multiple of prompt count {B}")

    device = next(policy_model.parameters()).device
    dtype = next(policy_model.parameters()).dtype

    # ── 1. Pack all B×G sequences ─────────────────────────────────────
    packed_ids, packed_labels, cum_seqlens = _pack_prompt_completion_pairs(
        tokenizer, prompt_ids_list, completion_ids_list, scorer_config,
    )
    packed_ids = packed_ids.to(device)
    packed_labels = packed_labels.to(device)
    cum_seqlens = cum_seqlens.to(device)

    bs = cum_seqlens.numel() - 1
    if bs != total:
        raise RuntimeError(f"Internal error: packed {bs} sequences, expected {total}")

    logger.debug("Scorer: %d sequences, %d tokens packed", bs, packed_ids.size(0))

    # ── Helper ────────────────────────────────────────────────────────
    def _score_one(logits_tensor: torch.Tensor, b: WeDLMBatch) -> torch.Tensor:
        scores, _logs = compute_block_scores(
            logits=logits_tensor,
            targets=b.original_ids,
            masked_indices=b.masked_indices,
            p_mask=b.p_mask,
            logical_positions=b.logical_positions,
            cum_seqlens=b.cum_seqlens,
            block_size=scorer_config.block_size,
            weighting_scheme=scorer_config.loss_weighting_scheme,
            block_reduce=scorer_config.block_reduce,
            seq_reduce=scorer_config.seq_reduce,
            eps=scorer_config.mask_eps,
        )
        return scores  # [bs]

    def _score_model(model: nn.Module, grad_enabled: bool) -> torch.Tensor:
        """Run K mask-sample forward passes on one model, return [bs] avg scores."""
        acc: List[torch.Tensor] = []
        ctx = torch.enable_grad() if grad_enabled else torch.no_grad()
        with ctx:
            for _k in range(K):
                batch = build_wedlm_batch(
                    packed_input_ids=packed_ids,
                    packed_labels=packed_labels,
                    cum_seqlens=cum_seqlens,
                    block_size=scorer_config.block_size,
                    mask_token_id=scorer_config.mask_token_id,
                    mask_per_block=scorer_config.mask_per_block,
                    backend=backend,
                    eps=scorer_config.mask_eps,
                )
                logits = _single_model_forward(model, batch, attn_wrapper, backend)
                s_k = _score_one(logits, batch)
                acc.append(s_k if grad_enabled else s_k.detach())
                del logits, batch
        return torch.stack(acc, dim=0).mean(dim=0)  # [bs]

    # ── 2. Score models SEQUENTIALLY ──────────────────────────────────
    #    Process ref first (no-grad), then old (no-grad), then policy (with-grad).
    #    After each frozen model is done, free its cached memory.
    #    With DeepSpeed ZeRO-3 the models are shard-managed; we rely on
    #    torch.cuda.empty_cache() + del to release intermediate tensors.
    s_ref: Optional[torch.Tensor] = None

    if ref_model is not None:
        s_ref = _score_model(ref_model, grad_enabled=False)
        del ref_model  # allow GC (caller still holds reference)
        torch.cuda.empty_cache()

    s_old = _score_model(old_model, grad_enabled=False)
    del old_model
    torch.cuda.empty_cache()

    s_policy = _score_model(policy_model, grad_enabled=True)

    return s_policy, s_old, s_ref
