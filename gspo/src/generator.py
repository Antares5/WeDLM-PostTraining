# coding=utf-8
"""Training-side WeDLM generator for online GSPO.

This module implements a lightweight WeDLM sliding-window generator
that works directly with the training model (AutoModelForCausalLM +
wedlm_forward), avoiding the need for the wedlm/engine inference stack.

The generator reuses dpo/src modules (masking, model, batch, attention)
and builds generation-style WeDLMBatch instances where:
  - committed_ids form the x0 stream (fully observable)
  - window_tokens form the xt stream (masked positions are at end of each block)

Key differences from training forward:
  - Mask positions are deterministic (from window_mask_flags), not randomly sampled
  - Only mask-position logits are extracted for sampling
  - Window state is updated incrementally across generation steps
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

# Reuse dpo/src modules (assume dpo/src is on PYTHONPATH)
from src.batch import WeDLMBatch
from src.masking import reorder_block, build_2d_attention_mask, build_magi_plan
from src.model import wedlm_forward

logger = logging.getLogger(__name__)

# Default fallback; prefer reading from model config.
_DEFAULT_MASK_TOKEN_ID = 151665


def get_mask_token_id(model) -> int:
    """Read the mask token id from a WeDLM model config.

    Tries ``model.config.mask_token_id`` first (HF config),
    then falls back to the hard-coded default.
    """
    if hasattr(model, "config") and hasattr(model.config, "mask_token_id"):
        mid = model.config.mask_token_id
        if mid is not None:
            return int(mid)
    return _DEFAULT_MASK_TOKEN_ID


# ──────────────────────────────────────────────────────────────────────────────
# Data Structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class WeDLMGenerationState:
    """Persistent state for one generation trajectory.

    Attributes:
        committed_ids: Confirmed token ids (prompt + already-generated completion).
            These can attend to each other causally and are never masked.
        window_tokens: Current sliding window token ids.  Some positions are
            still [MASK], waiting to be decoded.
        window_mask_flags: Parallel list; True = masked (needs prediction),
            False = already filled (non-mask).
        generated_ids: Completion tokens produced so far (subset of committed_ids
            beyond the prompt).  Used for EOS / max-length checks.
        is_finished: Set to True when EOS is generated or max_tokens reached.
    """
    committed_ids: List[int] = field(default_factory=list)
    window_tokens: List[int] = field(default_factory=list)
    window_mask_flags: List[bool] = field(default_factory=list)
    generated_ids: List[int] = field(default_factory=list)
    is_finished: bool = False


@dataclass
class GenerationParams:
    """Parameters controlling generation behaviour.

    Attributes:
        max_tokens: Maximum number of completion tokens to generate.
        temperature: Softmax temperature (0.0 = greedy argmax).
        block_size: WeDLM block size (must match training config).
        window_size: Initial sliding window size.
        mask_token_id: Token id used for mask placeholder.
        entropy_threshold: If set, fill ALL mask positions whose adjusted
            entropy falls below this threshold (parallel decode).  If None,
            fill only the single position with minimum adjusted entropy.
        pos_penalty_factor: Coefficient for position-based entropy penalty.
            Higher values favour filling earlier positions in the window.
        repetition_window: Number of recent tokens to check for repetition.
            Set to 0 to disable.
        max_repeated_ratio: If the fraction of the most common token in the
            repetition window exceeds this value, stop generation early.
    """
    max_tokens: int = 256
    temperature: float = 1.0
    block_size: int = 32
    window_size: int = 16
    mask_token_id: int = _DEFAULT_MASK_TOKEN_ID
    entropy_threshold: Optional[float] = None
    pos_penalty_factor: float = 0.02
    repetition_window: int = 64
    max_repeated_ratio: float = 0.75


# ──────────────────────────────────────────────────────────────────────────────
# Batch Construction (Generation Mode)
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def build_wedlm_generation_batch(
    committed_ids: torch.Tensor,         # [Lc]
    window_tokens: torch.Tensor,         # [W]
    window_mask_flags: torch.Tensor,     # [W] bool
    block_size: int,
    mask_token_id: int,
    backend: str = "dense",
    eps: float = 1e-8,
) -> WeDLMBatch:
    """Build a WeDLMBatch for a single generation forward step.

    Unlike ``build_wedlm_batch`` (which randomly samples mask ratios), this
    function uses *deterministic* mask positions derived from
    ``window_mask_flags``.

    Structure overview (single sequence):
        full_seq  = [committed_ids | window_tokens]            length L
        packed    = [full_seq (x0) | reordered xt stream]      length 2L
        cum_seqlens = [0, 2L]
        base_cum_seqlens = [0, L]

    Within each block of the xt stream, unmasked positions come first
    followed by masked positions (which hold ``mask_token_id``).
    """
    device = committed_ids.device
    B = block_size

    Lc = int(committed_ids.size(0))
    W = int(window_tokens.size(0))
    L = Lc + W
    if L <= 0:
        raise ValueError("committed_ids + window_tokens must have at least one token.")

    # Full "original" sequence and positions.
    full_seq = torch.cat([committed_ids, window_tokens])        # [L]
    positions = torch.arange(L, device=device, dtype=torch.long)

    # Protected mask: True = do NOT mask.
    # committed_ids are always protected; window positions are protected
    # when the corresponding mask flag is False.
    protected = torch.cat([
        torch.ones(Lc, dtype=torch.bool, device=device),
        ~window_mask_flags,
    ])                                                          # [L]

    nblk = (L + B - 1) // B

    xt_tokens_list: List[torch.Tensor] = []
    xt_orig_list: List[torch.Tensor] = []
    xt_pos_list: List[torch.Tensor] = []
    xt_mask_list: List[torch.Tensor] = []
    xt_p_list: List[torch.Tensor] = []

    for b in range(nblk):
        blk_st = b * B
        blk_ed = min(L, (b + 1) * B)

        tokens_blk = full_seq[blk_st:blk_ed]
        pos_blk = positions[blk_st:blk_ed]
        maskable = ~protected[blk_st:blk_ed]                    # True = mask this token

        # Deterministic: mask all maskable positions.
        mask_indices = maskable

        num_masked = int(mask_indices.sum().item())
        block_len = int(tokens_blk.size(0))
        p_val = max(float(num_masked) / max(block_len, 1), eps)

        xt_tok, orig_tok, pos_r, p_line = reorder_block(
            tokens_blk, pos_blk, mask_indices, p_val, mask_token_id,
        )

        xt_tokens_list.append(xt_tok)
        xt_orig_list.append(orig_tok)
        xt_pos_list.append(pos_r)
        xt_mask_list.append(torch.cat([
            torch.zeros((int((~mask_indices).sum().item()),), dtype=torch.bool, device=device),
            torch.ones((num_masked,), dtype=torch.bool, device=device),
        ]))
        xt_p_list.append(p_line)

    xt_seq = torch.cat(xt_tokens_list)                          # [L] – xt stream
    xt_orig_seq = torch.cat(xt_orig_list)
    xt_pos_seq = torch.cat(xt_pos_list)
    xt_mask_seq = torch.cat(xt_mask_list)
    xt_p_seq = torch.cat(xt_p_list)

    # Build full packed sequence (x0 | xt) = 2L tokens.
    packed = torch.cat([full_seq, xt_seq])                      # [2L]
    orig = torch.cat([full_seq, xt_orig_seq])
    pos = torch.cat([positions, xt_pos_seq])
    mask = torch.cat([
        torch.zeros(L, dtype=torch.bool, device=device),
        xt_mask_seq,
    ])
    p = torch.cat([
        torch.zeros(L, device=device, dtype=torch.float32),
        xt_p_seq,
    ])

    # Attention mask (per-sequence).
    attn_mask_2d: Optional[torch.Tensor] = None
    if backend == "dense":
        attn_mask_2d = build_2d_attention_mask(L, B, device)

    cum_seqlens = torch.tensor([0, 2 * L], device=device, dtype=torch.long)
    base_cum_seqlens = torch.tensor([0, L], device=device, dtype=torch.long)

    magi_plan: Optional[dict] = None
    if backend == "magi":
        magi_plan = build_magi_plan(base_cum_seqlens, cum_seqlens, B, device)

    return WeDLMBatch(
        packed_input_ids=packed,
        original_ids=orig,
        logical_positions=pos,
        masked_indices=mask,
        p_mask=p,
        cum_seqlens=cum_seqlens,
        base_cum_seqlens=base_cum_seqlens,
        max_seqlen=2 * L,
        attn_mask_2d=attn_mask_2d,
        magi_plan=magi_plan,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Token Sampling & Position Selection
# ──────────────────────────────────────────────────────────────────────────────

def _compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Compute per-token categorical entropy from logits.

    Args:
        logits: [N, V] raw logits.

    Returns:
        entropy: [N] entropy per position.
    """
    return torch.distributions.Categorical(logits=logits).entropy()


def sample_tokens(
    logits: torch.Tensor,              # [N, V]
    temperature: float,
    bad_token_ids: Optional[List[int]] = None,
) -> torch.Tensor:
    """Sample one token per position from logits.

    Args:
        logits: [N, V].
        temperature: 0.0 → greedy argmax.
        bad_token_ids: Token ids to suppress (e.g. mask_token_id, pad_token_id).
            Their logits are set to -inf before sampling.

    Returns:
        sampled_ids: [N] integer token ids.
    """
    # Suppress bad tokens (mask, pad, etc.) so they can never be sampled.
    if bad_token_ids:
        for bid in bad_token_ids:
            if 0 <= bid < logits.size(-1):
                logits[:, bid] = float('-inf')

    if temperature <= 0.0:
        return logits.argmax(dim=-1)

    probs = F.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def select_positions_to_fill(
    entropy: torch.Tensor,              # [N_masks]
    mask_window_indices: torch.Tensor,   # [N_masks] int – window-local indices
    entropy_threshold: Optional[float],
    pos_penalty_factor: float,
) -> List[int]:
    """Choose which mask positions in the window to fill this step.

    Args:
        entropy: Entropy of each mask position.
        mask_window_indices: Window-local index of each mask position.
        entropy_threshold: If set, fill all positions whose *adjusted* entropy
            is below the threshold.  If None, fill only the single position
            with minimum adjusted entropy.
        pos_penalty_factor: Added to entropy proportional to distance from
            the first mask position; encourages left-to-right filling.

    Returns:
        List of indices into ``mask_window_indices`` (the selections).
    """
    if mask_window_indices.numel() == 0:
        return []

    device = entropy.device

    # Position penalty: distance from the first mask position.
    base_pos = float(mask_window_indices[0].item())
    distances = mask_window_indices.float() - base_pos
    adjusted = entropy + distances * pos_penalty_factor

    if entropy_threshold is not None:
        candidates = (adjusted < entropy_threshold).nonzero(as_tuple=True)[0]
        if candidates.numel() > 0:
            return candidates.tolist()

    # Default: single position with minimum adjusted entropy.
    return [int(adjusted.argmin().item())]


# ──────────────────────────────────────────────────────────────────────────────
# Repetition Detection (Early Stopping)
# ──────────────────────────────────────────────────────────────────────────────

def _check_generation_repetition(
    generated_ids: List[int],
    params: GenerationParams,
) -> bool:
    """Check if recent generated tokens are excessively repetitive.

    When the model can't terminate via EOS (e.g. it's stuck in a degenerate
    state), it will often repeat the same token endlessly.  This function
    detects that pattern and returns True to signal early stopping.

    Args:
        generated_ids: All completion token ids generated so far.
        params: Generation parameters.

    Returns:
        True if generation should be stopped due to excessive repetition.
    """
    if params.repetition_window <= 0 or len(generated_ids) < params.repetition_window:
        return False

    recent = generated_ids[-params.repetition_window:]
    if not recent:
        return False

    # Count occurrences of each token id.
    from collections import Counter
    counts = Counter(recent)
    most_common_count = counts.most_common(1)[0][1]
    ratio = most_common_count / len(recent)

    if ratio >= params.max_repeated_ratio:
        logger.debug(
            "Repetition detected: most common token %d appears %d/%d (%.1f%%) in "
            "last %d tokens — stopping early",
            counts.most_common(1)[0][0], most_common_count, len(recent),
            ratio * 100, params.repetition_window,
        )
        return True
    return False


# ──────────────────────────────────────────────────────────────────────────────
# Window Management
# ──────────────────────────────────────────────────────────────────────────────

def _prune_window_prefix(
    state: WeDLMGenerationState,
    params: GenerationParams,
    eos_token_id: int,
) -> List[int]:
    """Prune fully-confirmed prefix tokens from the window.

    Scans window_mask_flags from the start.  Every token before the first
    remaining mask is considered "confirmed" and is moved to committed_ids.

    Respects ``params.max_tokens``: if committing the full prefix would
    exceed the budget, only commits up to the budget and marks the sequence
    as finished.

    Returns the list of newly committed token ids (from the window prefix).
    """
    window = state.window_tokens
    flags = state.window_mask_flags

    # Find first index that is still masked.
    mask_positions = [i for i, f in enumerate(flags) if f]
    full_prune_count = mask_positions[0] if mask_positions else len(window)

    if full_prune_count == 0:
        return []

    # Clamp to max_tokens budget.
    remaining_budget = params.max_tokens - len(state.generated_ids)
    if remaining_budget <= 0:
        state.is_finished = True
        state.window_tokens = []
        state.window_mask_flags = []
        return []

    prune_count = min(full_prune_count, remaining_budget)

    pruned = window[:prune_count]
    newly_generated: List[int] = []

    for tok in pruned:
        state.committed_ids.append(tok)
        state.generated_ids.append(tok)
        newly_generated.append(tok)

        if tok == eos_token_id:
            state.is_finished = True
            break

    # If budget was exhausted (truncation), mark finished.
    if len(state.generated_ids) >= params.max_tokens:
        state.is_finished = True

    if state.is_finished:
        state.window_tokens = []
        state.window_mask_flags = []
    else:
        # Shift window: remove pruned tokens, refill with new masks.
        # Use actual committed count (may be < prune_count if EOS interrupted).
        actual_shift = len(newly_generated)
        if actual_shift == 0:
            # Nothing was committed (shouldn't happen if full_prune_count > 0).
            return newly_generated
        refill_count = actual_shift
        state.window_tokens = (
            window[actual_shift:] + [params.mask_token_id] * refill_count
        )
        state.window_mask_flags = (
            flags[actual_shift:] + [True] * refill_count
        )

    return newly_generated


# ──────────────────────────────────────────────────────────────────────────────
# Main Generation Loop
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def wedlm_generate(
    model,
    tokenizer,
    prompt: str,
    params: GenerationParams,
    attn_wrapper,
    backend: str,
    eos_token_id: Optional[int] = None,
    seed: Optional[int] = None,
    system_prompt: Optional[str] = None,
) -> Tuple[List[int], WeDLMGenerationState]:
    """Generate completion tokens using WeDLM sliding-window decoding.

    Args:
        ...
        system_prompt: Optional system instruction prepended to the prompt
            (e.g. to guide answer format).  If given, the tokenized prompt
            becomes ``system_prompt + "\n\n" + prompt`` as a chat template.
    """
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id
        if eos_token_id is None:
            raise ValueError("eos_token_id must be provided or set in tokenizer.")

    # Collect token ids that should NEVER be generated.
    # CRITICAL: Do NOT suppress eos_token_id — the model must be able to
    # generate <|im_end|> to terminate naturally.  pad_token_id is often
    # set to the same token, so always exclude eos_token_id.
    bad_token_ids: List[int] = [params.mask_token_id]
    _pad = tokenizer.pad_token_id
    if _pad is not None and _pad != params.mask_token_id and _pad != eos_token_id:
        bad_token_ids.append(_pad)

    if seed is not None:
        torch.manual_seed(seed)

    device = next(model.parameters()).device

    # ── 1. Tokenize prompt (with optional system instruction) ────────
    if system_prompt:
        # Build chat-style messages: system + user.
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        full_prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    else:
        full_prompt_text = prompt

    prompt_ids = tokenizer.encode(full_prompt_text, add_special_tokens=False)
    if not prompt_ids:
        raise ValueError("Prompt tokenized to empty list.")

    # ── 2. Init state ───────────────────────────────────────────────────
    state = WeDLMGenerationState(
        committed_ids=list(prompt_ids),
        window_tokens=[params.mask_token_id] * params.window_size,
        window_mask_flags=[True] * params.window_size,
        generated_ids=[],
        is_finished=False,
    )

    # ── 3. Decode loop ──────────────────────────────────────────────────
    while len(state.generated_ids) < params.max_tokens and not state.is_finished:
        # 3a. Build batch.
        batch = build_wedlm_generation_batch(
            committed_ids=torch.tensor(state.committed_ids, device=device, dtype=torch.long),
            window_tokens=torch.tensor(state.window_tokens, device=device, dtype=torch.long),
            window_mask_flags=torch.tensor(state.window_mask_flags, device=device, dtype=torch.bool),
            block_size=params.block_size,
            mask_token_id=params.mask_token_id,
            backend=backend,
        )

        # 3b. Forward.
        logits = wedlm_forward(model, batch, attn_wrapper, backend)   # [2L, V]

        # 3c. Extract mask-position logits.
        #     batch.masked_indices is [2L] bool; True at xt-stream mask positions.
        if not batch.masked_indices.any():
            # No masks left – this shouldn't happen normally, but be safe.
            state.is_finished = True
            break

        mask_logits = logits[batch.masked_indices]                     # [N_masks, V]
        mask_positions = batch.logical_positions[batch.masked_indices] # [N_masks]

        # Map logical positions to window-local indices.
        Lc = len(state.committed_ids)
        window_local_idx = mask_positions - Lc                        # [N_masks]

        # 3d. Sample tokens, suppressing bad token ids.
        sampled = sample_tokens(mask_logits, params.temperature, bad_token_ids)      # [N_masks]

        # 3e. Select which positions to fill.
        entropy = _compute_entropy(mask_logits)                       # [N_masks]
        fill_sel = select_positions_to_fill(
            entropy,
            window_local_idx,
            params.entropy_threshold,
            params.pos_penalty_factor,
        )

        # 3f. Fill selected positions in the window, check EOS instantly.
        hit_eos = False
        eos_window_pos = -1
        for sel in fill_sel:
            win_pos = int(window_local_idx[sel].item())
            tok = int(sampled[sel].item())
            # Safety: skip if model still produced a bad token (should not
            # happen after sample_tokens suppression, but guard defensively).
            if tok in bad_token_ids:
                logger.warning(
                    "Step %d: sampled bad token %d at win_pos %d — skipping fill",
                    len(state.generated_ids), tok, win_pos,
                )
                continue
            if 0 <= win_pos < len(state.window_tokens):
                state.window_tokens[win_pos] = tok
                state.window_mask_flags[win_pos] = False
                if tok == eos_token_id:
                    hit_eos = True
                    eos_window_pos = win_pos

        # 3g. If EOS was filled: commit all completion tokens before EOS,
        #     plus EOS itself, then stop.
        if hit_eos:
            for j in range(eos_window_pos + 1):
                tok = state.window_tokens[j]
                if not state.window_mask_flags[j]:
                    # This is a filled (non-mask) token; commit it.
                    state.committed_ids.append(tok)
                    state.generated_ids.append(tok)
            state.window_tokens = []
            state.window_mask_flags = []
            state.is_finished = True
            break

        # 3h. Prune confirmed prefix & check stop conditions.
        newly_pruned = _prune_window_prefix(state, params, eos_token_id)

        # 3i. Early stop: detect repetitive generation (model stuck in loop).
        # Only check when new tokens were actually committed this step.
        if newly_pruned and _check_generation_repetition(state.generated_ids, params):
            logger.info(
                "Early stop at %d tokens due to repetition",
                len(state.generated_ids),
            )
            state.is_finished = True

        # Guard: if window became empty, stop.
        if len(state.window_tokens) == 0:
            state.is_finished = True

    return state.generated_ids, state
