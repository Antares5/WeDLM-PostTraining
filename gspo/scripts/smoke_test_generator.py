# coding=utf-8
"""Smoke test for gspo/src/generator.py — Step 1 validation.

This script validates the training-side WeDLM generator WITHOUT needing
to load a full model.  It runs in two phases:

  Phase A (no model): Tests batch construction, token sampling, position
      selection, and window pruning with small synthetic tensors.

  Phase B (with model): If ``--model-path`` is provided, loads a real
      WeDLM model and runs one full generation to verify end-to-end
      correctness.

Usage:
    # Phase A only (fast, no GPU model needed):
    python scripts/smoke_test_generator.py

    # Phase A + B (needs GPU + HuggingFace model):
    python scripts/smoke_test_generator.py --model-path tencent/WeDLM-8B-Base
"""

from __future__ import annotations

import os
import sys
import argparse
import logging
import warnings

# Ensure dpo/src and gspo/src are importable.
# IMPORTANT: order matters.  dpo/ must come before gspo/src so that
# ``import src`` resolves to dpo/src (which has batch, masking, model, etc.).
# gspo/src is added separately so that ``import generator`` works.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)          # gspo/
_PARENT_DIR = os.path.dirname(_PROJECT_DIR)           # repo root
sys.path.insert(0, os.path.join(_PARENT_DIR, "dpo"))  # for dpo/src/ (src.batch, src.masking, etc.)
sys.path.insert(0, os.path.join(_PROJECT_DIR, "src"))  # for gspo/src/ (generator)

import torch

from generator import (
    MASK_TOKEN_ID,
    WeDLMGenerationState,
    GenerationParams,
    build_wedlm_generation_batch,
    sample_tokens,
    select_positions_to_fill,
    wedlm_generate,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Phase A: Unit Tests (no model required)
# ═══════════════════════════════════════════════════════════════════════════════

def _allclose(a, b, **kw):
    return bool(torch.allclose(torch.tensor(a, dtype=torch.float32),
                                torch.tensor(b, dtype=torch.float32), **kw))


def test_build_generation_batch_structure():
    """Verify build_wedlm_generation_batch produces a valid WeDLMBatch.

    Checks:
      - Correct packed length (2L).
      - x0 stream equals full_seq.
      - masked_indices only appear in xt portion.
      - Attention mask shape is [2L, 2L].
    """
    logger.info("=== test_build_generation_batch_structure ===")

    committed = torch.tensor([1, 2, 3, 4], dtype=torch.long, device="cpu")
    window = torch.tensor([MASK_TOKEN_ID, MASK_TOKEN_ID, 10, MASK_TOKEN_ID],
                          dtype=torch.long, device="cpu")
    flags = torch.tensor([True, True, False, True], dtype=torch.bool, device="cpu")

    batch = build_wedlm_generation_batch(
        committed_ids=committed,
        window_tokens=window,
        window_mask_flags=flags,
        block_size=4,
        mask_token_id=MASK_TOKEN_ID,
        backend="dense",
    )

    Lc = 4
    W = 4
    L = Lc + W

    # 1) Packed length = 2L.
    assert batch.packed_input_ids.size(0) == 2 * L, \
        f"Expected packed length {2*L}, got {batch.packed_input_ids.size(0)}"
    logger.info("  ✓ packed length = %d", batch.packed_input_ids.size(0))

    # 2) x0 portion (first L) == committed + window.
    x0 = batch.packed_input_ids[:L]
    expected_x0 = torch.cat([committed, window])
    assert torch.equal(x0, expected_x0), "x0 stream mismatch"
    logger.info("  ✓ x0 stream matches committed + window")

    # 3) masked_indices are all in xt portion (positions >= L).
    mask_positions = batch.masked_indices.nonzero(as_tuple=True)[0]
    assert (mask_positions >= L).all(), \
        f"Mask found in x0 portion at {mask_positions[mask_positions < L]}"
    logger.info("  ✓ all masks in xt portion (%d mask positions)", mask_positions.numel())

    # 4) Attention mask.
    if batch.attn_mask_2d is not None:
        assert batch.attn_mask_2d.shape == (2 * L, 2 * L), \
            f"attn_mask shape {batch.attn_mask_2d.shape} != {(2*L, 2*L)}"
        logger.info("  ✓ attention mask shape [%d, %d]", 2 * L, 2 * L)

    # 5) cum_seqlens / base_cum_seqlens.
    assert torch.equal(batch.cum_seqlens, torch.tensor([0, 2 * L])), "cum_seqlens wrong"
    assert torch.equal(batch.base_cum_seqlens, torch.tensor([0, L])), "base_cum_seqlens wrong"
    logger.info("  ✓ cum_seqlens / base_cum_seqlens correct")

    logger.info("  PASSED\n")


def test_build_generation_batch_mask_mapping():
    """Verify that mask positions in the batch correctly map back to window.

    Checks:
      - logical_positions at masked indices point to original positions.
      - window-local indices (logical_pos - Lc) match window_mask_flags.
    """
    logger.info("=== test_build_generation_batch_mask_mapping ===")

    committed = torch.tensor([1, 2, 3], dtype=torch.long)
    # window: 4 tokens, positions 3,4 are masked; position 5 is filled; position 6 is masked.
    window = torch.tensor([MASK_TOKEN_ID, MASK_TOKEN_ID, 42, MASK_TOKEN_ID],
                          dtype=torch.long)
    flags = torch.tensor([True, True, False, True], dtype=torch.bool)

    batch = build_wedlm_generation_batch(
        committed_ids=committed,
        window_tokens=window,
        window_mask_flags=flags,
        block_size=4,
        mask_token_id=MASK_TOKEN_ID,
        backend="dense",
    )

    Lc = 3

    # Mask positions in the full 2L sequence.
    mask_pos = batch.masked_indices.nonzero(as_tuple=True)[0]       # positions in [L, 2L-1]
    mask_lpos = batch.logical_positions[mask_pos]                   # original L-positions

    # Map to window-local index.
    window_local = mask_lpos - Lc
    logger.info("  mask global pos: %s", mask_pos.tolist())
    logger.info("  logical (orig) pos: %s", mask_lpos.tolist())
    logger.info("  window-local idx: %s", window_local.tolist())

    # Verify: window_local should be [0, 1, 3] (positions 0,1,3 in window are masked).
    expected_local = torch.tensor([0, 1, 3], dtype=torch.long)
    assert torch.equal(window_local, expected_local), \
        f"window-local mismatch: {window_local.tolist()} != {expected_local.tolist()}"
    logger.info("  ✓ window-local mapping correct")

    # Verify that these window positions indeed have mask flag True.
    for idx in window_local.tolist():
        assert flags[idx], f"Position {idx} should be masked but is not"
    logger.info("  ✓ all mapped positions are masked in window_mask_flags")

    logger.info("  PASSED\n")


def test_sample_tokens():
    """Test token sampling correctness."""
    logger.info("=== test_sample_tokens ===")

    torch.manual_seed(42)
    logits = torch.randn(10, 100)

    # Greedy (temp=0).
    greedy = sample_tokens(logits, 0.0)
    assert torch.equal(greedy, logits.argmax(dim=-1)), "greedy != argmax"
    logger.info("  ✓ greedy sampling matches argmax")

    # Random (temp=1.0).
    sampled = sample_tokens(logits, 1.0)
    assert sampled.shape == (10,), f"shape {sampled.shape} != (10,)"
    assert (sampled >= 0).all() and (sampled < 100).all(), "token ids out of range"
    logger.info("  ✓ random sampling produces valid token ids")

    # Reproducibility.
    torch.manual_seed(42)
    s1 = sample_tokens(logits, 1.0)
    torch.manual_seed(42)
    s2 = sample_tokens(logits, 1.0)
    assert torch.equal(s1, s2), "same seed gave different results"
    logger.info("  ✓ reproducibility with fixed seed")

    logger.info("  PASSED\n")


def test_select_positions():
    """Test position selection logic."""
    logger.info("=== test_select_positions ===")

    # 5 mask positions at window indices [0, 3, 5, 8, 10].
    entropy = torch.tensor([0.5, 0.2, 0.8, 0.1, 0.9])
    indices = torch.tensor([0, 3, 5, 8, 10], dtype=torch.float32)

    # No threshold → pick minimum adjusted entropy.
    sel = select_positions_to_fill(entropy, indices, None, 0.0)
    assert sel == [3], f"expected [3] (min entropy), got {sel}"
    logger.info("  ✓ no threshold: picks min-entropy position (idx 3)")

    # With position penalty.
    sel2 = select_positions_to_fill(entropy, indices, None, 0.1)
    logger.info("  ✓ with pos_penalty=0.1: selected indices = %s", sel2)

    # Low threshold → select all.
    sel3 = select_positions_to_fill(entropy, indices, 2.0, 0.0)
    assert len(sel3) == 5, f"expected all 5, got {len(sel3)}"
    logger.info("  ✓ threshold=2.0: selects all positions")

    # Medium threshold.
    sel4 = select_positions_to_fill(entropy, indices, 0.6, 0.0)
    logger.info("  ✓ threshold=0.6: selected indices = %s", sel4)

    logger.info("  PASSED\n")


def test_window_pruning():
    """Test the internal window pruning logic via _prune_window_prefix."""
    logger.info("=== test_window_pruning ===")

    # Import the internal function directly.
    from generator import _prune_window_prefix

    params = GenerationParams(window_size=4, max_tokens=100)

    # State: 2 committed tokens, window [A, B, MASK, MASK].
    state = WeDLMGenerationState(
        committed_ids=[1, 2],
        window_tokens=[10, 20, MASK_TOKEN_ID, MASK_TOKEN_ID],
        window_mask_flags=[False, False, True, True],
        generated_ids=[],
    )

    pruned = _prune_window_prefix(state, params, eos_token_id=99999)
    assert pruned == [10, 20], f"expected [10, 20], got {pruned}"
    assert state.committed_ids == [1, 2, 10, 20], f"committed: {state.committed_ids}"
    assert state.generated_ids == [10, 20]
    assert len(state.window_tokens) == 4, "window should still be size 4"
    assert state.window_mask_flags == [True, True, True, True], \
        f"refilled should be all masked, got {state.window_mask_flags}"
    logger.info("  ✓ prefix prune + refill correct")
    logger.info("    new window: %s", state.window_tokens)
    logger.info("    new flags:  %s", state.window_mask_flags)

    logger.info("  PASSED\n")


def test_eos_detection():
    """Verify that EOS during prefix prune terminates generation."""
    logger.info("=== test_eos_detection ===")
    from generator import _prune_window_prefix

    params = GenerationParams(window_size=4, max_tokens=100)
    state = WeDLMGenerationState(
        committed_ids=[1],
        window_tokens=[2, MASK_TOKEN_ID, MASK_TOKEN_ID, MASK_TOKEN_ID],
        window_mask_flags=[False, True, True, True],
        generated_ids=[],
    )

    # Token 2 is EOS.
    pruned = _prune_window_prefix(state, params, eos_token_id=2)
    assert state.is_finished, "should be finished after EOS"
    assert pruned == [2], f"expected [2], got {pruned}"
    assert state.window_tokens == [], "window should be cleared"
    logger.info("  ✓ EOS termination correct")

    logger.info("  PASSED\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Phase B: Integration Test (requires model)
# ═══════════════════════════════════════════════════════════════════════════════

def test_real_generation(model_path: str):
    """Load a real WeDLM model and run one generation."""
    logger.info("=== test_real_generation (model: %s) ===", model_path)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from src.attention import get_available_backend, get_attention_wrapper
    from src.model import wedlm_forward  # verify import works

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("  device: %s", device)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    logger.info("  tokenizer loaded (vocab=%d)", tokenizer.vocab_size)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation="eager",
    ).to(device)
    model.eval()
    logger.info("  model loaded (%s)", type(model).__name__)

    head_dim = model.config.hidden_size // model.config.num_attention_heads
    backend = get_available_backend()
    attn_wrapper = get_attention_wrapper(backend, head_dim)
    if hasattr(attn_wrapper, "to"):
        attn_wrapper = attn_wrapper.to(device)
    logger.info("  attention backend: %s", backend)

    eos_id = tokenizer.eos_token_id
    logger.info("  eos_token_id: %s", eos_id)

    params = GenerationParams(
        max_tokens=32,
        temperature=0.8,
        block_size=32,
        window_size=16,
        mask_token_id=MASK_TOKEN_ID,
    )

    prompt = "Hello, how are you?"
    logger.info("  prompt: %r", prompt)

    torch.manual_seed(42)
    completion_ids, state = wedlm_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        params=params,
        attn_wrapper=attn_wrapper,
        backend=backend,
        eos_token_id=eos_id,
        seed=42,
    )

    logger.info("  generated %d tokens", len(completion_ids))
    completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
    logger.info("  completion: %r", completion_text)
    logger.info("  is_finished: %s", state.is_finished)

    # Basic sanity checks.
    assert len(completion_ids) > 0, "no tokens generated!"
    logger.info("  ✓ generated tokens > 0")
    assert len(completion_ids) <= params.max_tokens, "exceeded max_tokens"
    logger.info("  ✓ within max_tokens limit")

    logger.info("  PASSED\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Smoke test for WeDLM generator")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to WeDLM model for Phase B integration test")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("GSPO Generator Smoke Test")
    logger.info("=" * 60)
    logger.info("MASK_TOKEN_ID = %d", MASK_TOKEN_ID)
    logger.info("Torch version: %s", torch.__version__)
    logger.info("CUDA available: %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        logger.info("CUDA device: %s", torch.cuda.get_device_name(0))
    logger.info("")

    # ── Phase A ──
    logger.info("─" * 40)
    logger.info("Phase A: Unit tests (no model)")
    logger.info("─" * 40)
    test_build_generation_batch_structure()
    test_build_generation_batch_mask_mapping()
    test_sample_tokens()
    test_select_positions()
    test_window_pruning()
    test_eos_detection()
    logger.info("Phase A: ALL PASSED ✓")

    # ── Phase B (optional) ──
    if args.model_path:
        logger.info("─" * 40)
        logger.info("Phase B: Integration test (with model)")
        logger.info("─" * 40)
        test_real_generation(args.model_path)
        logger.info("Phase B: PASSED ✓")

    logger.info("=" * 60)
    logger.info("All tests passed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
