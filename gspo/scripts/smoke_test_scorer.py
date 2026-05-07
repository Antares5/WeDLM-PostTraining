# coding=utf-8
"""Smoke test for gspo/src/scorer.py — Step 2 validation.

Validates batch block-score computation for GSPO.

Usage:
    # Phase A: unit tests (no model needed)
    python scripts/smoke_test_scorer.py

    # Phase A + B: integration with real model
    python scripts/smoke_test_scorer.py --model-path tencent/WeDLM-8B-Instruct
"""

from __future__ import annotations

import os
import sys
import argparse
import logging
import warnings

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
_PARENT_DIR = os.path.dirname(_PROJECT_DIR)
sys.path.insert(0, os.path.join(_PARENT_DIR, "dpo"))
sys.path.insert(0, os.path.join(_PROJECT_DIR, "src"))

import torch

from scorer import (
    MASK_TOKEN_ID,
    ScorerConfig,
    _tokenize_prompt_completion,
    _pack_prompt_completion_pairs,
    compute_gspo_scores,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Phase A: Unit Tests (no model required)
# ═══════════════════════════════════════════════════════════════════════════════

class _FakeTokenizer:
    """Minimal tokenizer stub for unit tests that don't involve a real model."""
    def __init__(self, vocab_size=1000, eos_token_id=999):
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id
        self.pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        # Deterministic fake encoding: each char → its ord mod 100.
        return [ord(c) % 100 + 1 for c in text]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr((i - 1) % 100) for i in ids if i > 0)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        # Simple concatenation for testing.
        parts = [msg.get("content", "") for msg in messages]
        full = "\n".join(parts)
        if tokenize:
            return self.encode(full)
        return full


class _FakeTokenizerWithIMEnd(_FakeTokenizer):
    """Fake tokenizer that knows about <|im_end|> token."""
    def __init__(self, vocab_size=1000, im_end_id=888):
        super().__init__(vocab_size=vocab_size)
        self._im_end_id = im_end_id

    def convert_tokens_to_ids(self, token):
        if token == "<|im_end|>":
            return self._im_end_id
        return 0


def test_pack_single_pair():
    """Verify packing of a single prompt+completion pair."""
    logger.info("=== test_pack_single_pair ===")

    tokenizer = _FakeTokenizerWithIMEnd(im_end_id=888)
    config = ScorerConfig(max_seq_length=64, num_learnable_im_end=0)

    prompt_ids = [1, 2, 3]
    completion_ids = [4, 5, 6, 7]
    result = _tokenize_prompt_completion(tokenizer, prompt_ids, completion_ids, config)
    assert result is not None
    ids, labels, orig_len = result

    # ids = [1,2,3,4,5,6,7]
    assert ids.tolist() == [1, 2, 3, 4, 5, 6, 7], f"ids: {ids.tolist()}"
    # labels: prompt=-100, completion=ids
    assert labels.tolist()[:3] == [-100, -100, -100]
    assert labels.tolist()[3:] == [4, 5, 6, 7]
    logger.info("  ✓ single pair tokenization correct")

    logger.info("  PASSED\n")


def test_pack_multiple_pairs():
    """Verify packing of B×G pairs with cum_seqlens."""
    logger.info("=== test_pack_multiple_pairs ===")

    tokenizer = _FakeTokenizerWithIMEnd(im_end_id=888)
    config = ScorerConfig(max_seq_length=64, num_learnable_im_end=0)

    # B=2 prompts, G=2 completions each → 4 pairs
    prompt_ids_list = [
        [1, 2],        # prompt 0
        [3, 4, 5],     # prompt 1
    ]
    completion_ids_list = [
        [10, 11],      # p0, c0
        [20, 21, 22],  # p0, c1
        [30, 31],      # p1, c0
        [40, 41, 42],  # p1, c1
    ]

    packed_ids, packed_labels, cum_seqlens = _pack_prompt_completion_pairs(
        tokenizer, prompt_ids_list, completion_ids_list, config,
    )

    assert cum_seqlens.numel() == 5, f"cum_seqlens should have 5 elements, got {cum_seqlens.numel()}"
    # seq 0: [1,2,10,11] len=4
    # seq 1: [1,2,20,21,22] len=5
    # seq 2: [3,4,5,30,31] len=5
    # seq 3: [3,4,5,40,41,42] len=6
    expected_cum = [0, 4, 9, 14, 20]
    assert cum_seqlens.tolist() == expected_cum, f"cum_seqlens: {cum_seqlens.tolist()} != {expected_cum}"
    logger.info("  ✓ cum_seqlens correct: %s", cum_seqlens.tolist())

    # Verify that each prompt's labels have -100 at prompt positions.
    for seq_idx in range(4):
        st = cum_seqlens[seq_idx].item()
        ed = cum_seqlens[seq_idx + 1].item()
        seq_labels = packed_labels[st:ed]
        prompt_idx = seq_idx // 2
        prompt_len = len(prompt_ids_list[prompt_idx])
        assert (seq_labels[:prompt_len] == -100).all(), \
            f"seq {seq_idx}: first {prompt_len} labels should be -100"
        assert (seq_labels[prompt_len:] >= 0).all(), \
            f"seq {seq_idx}: completion labels should be >= 0"
    logger.info("  ✓ prompt masking correct for all sequences")

    logger.info("  PASSED\n")


def test_truncation():
    """Verify that long completions are truncated to max_seq_length."""
    logger.info("=== test_truncation ===")

    tokenizer = _FakeTokenizerWithIMEnd(im_end_id=888)
    config = ScorerConfig(max_seq_length=10, num_learnable_im_end=0)

    prompt_ids = [1, 2, 3]
    long_completion = list(range(4, 50))  # way too long
    result = _tokenize_prompt_completion(tokenizer, prompt_ids, long_completion, config)
    assert result is not None
    ids, labels, orig_len = result

    # prompt=3, completion allowed up to 10-3=7
    assert ids.size(0) <= 10, f"ids length {ids.size(0)} > 10"
    assert ids.size(0) == 10, f"expected max 10, got {ids.size(0)}"
    logger.info("  ✓ truncated to max_seq_length: %d tokens", ids.size(0))

    # Edge case: prompt alone fills max_seq_length.
    long_prompt = list(range(1, 12))
    result2 = _tokenize_prompt_completion(tokenizer, long_prompt, [100], config)
    assert result2 is None, "should return None when no room for completion"
    logger.info("  ✓ returns None when prompt fills budget")

    logger.info("  PASSED\n")


def test_scorer_config_defaults():
    """Verify ScorerConfig default values."""
    logger.info("=== test_scorer_config_defaults ===")
    cfg = ScorerConfig()
    assert cfg.block_size == 32
    assert cfg.mask_token_id == MASK_TOKEN_ID
    assert cfg.loss_weighting_scheme == "weighted"
    assert cfg.block_reduce == "mean"
    assert cfg.seq_reduce == "mean"
    assert cfg.num_mask_samples == 1
    logger.info("  ✓ defaults correct")
    logger.info("  PASSED\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Phase B: Integration Test (requires model + GPU)
# ═══════════════════════════════════════════════════════════════════════════════

def test_real_scoring(model_path: str):
    """Load a real WeDLM model and verify multi-model scoring pipeline.

    This test:
    1. Loads policy model (with grad).
    2. Uses same model as old_model (for test; real training would snapshot).
    3. Computes scores on a few synthetic prompt+completion pairs.
    4. Verifies shapes, gradient flow, and score consistency.
    """
    logger.info("=== test_real_scoring (model: %s) ===", model_path)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from src.attention import get_available_backend, get_attention_wrapper

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("  device: %s", device)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    logger.info("  tokenizer loaded (vocab=%d)", tokenizer.vocab_size)

    model_kwargs = dict(
        trust_remote_code=True,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation="eager",
    )
    policy_model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs).to(device)
    policy_model.train()
    logger.info("  policy model loaded (%s)", type(policy_model).__name__)

    # For test, old_model = same model (real training would snapshot).
    old_model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs).to(device)
    old_model.eval()
    for p in old_model.parameters():
        p.requires_grad = False
    logger.info("  old_model loaded")

    ref_model = None  # Not strictly needed for scoring test.

    head_dim = policy_model.config.hidden_size // policy_model.config.num_attention_heads
    backend = get_available_backend()
    attn_wrapper = get_attention_wrapper(backend, head_dim)
    if hasattr(attn_wrapper, "to"):
        attn_wrapper = attn_wrapper.to(device)
    logger.info("  attention backend: %s", backend)

    scorer_config = ScorerConfig(
        block_size=32,
        mask_token_id=MASK_TOKEN_ID,
        max_seq_length=128,
        num_learnable_im_end=0,
        mask_per_block=True,
        loss_weighting_scheme="weighted",
        block_reduce="mean",
        seq_reduce="mean",
        num_mask_samples=2,  # 2 MC samples for variance reduction test
    )

    # ── Prepare test data ────────────────────────────────────────────
    prompts = ["What is 1+1?", "What is the capital of France?"]
    prompt_ids_list = [tokenizer.encode(p, add_special_tokens=False) for p in prompts]

    completions = [
        "2", "It's 2.",                                    # prompt 0: G=2
        "Paris", "The capital is Paris.",                   # prompt 1: G=2
    ]
    completion_ids_list = [tokenizer.encode(c, add_special_tokens=False) for c in completions]

    logger.info("  B=%d prompts, G=%d completions each", len(prompts), len(completions) // len(prompts))

    # ── Compute scores ───────────────────────────────────────────────
    s_policy, s_old, s_ref = compute_gspo_scores(
        policy_model=policy_model,
        old_model=old_model,
        ref_model=ref_model,
        prompt_ids_list=prompt_ids_list,
        completion_ids_list=completion_ids_list,
        tokenizer=tokenizer,
        scorer_config=scorer_config,
        attn_wrapper=attn_wrapper,
        backend=backend,
    )

    # ── Checks ───────────────────────────────────────────────────────
    expected_len = len(completions)  # 4
    assert s_policy.shape == (expected_len,), \
        f"s_policy shape {s_policy.shape} != ({expected_len},)"
    logger.info("  ✓ s_policy shape: %s", tuple(s_policy.shape))

    assert s_old.shape == (expected_len,), \
        f"s_old shape {s_old.shape} != ({expected_len},)"
    logger.info("  ✓ s_old shape: %s", tuple(s_old.shape))

    assert s_ref is None, "ref_model is None, s_ref should be None"
    logger.info("  ✓ s_ref is None (no ref_model)")

    # Grad check.
    assert s_policy.requires_grad, "s_policy must have requires_grad=True"
    logger.info("  ✓ s_policy.requires_grad = True")

    assert not s_old.requires_grad, "s_old must be detached"
    logger.info("  ✓ s_old.requires_grad = False")

    # Grad flow: backward through s_policy.
    s_policy.mean().backward()
    grad_found = False
    for name, p in policy_model.named_parameters():
        if p.grad is not None:
            grad_found = True
            break
    assert grad_found, "no gradients flowed to policy model!"
    logger.info("  ✓ gradient flows back to policy model")

    # old_model should have NO gradients.
    for name, p in old_model.named_parameters():
        assert p.grad is None, f"old_model param {name} has grad (should be None)"
    logger.info("  ✓ old_model has no gradients")

    # Score values should be finite.
    assert torch.isfinite(s_policy).all(), "s_policy contains NaN/Inf"
    assert torch.isfinite(s_old).all(), "s_old contains NaN/Inf"
    logger.info("  ✓ all scores are finite")
    logger.info("    s_policy: %s", s_policy.detach().tolist())
    logger.info("    s_old:    %s", s_old.tolist())

    # Score consistency: same completions under the same model should
    # get similar scores (policy ≈ old since same weights).
    diff = (s_policy - s_old).abs().max().item()
    logger.info("    max |s_policy - s_old|: %.6f", diff)
    # They should be close since same weights, but not identical due to
    # random masking + policy forward runs with train() mode (dropout etc).
    # We just check they're within a reasonable range.
    assert diff < 20.0, f"Score discrepancy too large: {diff}"

    logger.info("  PASSED\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Smoke test for GSPO scorer")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to WeDLM model for Phase B integration test")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("GSPO Scorer Smoke Test")
    logger.info("=" * 60)
    logger.info("MASK_TOKEN_ID = %d", MASK_TOKEN_ID)
    logger.info("Torch version: %s", torch.__version__)
    logger.info("CUDA available: %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        logger.info("CUDA device: %s", torch.cuda.get_device_name(0))
    logger.info("")

    # ── Phase A ──
    logger.info("-" * 40)
    logger.info("Phase A: Unit tests (no model)")
    logger.info("-" * 40)
    test_pack_single_pair()
    test_pack_multiple_pairs()
    test_truncation()
    test_scorer_config_defaults()
    logger.info("Phase A: ALL PASSED ✓")

    # ── Phase B (optional) ──
    if args.model_path:
        logger.info("-" * 40)
        logger.info("Phase B: Integration test (with model)")
        logger.info("-" * 40)
        test_real_scoring(args.model_path)
        logger.info("Phase B: PASSED ✓")

    logger.info("=" * 60)
    logger.info("All tests passed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
