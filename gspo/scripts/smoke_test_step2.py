#!/usr/bin/env python
# coding=utf-8
"""Smoke test for Step 2: batch construction + model forward + score + loss pipeline.

Run from the project root:
    # CPU-only quick test (no model needed):
    python gspo/scripts/smoke_test_step2.py

    # Full test with model (requires GPU):
    python gspo/scripts/smoke_test_step2.py --model_path /path/to/model

This test validates:
  1. All gspo.src modules import correctly (no external deps on dpo/finetune).
  2. build_wedlm_batch produces correct shapes.
  3. compute_block_scores runs on random logits.
  4. The full pipeline: logits → scores → GSPO loss → backward.
  5. (Optional, with --model_path) Real model forward pass.
"""

import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import argparse
import torch


# ── helpers ──

def green(s: str) -> str: return f"\033[32m{s}\033[0m"
def red(s: str) -> str:   return f"\033[31m{s}\033[0m"

def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

passed = 0
failed = 0

def check(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        print(f"  {green('[PASS]')} {name}")
        passed += 1
    else:
        print(f"  {red('[FAIL]')} {name}" + (f"  → {detail}" if detail else ""))
        failed += 1


# ═══════════════════════════════════════════════════════════════
# TEST 1: Module imports (all self-contained)
# ═══════════════════════════════════════════════════════════════

section("Test 1: Module imports")

try:
    from gspo.src.config import GSPOConfig
    check("1-import: config", True)
except Exception as e:
    check("1-import: config", False, str(e))

try:
    from gspo.src.loss import compute_gspo_loss, compute_block_scores
    check("1-import: loss (gspo + block_scores)", True)
except Exception as e:
    check("1-import: loss", False, str(e))

try:
    from gspo.src.batch import WeDLMBatch, build_wedlm_batch, build_wedlm_batch_from_response
    check("1-import: batch", True)
except Exception as e:
    check("1-import: batch", False, str(e))

try:
    from gspo.src.model import wedlm_forward, wedlm_attention_forward
    check("1-import: model", True)
except Exception as e:
    check("1-import: model", False, str(e))

try:
    from gspo.src.attention import check_backend_available, get_available_backend, get_attention_wrapper
    check("1-import: attention", True)
except Exception as e:
    check("1-import: attention", False, str(e))

try:
    from gspo.src.data import GSPOPromptDataset, gspo_collate_fn
    check("1-import: data", True)
except Exception as e:
    check("1-import: data", False, str(e))

try:
    from gspo.src.masking import build_2d_attention_mask, reorder_block
    check("1-import: masking", True)
except Exception as e:
    check("1-import: masking", False, str(e))

# Verify no dpo/finetune imports leaked into gspo
import gspo.src.config as cfg_mod
import gspo.src.batch as bat_mod
import gspo.src.model as mod_mod
import gspo.src.attention as att_mod

for mod, name in [(cfg_mod, "config"), (bat_mod, "batch"), (mod_mod, "model"), (att_mod, "attention")]:
    src = open(mod.__file__).read()
    has_dpo_import = "from dpo" in src or "import dpo" in src
    has_finetune_import = "from finetune" in src or "import finetune" in src
    check(f"1-no-leak: {name} has no dpo import", not has_dpo_import,
          "Found 'from dpo' or 'import dpo'" if has_dpo_import else "")
    check(f"1-no-leak: {name} has no finetune import", not has_finetune_import,
          "Found 'from finetune' or 'import finetune'" if has_finetune_import else "")


# ═══════════════════════════════════════════════════════════════
# TEST 2: build_wedlm_batch shape correctness
# ═══════════════════════════════════════════════════════════════

section("Test 2: build_wedlm_batch")

from gspo.src.batch import build_wedlm_batch

MASK_TOKEN_ID = 151665
BLOCK_SIZE = 32
device = torch.device("cpu")

# Create 2 sequences: [10, 20] tokens
seq1 = torch.randint(0, 50000, (10,))
seq2 = torch.randint(0, 50000, (20,))
packed = torch.cat([seq1, seq2])
labels = torch.cat([seq1.clone(), seq2.clone()])  # no prompt masking
cum_seqlens = torch.tensor([0, 10, 30])

batch = build_wedlm_batch(
    packed_input_ids=packed,
    packed_labels=labels,
    cum_seqlens=cum_seqlens,
    block_size=BLOCK_SIZE,
    mask_token_id=MASK_TOKEN_ID,
    mask_per_block=True,
    backend="dense",
    eps=1e-8,
)

# Total tokens after dual-stream: 2 * sum(L_i) = 2 * 30 = 60
expected_total = 60
check("2-shape: packed_input_ids total", batch.packed_input_ids.size(0) == expected_total,
      f"expected {expected_total}, got {batch.packed_input_ids.size(0)}")
check("2-shape: original_ids == packed_input_ids shape",
      batch.original_ids.shape == batch.packed_input_ids.shape)
check("2-shape: logical_positions shape",
      batch.logical_positions.shape == batch.packed_input_ids.shape)
check("2-shape: masked_indices shape",
      batch.masked_indices.shape == batch.packed_input_ids.shape)
check("2-shape: p_mask shape",
      batch.p_mask.shape == batch.packed_input_ids.shape)
check("2-shape: cum_seqlens", batch.cum_seqlens.numel() == 3,
      f"expected 3, got {batch.cum_seqlens.numel()}")
check("2-shape: base_cum_seqlens", batch.base_cum_seqlens.numel() == 3)

# At least some positions should be masked
check("2-mask: has masked tokens", batch.masked_indices.sum().item() > 0,
      f"masked count = {batch.masked_indices.sum().item()}")

# Attention mask should be square
if batch.attn_mask_2d is not None:
    T = batch.packed_input_ids.size(0)
    check("2-attn: mask is square", batch.attn_mask_2d.shape == (T, T),
          f"shape = {batch.attn_mask_2d.shape}")


# ═══════════════════════════════════════════════════════════════
# TEST 3: build_wedlm_batch_from_response
# ═══════════════════════════════════════════════════════════════

section("Test 3: build_wedlm_batch_from_response")

from gspo.src.batch import build_wedlm_batch_from_response

response = torch.randint(0, 50000, (50,))  # prompt(10) + completion(40)
prompt_len = 10
batch2 = build_wedlm_batch_from_response(
    response_ids=response,
    prompt_len=prompt_len,
    block_size=BLOCK_SIZE,
    mask_token_id=MASK_TOKEN_ID,
    backend="dense",
)

check("3-resp: batch created", True)
check("3-resp: total > 0", batch2.packed_input_ids.size(0) > 0)
check("3-resp: no mask on x0 stream",
      batch2.masked_indices[:batch2.packed_input_ids.size(0) // 2].sum().item() == 0)


# ═══════════════════════════════════════════════════════════════
# TEST 4: compute_block_scores on random logits
# ═══════════════════════════════════════════════════════════════

section("Test 4: compute_block_scores (random logits)")

from gspo.src.loss import compute_block_scores

V = 50000  # fake vocab size
logits = torch.randn(batch.packed_input_ids.size(0), V)
scores, logs = compute_block_scores(
    logits=logits,
    targets=batch.original_ids,
    masked_indices=batch.masked_indices,
    p_mask=batch.p_mask,
    logical_positions=batch.logical_positions,
    cum_seqlens=batch.cum_seqlens,
    block_size=BLOCK_SIZE,
    weighting_scheme="weighted",
    block_reduce="mean",
    seq_reduce="mean",
    eps=1e-8,
)

check("4-scores: shape [bs]", scores.dim() == 1 and scores.size(0) == 2,
      f"shape = {scores.shape}")
check("4-scores: finite", torch.all(torch.isfinite(scores)).item())
check("4-logs: score/mean present", "score/mean" in logs)

# Gradient test
logits_grad = torch.randn(batch.packed_input_ids.size(0), V, requires_grad=True)
scores_g, _ = compute_block_scores(
    logits=logits_grad,
    targets=batch.original_ids,
    masked_indices=batch.masked_indices,
    p_mask=batch.p_mask,
    logical_positions=batch.logical_positions,
    cum_seqlens=batch.cum_seqlens,
    block_size=BLOCK_SIZE,
    weighting_scheme="weighted",
    block_reduce="mean",
    seq_reduce="mean",
    eps=1e-8,
)
loss_g = scores_g.mean()
loss_g.backward()
check("4-grad: logits.grad exists", logits_grad.grad is not None)
check("4-grad: logits.grad finite", torch.all(torch.isfinite(logits_grad.grad)).item())


# ═══════════════════════════════════════════════════════════════
# TEST 5: Full pipeline (scores → GSPO loss → backward)
# ═══════════════════════════════════════════════════════════════

section("Test 5: Full pipeline (scores + GSPO loss + backward)")

from gspo.src.loss import compute_gspo_loss

# Simulate a batch with B=2 prompts, G=3 responses each
# Create 6 individual responses and score them
all_scores = []
all_rewards = []
all_prompt_idx = []

torch.manual_seed(42)
for b in range(2):
    for g in range(3):
        # Create a fake response
        resp_len = torch.randint(20, 80, (1,)).item()
        resp = torch.randint(0, 50000, (resp_len,))
        prompt_l = 10
        labels_r = resp.clone()
        labels_r[:prompt_l] = -100
        cum = torch.tensor([0, resp_len])

        resp_batch = build_wedlm_batch(
            packed_input_ids=resp,
            packed_labels=labels_r,
            cum_seqlens=cum,
            block_size=BLOCK_SIZE,
            mask_token_id=MASK_TOKEN_ID,
            mask_per_block=True,
            backend="dense",
        )
        logits_r = torch.randn(resp_batch.packed_input_ids.size(0), V)
        score_r, _ = compute_block_scores(
            logits=logits_r,
            targets=resp_batch.original_ids,
            masked_indices=resp_batch.masked_indices,
            p_mask=resp_batch.p_mask,
            logical_positions=resp_batch.logical_positions,
            cum_seqlens=resp_batch.cum_seqlens,
            block_size=BLOCK_SIZE,
        )
        all_scores.append(score_r[0])  # single-sequence score
        all_rewards.append(torch.randn(1).item())  # fake reward
        all_prompt_idx.append(b)

scores_t = torch.stack(all_scores)
rewards_t = torch.tensor(all_rewards)
prompt_t = torch.tensor(all_prompt_idx)

# Gradient test
scores_t_grad = scores_t.clone().detach().requires_grad_(True)
loss, logs5 = compute_gspo_loss(scores_t_grad, rewards_t, prompt_t)
loss.backward()
check("5-pipeline: loss scalar", loss.dim() == 0)
check("5-pipeline: loss finite", torch.isfinite(loss))
check("5-pipeline: scores.grad exists", scores_t_grad.grad is not None)
check("5-pipeline: scores.grad finite", torch.all(torch.isfinite(scores_t_grad.grad)).item())
check("5-pipeline: gspo/loss in logs", "gspo/loss" in logs5)

# Check that higher rewards → positive or negative advantage is group-normalized
adv_mean = logs5["gspo/adv_mean"].item()
check("5-pipeline: adv_mean ≈ 0 (group-normalized)", abs(adv_mean) < 1e-6,
      f"adv_mean = {adv_mean:.8f}")

print(f"  5-pipeline: loss={loss.detach().item():.6f}, adv_mean={adv_mean:.6f}")


# ═══════════════════════════════════════════════════════════════
# TEST 6: (Optional) Real model forward
# ═══════════════════════════════════════════════════════════════

section("Test 6: Real model forward (optional)")

args = argparse.Namespace()
# We try to parse --model_path manually
try:
    _parser = argparse.ArgumentParser()
    _parser.add_argument("--model_path", type=str, default=None)
    _args, _ = _parser.parse_known_args()
    args.model_path = _args.model_path
except Exception:
    args.model_path = None

if args.model_path:
    print(f"  Model path: {args.model_path}")
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from gspo.src.attention import get_attention_wrapper, get_available_backend

        backend = get_available_backend()
        print(f"  Using backend: {backend}")

        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            device_map="auto" if torch.cuda.is_available() else None,
        )
        model.eval()

        # Determine the primary device of the model
        model_device = next(model.parameters()).device
        print(f"  Model device: {model_device}")

        head_dim = model.config.hidden_size // model.config.num_attention_heads
        wrapper = get_attention_wrapper(backend, head_dim, deterministic=False)

        # Build a small batch on the same device as the model
        test_ids = torch.randint(0, tokenizer.vocab_size, (64,), device=model_device)
        test_labels = test_ids.clone()
        test_cum = torch.tensor([0, 64], device=model_device)

        test_batch = build_wedlm_batch(
            packed_input_ids=test_ids,
            packed_labels=test_labels,
            cum_seqlens=test_cum,
            block_size=BLOCK_SIZE,
            mask_token_id=MASK_TOKEN_ID,
            backend=backend,
        )

        with torch.no_grad():
            logits_real = wedlm_forward(
                model, test_batch, wrapper, backend
            )
        check("6-real: logits shape correct",
              logits_real.dim() == 2 and logits_real.size(1) == tokenizer.vocab_size,
              f"shape = {logits_real.shape}")
        check("6-real: logits finite", torch.all(torch.isfinite(logits_real)).item())

        scores_real, _ = compute_block_scores(
            logits=logits_real,
            targets=test_batch.original_ids,
            masked_indices=test_batch.masked_indices,
            p_mask=test_batch.p_mask,
            logical_positions=test_batch.logical_positions,
            cum_seqlens=test_batch.cum_seqlens,
            block_size=BLOCK_SIZE,
        )
        check("6-real: scores shape [1]", scores_real.shape == (1,))
        check("6-real: scores finite", torch.all(torch.isfinite(scores_real)).item())
        print(f"  6-real: score = {scores_real[0].item():.6f}")

    except Exception as e:
        import traceback
        print(f"  {red('[SKIP]')} Real model test failed:")
        traceback.print_exc()
        check("6-real: overall", False, str(e))
else:
    print(f"  {green('[SKIP]')} No --model_path provided. "
          "Run with --model_path /path/to/model for full test.")


# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════

section("Summary")
total = passed + failed
print(f"  {passed}/{total} tests passed")
if failed == 0:
    print(f"\n  {green('ALL TESTS PASSED — Step 2 is ready.')}")
else:
    print(f"\n  {red(f'{failed} TEST(S) FAILED — fix before proceeding.')}")

sys.exit(0 if failed == 0 else 1)
