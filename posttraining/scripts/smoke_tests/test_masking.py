#!/usr/bin/env python
# coding=utf-8
"""Smoke test for WeDLM masking utilities (migrated to wedlm_train).

Usage:
    cd posttraining && python scripts/smoke_tests/test_masking.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from wedlm_train.batch.masking import (
    sample_block_mask_ratios,
    sample_mask_indices,
    reorder_block,
    build_2d_attention_mask,
)


def main():
    torch.manual_seed(42)
    device = torch.device("cpu")
    block_size = 8

    # Test block_mask_ratios
    ratios = sample_block_mask_ratios(4, mask_per_block=True, device=device)
    print(f"Block mask ratios (per-block): {ratios}")
    assert ratios.numel() == 4

    ratios_uniform = sample_block_mask_ratios(4, mask_per_block=False, device=device)
    print(f"Block mask ratios (uniform): {ratios_uniform}")
    assert ratios_uniform.numel() == 4
    assert (ratios_uniform == ratios_uniform[0]).all()

    # Test mask_indices
    maskable = torch.tensor([False, True, True, True, False, True], dtype=torch.bool)
    indices = sample_mask_indices(maskable, 0.5, device)
    print(f"Mask indices: {indices}")
    assert indices.sum() == 2  # 0.5 * 4 = 2

    # Test reorder_block
    tokens = torch.tensor([10, 20, 30, 40, 50], dtype=torch.long)
    positions = torch.arange(5, dtype=torch.long)
    mask_indices = torch.tensor([False, True, False, True, False], dtype=torch.bool)
    xt_reordered, orig_reordered, pos_reordered, p_values = reorder_block(
        tokens, positions, mask_indices, 0.4, mask_token_id=999
    )
    print(f"Reordered: xt={xt_reordered}, orig={orig_reordered}, pos={pos_reordered}")
    assert len(xt_reordered) == 5
    # First 3 are unmasked (original), last 2 are masked
    assert (xt_reordered[:3] == tokens[~mask_indices]).all()

    # Test 2d attention mask
    mask_2d = build_2d_attention_mask(4, 2, device)
    print(f"2D attention mask shape: {mask_2d.shape}")  # [8, 8]
    assert mask_2d.shape == (8, 8)

    print("Masking smoke tests all passed!")


if __name__ == "__main__":
    main()
