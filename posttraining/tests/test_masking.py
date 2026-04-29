# coding=utf-8
"""Unit tests for WeDLM masking utilities."""

import torch
from wedlm_train.batch.masking import (
    sample_block_mask_ratios,
    sample_mask_indices,
    reorder_block,
    build_2d_attention_mask,
    build_magi_plan,
)


class TestBlockMaskRatios:
    def test_per_block_mode(self, device):
        ratios = sample_block_mask_ratios(4, mask_per_block=True, device=device)
        assert ratios.shape == (4,)
        assert (ratios > 0).all()

    def test_uniform_mode(self, device):
        ratios = sample_block_mask_ratios(4, mask_per_block=False, device=device)
        assert ratios.shape == (4,)
        assert (ratios == ratios[0]).all()


class TestMaskIndices:
    def test_mask_count(self, device):
        maskable = torch.tensor([False, True, True, True, False, True], dtype=torch.bool)
        indices = sample_mask_indices(maskable, 0.5, device)
        assert indices.sum() == 2  # 0.5 * 4

    def test_empty_maskable(self, device):
        maskable = torch.zeros(5, dtype=torch.bool)
        indices = sample_mask_indices(maskable, 0.5, device)
        assert indices.sum() == 0


class TestReorderBlock:
    def test_basic_reorder(self, device):
        tokens = torch.tensor([10, 20, 30, 40, 50], dtype=torch.long)
        positions = torch.arange(5, dtype=torch.long)
        mask_indices = torch.tensor([False, True, False, True, False], dtype=torch.bool)
        xt, orig, pos, p = reorder_block(tokens, positions, mask_indices, 0.4, 999)
        assert len(xt) == 5
        # Unmasked tokens first, then masked
        assert (xt[:3] == tokens[~mask_indices]).all()
        assert (orig[:3] == tokens[~mask_indices]).all()


class TestAttentionMask:
    def test_2d_mask_shape(self, device):
        mask = build_2d_attention_mask(4, 2, device)
        assert mask.shape == (8, 8)

    def test_empty_seq(self, device):
        mask = build_2d_attention_mask(0, 2, device)
        assert mask.shape == (0, 0)


class TestMagiPlan:
    def test_empty_input(self, device):
        base_cum = torch.tensor([0], dtype=torch.long, device=device)
        packed_cum = torch.tensor([0], dtype=torch.long, device=device)
        plan = build_magi_plan(base_cum, packed_cum, 32, device)
        assert plan["q_ranges"].shape == (0, 2)
