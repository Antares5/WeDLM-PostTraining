# coding=utf-8
"""Unit tests for DPO and GSPO loss functions."""

import torch
from wedlm_train.loss import compute_dpo_loss, compute_gspo_loss


class TestDPOLoss:
    def test_basic_dpo(self):
        pc = torch.tensor([0.5, 0.3], dtype=torch.float32)
        pr = torch.tensor([0.1, 0.2], dtype=torch.float32)
        rc = torch.tensor([0.0, 0.0], dtype=torch.float32)
        rr = torch.tensor([0.0, 0.0], dtype=torch.float32)

        loss, logs = compute_dpo_loss(pc, pr, rc, rr, beta=0.1)
        assert loss.item() > 0
        assert "dpo/loss" in logs
        assert "dpo/rewards_accuracy" in logs

    def test_shape_mismatch(self):
        pc = torch.tensor([0.5], dtype=torch.float32)
        pr = torch.tensor([0.1, 0.2], dtype=torch.float32)
        rc = torch.tensor([0.0, 0.0], dtype=torch.float32)
        rr = torch.tensor([0.0, 0.0], dtype=torch.float32)

        try:
            compute_dpo_loss(pc, pr, rc, rr, beta=0.1)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_zero_beta(self):
        pc = torch.tensor([0.5], dtype=torch.float32)
        try:
            compute_dpo_loss(pc, pc, pc, pc, beta=0.0)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass


class TestGSPOLoss:
    def test_basic_gspo(self):
        policy = torch.tensor([0.1, 0.3, -0.2, 0.5, 0.2, -0.1], dtype=torch.float32)
        ref = torch.tensor([0.0, 0.1, -0.1, 0.3, 0.15, -0.2], dtype=torch.float32)
        rewards = 0.1 * (policy - ref)
        groups = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)

        loss, logs = compute_gspo_loss(
            policy_scores=policy,
            reference_scores=ref,
            group_ids=groups,
            rewards=rewards,
        )
        assert loss.item() >= 0
        assert logs["gspo/num_valid_groups"] == 2.0

    def test_no_groups(self):
        policy = torch.tensor([0.1, 0.3], dtype=torch.float32)
        ref = torch.tensor([0.0, 0.1], dtype=torch.float32)
        groups = torch.tensor([0, 1], dtype=torch.long)  # size=1 per group

        loss, logs = compute_gspo_loss(policy, ref, groups)
        assert loss.item() == 0.0
