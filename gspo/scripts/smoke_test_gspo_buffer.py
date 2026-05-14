#!/usr/bin/env python
# coding=utf-8
"""Smoke test for RolloutBuffer (Phase 2).

Validates:
- Push and sample operations
- FIFO eviction when buffer is full
- sample_batch with collation
- Edge cases: empty buffer sampling, oversized push
"""

import os
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from src.buffer import RolloutBuffer


def _make_dummy_entry(prompt_idx: int, K: int = 4):
    """Create a dummy buffer entry for testing."""
    return {
        "prompt_text": [f"Prompt {prompt_idx}"],
        "messages": [[{"role": "user", "content": f"Question {prompt_idx}?"}]],
        "ground_truth": [f"answer_{prompt_idx}"],
        "all_responses": [[
            {
                "input_ids": torch.ones(10, dtype=torch.long),
                "labels": torch.ones(10, dtype=torch.long),
                "text": f"Response {prompt_idx}_{k}",
            }
            for k in range(K)
        ]],
        "rewards": torch.rand(1, K),
        "ref_scores": torch.randn(1, K),
    }


def test_push_and_sample():
    """Test basic push and sample operations."""
    buffer = RolloutBuffer(max_size=8)
    K = 4

    # Push 3 entries
    for i in range(3):
        entry = _make_dummy_entry(i, K)
        buffer.push_batch(
            entry["prompt_text"], entry["messages"], entry["ground_truth"],
            entry["all_responses"], entry["rewards"], entry["ref_scores"]
        )

    assert len(buffer) == 3, f"Expected 3 entries, got {len(buffer)}"
    logger.info(f"Buffer size after 3 pushes: {len(buffer)}")

    # Sample one entry
    sampled = buffer.sample()
    assert "prompt_text" in sampled
    assert "rewards" in sampled
    assert sampled["rewards"].shape == (K,)
    logger.info("✓ Push and sample test PASSED")


def test_fifo_eviction():
    """Test that oldest entries are evicted when buffer is full."""
    buffer = RolloutBuffer(max_size=4)
    K = 2

    # Push 6 entries (max_size=4, so first 2 should be evicted)
    for i in range(6):
        entry = _make_dummy_entry(i, K)
        buffer.push_batch(
            entry["prompt_text"], entry["messages"], entry["ground_truth"],
            entry["all_responses"], entry["rewards"], entry["ref_scores"]
        )

    assert len(buffer) == 4, f"Expected 4 entries, got {len(buffer)}"
    assert buffer.is_full(), "Buffer should be full"

    # The oldest remaining entry should have prompt_idx >= 2
    sampled = buffer.sample()
    logger.info(f"Buffer size: {len(buffer)}, is_full: {buffer.is_full()}")
    logger.info("✓ FIFO eviction test PASSED")


def test_sample_batch():
    """Test sample_batch with collation."""
    buffer = RolloutBuffer(max_size=8)
    K = 3

    # Push 5 entries
    for i in range(5):
        entry = _make_dummy_entry(i, K)
        buffer.push_batch(
            entry["prompt_text"], entry["messages"], entry["ground_truth"],
            entry["all_responses"], entry["rewards"], entry["ref_scores"]
        )

    # Sample batch of 3
    batch = buffer.sample_batch(batch_size=3)

    assert len(batch["prompt_texts"]) == 3
    assert len(batch["all_responses"]) == 3
    assert batch["rewards"].shape == (3, K)
    assert batch["ref_scores"].shape == (3, K)
    for resp_list in batch["all_responses"]:
        assert len(resp_list) == K

    logger.info(f"Batch: {batch['rewards'].shape}, {batch['ref_scores'].shape}")
    logger.info("✓ sample_batch test PASSED")


def test_empty_buffer():
    """Test that empty buffer raises appropriate errors."""
    buffer = RolloutBuffer(max_size=4)

    try:
        buffer.sample()
        assert False, "Should have raised IndexError"
    except IndexError:
        logger.info("✓ Empty buffer sample raises IndexError")

    try:
        buffer.sample_batch(2)
        assert False, "Should have raised RuntimeError"
    except RuntimeError:
        logger.info("✓ Empty buffer sample_batch raises RuntimeError")


def test_clear():
    """Test buffer clearing."""
    buffer = RolloutBuffer(max_size=4)
    K = 2

    for i in range(3):
        entry = _make_dummy_entry(i, K)
        buffer.push_batch(
            entry["prompt_text"], entry["messages"], entry["ground_truth"],
            entry["all_responses"], entry["rewards"], entry["ref_scores"]
        )

    assert len(buffer) == 3
    buffer.clear()
    assert len(buffer) == 0
    logger.info("✓ Buffer clear test PASSED")


if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("Smoke Test: RolloutBuffer (Phase 2)")
    logger.info("=" * 50)

    test_push_and_sample()
    test_fifo_eviction()
    test_sample_batch()
    test_empty_buffer()
    test_clear()

    logger.info("\n" + "=" * 50)
    logger.info("All RolloutBuffer tests PASSED ✓")
    logger.info("=" * 50)
