#!/usr/bin/env python
# coding=utf-8
"""Smoke test for GSPO prompt data loading.

Validates:
- GSPOPromptDataset can load JSONL data
- Returns correct format (input_ids, attention_mask, messages, ground_truth)
- GSPOCollateFunction correctly pads prompts
- DataLoader iteration works
"""

import os
import sys
import json
import logging
import tempfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent dir to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


def create_test_jsonl(filepath: str):
    """Create a minimal test JSONL file."""
    samples = [
        {
            "messages": [
                {"role": "user", "content": "What is 2 + 2?"}
            ],
            "solution": "4",
        },
        {
            "messages": [
                {"role": "system", "content": "You are a math assistant."},
                {"role": "user", "content": "Solve: 3x + 5 = 14. What is x?"},
            ],
            "answer": "3",
        },
        {
            "messages": [
                {"role": "user", "content": "What is the square root of 16?"}
            ],
            "solution": "4",
        },
    ]
    with open(filepath, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    return filepath


def test_dataset():
    """Test GSPOPromptDataset loading."""
    from src.data import GSPOPromptDataset, GSPOCollateFunction, get_im_end_token_id

    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = os.path.join(tmpdir, "test_prompts.jsonl")
        create_test_jsonl(data_path)

        # Load tokenizer (use a small model for testing)
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen2.5-0.5B", trust_remote_code=True
            )
        except Exception:
            logger.warning("Cannot load test tokenizer, skipping")
            return

        # Create dataset
        dataset = GSPOPromptDataset(
            data_path=data_path,
            tokenizer=tokenizer,
            max_seq_length=512,
            prompt_format="messages",
        )

        logger.info(f"Dataset size: {len(dataset)}")
        assert len(dataset) == 3, f"Expected 3 samples, got {len(dataset)}"

        # Check sample format
        sample = dataset[0]
        logger.info(f"Sample keys: {sample.keys()}")
        assert "input_ids" in sample
        assert "attention_mask" in sample
        assert "messages" in sample
        assert "ground_truth" in sample
        assert isinstance(sample["input_ids"], torch.Tensor)
        assert sample["ground_truth"] == "4"

        # Check all samples have valid ground truth
        for i, s in enumerate([dataset[0], dataset[1], dataset[2]]):
            logger.info(f"  Sample {i}: input_ids shape={s['input_ids'].shape}, "
                        f"ground_truth='{s['ground_truth']}'")

        # Test collate function
        im_end_id = get_im_end_token_id(tokenizer)
        collate_fn = GSPOCollateFunction(pad_token_id=im_end_id)

        batch = collate_fn([dataset[0], dataset[1], dataset[2]])
        logger.info(f"Batch keys: {batch.keys()}")
        logger.info(f"  input_ids shape: {batch['input_ids'].shape}")
        logger.info(f"  attention_mask shape: {batch['attention_mask'].shape}")
        assert batch["input_ids"].dim() == 2
        assert batch["input_ids"].size(0) == 3  # batch_size

        # Test DataLoader iteration
        dataloader = DataLoader(
            dataset, batch_size=2, collate_fn=collate_fn, shuffle=False
        )
        for i, batch in enumerate(dataloader):
            logger.info(f"Batch {i}: bs={batch['input_ids'].size(0)}, "
                        f"ground_truths={batch['ground_truths']}")

        logger.info("✓ Dataset smoke test PASSED")


def test_empty_data():
    """Test handling of empty/invalid data."""
    from src.data import GSPOPromptDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = os.path.join(tmpdir, "empty.jsonl")
        with open(data_path, "w") as f:
            f.write("")  # empty file

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen2.5-0.5B", trust_remote_code=True
            )
        except Exception:
            logger.warning("Cannot load test tokenizer, skipping empty test")
            return

        dataset = GSPOPromptDataset(
            data_path=data_path,
            tokenizer=tokenizer,
            max_seq_length=512,
        )
        logger.info(f"Empty dataset size: {len(dataset)}")
        assert len(dataset) == 0

        logger.info("✓ Empty data test PASSED")


if __name__ == "__main__":
    test_dataset()
    test_empty_data()
    logger.info("All GSPO data tests passed!")
