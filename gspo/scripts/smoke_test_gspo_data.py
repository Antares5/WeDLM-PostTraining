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


def test_deepmath_parquet():
    """Test DeepMath parquet format (prompt column as list of {role, content} dicts)."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        logger.warning("pyarrow not installed, skipping DeepMath parquet test")
        return

    from src.data import GSPOPromptDataset, GSPOCollateFunction, get_im_end_token_id

    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = os.path.join(tmpdir, "test_deepmath.parquet")

        # Create parquet with DeepMath schema:
        # prompt: list<struct<content: string, role: string>>
        # solution: string
        prompt_type = pa.list_(pa.struct([
            ("content", pa.string()),
            ("role", pa.string()),
        ]))
        schema = pa.schema([
            ("prompt", prompt_type),
            ("solution", pa.string()),
        ])

        data = [
            {
                "prompt": [{"content": "What is 2 + 2?", "role": "user"}],
                "solution": "4",
            },
            {
                "prompt": [{"content": "Solve: x^2 = 4. What is x?", "role": "user"}],
                "solution": "$2$",
            },
            {
                "prompt": [{"content": "Is pi greater than 3?", "role": "user"}],
                "solution": "Yes",
            },
        ]
        table = pa.Table.from_pylist(data, schema=schema)
        pq.write_table(table, data_path)
        logger.info(f"Created test DeepMath parquet: {len(data)} rows")

        # Load tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen2.5-0.5B", trust_remote_code=True
            )
        except Exception:
            logger.warning("Cannot load test tokenizer, skipping DeepMath parquet test")
            return

        # Create dataset with deepmath format
        dataset = GSPOPromptDataset(
            data_path=data_path,
            tokenizer=tokenizer,
            max_seq_length=512,
            prompt_format="deepmath",
        )

        logger.info(f"DeepMath dataset size: {len(dataset)}")
        assert len(dataset) == 3, f"Expected 3 samples, got {len(dataset)}"

        # Check all samples
        for i in range(len(dataset)):
            sample = dataset[i]
            assert "input_ids" in sample
            assert "ground_truth" in sample
            logger.info(f"  Sample {i}: input_ids shape={sample['input_ids'].shape}, "
                        f"ground_truth='{sample['ground_truth']}'")

        # Verify ground truth extraction
        assert dataset[0]["ground_truth"] == "4"
        assert dataset[1]["ground_truth"] == "$2$"
        assert dataset[2]["ground_truth"] == "Yes"

        # Test collate function
        im_end_id = get_im_end_token_id(tokenizer)
        collate_fn = GSPOCollateFunction(pad_token_id=im_end_id)
        batch = collate_fn([dataset[0], dataset[1], dataset[2]])
        assert batch["input_ids"].dim() == 2
        assert batch["input_ids"].size(0) == 3
        assert len(batch["ground_truths"]) == 3

        # Test DataLoader iteration
        dataloader = DataLoader(
            dataset, batch_size=2, collate_fn=collate_fn, shuffle=False
        )
        for i, batch in enumerate(dataloader):
            logger.info(f"DeepMath Batch {i}: bs={batch['input_ids'].size(0)}, "
                        f"ground_truths={batch['ground_truths']}")

        logger.info("✓ DeepMath parquet test PASSED")


def test_deepmath_reward():
    """Test MathReward with DeepMath answer formats (LaTeX, Boolean)."""
    from src.reward import MathReward

    reward = MathReward(reward_type="math_verify")

    # Test 1: LaTeX $...$ answer extraction
    test_cases = [
        # (response_text, ground_truth, expected_reward)
        # DeepMath style: model outputs $...$ inline math
        ("The limit evaluates to $0$.", "$0$", 1.0),
        ("Therefore, the answer is $\\frac{1}{5}$.", "$\\frac{1}{5}$", 1.0),
        ("The value is $\\sqrt{2\\pi}$.", "$\\sqrt{2\\pi}$", 1.0),
        # Boolean answers
        ("The construction is possible. So the answer is Yes.", "Yes", 1.0),
        ("No such set exists. Therefore, no.", "No", 1.0),
        # Boxed answers (model uses \boxed but ground truth is $...$)
        ("The final answer is \\boxed{2}.", "$2$", 1.0),
        # Numeric within tolerance
        ("The answer is approximately 3.14159.", "3.1416", 1.0),
        # Wrong answers
        ("The answer is $5$.", "$3$", 0.0),
        ("The limit is $1$.", "$0$", 0.0),
    ]

    for response, gt, expected in test_cases:
        r = reward._compute_single_reward(response, gt)
        status = "✓" if r == expected else "✗"
        logger.info(f"  {status} reward={r} (expected={expected}): "
                    f"response='{response[:60]}...' gt='{gt}'")
        if r != expected:
            logger.warning(f"    Extracted: '{reward.extract_answer(response)}'")
            logger.warning(f"    Verified: {reward.verify_answer(reward.extract_answer(response), gt)}")

    # Bulk assert
    for response, gt, expected in test_cases:
        r = reward._compute_single_reward(response, gt)
        assert r == expected, f"Mismatch: response='{response[:50]}...' gt='{gt}' got={r} expected={expected}"

    logger.info("✓ DeepMath reward test PASSED")


if __name__ == "__main__":
    test_dataset()
    test_empty_data()
    test_deepmath_parquet()
    test_deepmath_reward()
    logger.info("All GSPO data tests passed!")
