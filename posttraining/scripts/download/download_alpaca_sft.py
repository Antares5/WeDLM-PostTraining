#!/usr/bin/env python
# coding=utf-8
"""Download and convert Alpaca dataset to chat SFT format.
Migrated from dpo/scripts/download_alpaca_sft.py.

Usage:
    cd posttraining && python scripts/download/download_alpaca_sft.py
"""

import json
import os
from datasets import load_dataset
from tqdm import tqdm


def format_alpaca_to_chat(example):
    """Convert Alpaca format (instruction, input, output) to Chat SFT format."""
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output_text = example.get("output", "")

    if input_text and input_text.strip():
        user_content = f"{instruction}\n\n{input_text}"
    else:
        user_content = instruction

    chat_format = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": output_text},
    ]
    return chat_format


def main():
    DATASET_NAME = "yahma/alpaca-cleaned"
    OUTPUT_DIR = "data"
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, "alpaca_cleaned_sft.jsonl")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Loading dataset: {DATASET_NAME} ...")

    try:
        dataset = load_dataset(DATASET_NAME, split="train")
    except Exception as e:
        print(f"Download failed. Error: {e}")
        return

    print(f"Dataset loaded. Total {len(dataset)} entries. Converting...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for example in tqdm(dataset):
            f.write(json.dumps(format_alpaca_to_chat(example), ensure_ascii=False) + "\n")

    print(f"Done! File saved as: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
