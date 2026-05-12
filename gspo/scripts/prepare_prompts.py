#!/usr/bin/env python
# coding=utf-8
"""Generate test prompts.jsonl for GSPO from GSM8K dataset.

Usage:
    cd gspo
    python scripts/prepare_prompts.py                    # Download full GSM8K
    python scripts/prepare_prompts.py --num 20            # Download 20 prompts only
    python scripts/prepare_prompts.py --source math       # Use MATH dataset instead
    python scripts/prepare_prompts.py --num 50 --split train  # Use training split
"""

import argparse
import json
import os
import sys


def prepare_gsm8k(num: int = None, split: str = "test", output: str = "data/prompts.jsonl"):
    """Download GSM8K from HuggingFace and convert to GSPO format.

    Each output line: {"messages": [...], "solution": "42", "answer_type": "numeric"}
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' not installed. Run: pip install datasets")
        sys.exit(1)

    print(f"Downloading GSM8K ({split} split)...")
    dataset = load_dataset("openai/gsm8k", "main", split=split)

    samples = []
    for item in dataset:
        question = item["question"]
        # GSM8K answer format: "#### 42" → extract numeric answer
        raw_answer = item["answer"]
        # Extract the numeric answer after "####"
        answer = raw_answer.split("####")[-1].strip() if "####" in raw_answer else raw_answer.strip()

        samples.append({
            "messages": [
                {"role": "user", "content": question}
            ],
            "solution": answer,
        })

    # Limit number of prompts
    if num and num < len(samples):
        samples = samples[:num]

    # Save
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Saved {len(samples)} prompts to {output}")
    print(f"Example: {json.dumps(samples[0], ensure_ascii=False)[:200]}...")


def prepare_math(num: int = None, split: str = "test", output: str = "data/prompts.jsonl"):
    """Download MATH dataset from HuggingFace and convert to GSPO format."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' not installed. Run: pip install datasets")
        sys.exit(1)

    print(f"Downloading MATH ({split} split)...")
    dataset = load_dataset("hendrycks/competition_math", split=split)

    samples = []
    for item in dataset:
        problem = item["problem"]
        solution = item["solution"]
        # Extract \boxed{...} answer
        import re
        boxed_match = re.findall(r"\\boxed\{([^}]*)\}", solution)
        answer = boxed_match[-1].strip() if boxed_match else ""

        if not answer:
            continue  # skip samples without extractable answer

        samples.append({
            "messages": [
                {"role": "user", "content": f"Solve the following math problem step by step. Put your final answer within \\boxed{{}}.\n\n{problem}"}
            ],
            "solution": answer,
        })

    if num and num < len(samples):
        samples = samples[:num]

    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Saved {len(samples)} prompts to {output}")
    print(f"Example: {json.dumps(samples[0], ensure_ascii=False)[:200]}...")


def prepare_from_parquet(parquet_path: str, num: int = None, output: str = "data/prompts.jsonl"):
    """Convert existing DeepMath parquet to GSPO prompt format."""
    try:
        import pandas as pd
    except ImportError:
        print("Error: 'pandas' not installed. Run: pip install pandas pyarrow")
        sys.exit(1)

    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} rows from {parquet_path}")
    print(f"Columns: {list(df.columns)[:10]}...")

    samples = []
    for _, row in df.iterrows():
        item = row.to_dict()

        # Extract messages
        messages = []
        system = item.get("system")
        if system and isinstance(system, str) and system.strip():
            messages.append({"role": "system", "content": system.strip()})

        prompt = item.get("prompt") or item.get("question") or ""
        if prompt:
            messages.append({"role": "user", "content": str(prompt).strip()})

        if not messages:
            continue

        # Extract ground truth
        gt = item.get("solution") or item.get("answer")
        if gt is None:
            continue
        if isinstance(gt, (int, float)):
            gt = str(gt)

        samples.append({
            "messages": messages,
            "solution": str(gt).strip(),
        })

    if num and num < len(samples):
        samples = samples[:num]

    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Saved {len(samples)} prompts to {output}")


def main():
    parser = argparse.ArgumentParser(description="Prepare GSPO prompt data")
    parser.add_argument("--source", type=str, default="gsm8k",
                        choices=["gsm8k", "math", "parquet"],
                        help="Data source")
    parser.add_argument("--num", type=int, default=None,
                        help="Number of prompts (default: all)")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split (for gsm8k/math)")
    parser.add_argument("--parquet_path", type=str, default=None,
                        help="Path to parquet file (for source=parquet)")
    parser.add_argument("--output", type=str, default="data/prompts.jsonl",
                        help="Output file path")
    args = parser.parse_args()

    if args.source == "gsm8k":
        prepare_gsm8k(num=args.num, split=args.split, output=args.output)
    elif args.source == "math":
        prepare_math(num=args.num, split=args.split, output=args.output)
    elif args.source == "parquet":
        if not args.parquet_path:
            print("Error: --parquet_path required for source=parquet")
            sys.exit(1)
        prepare_from_parquet(parquet_path=args.parquet_path, num=args.num, output=args.output)


if __name__ == "__main__":
    main()
