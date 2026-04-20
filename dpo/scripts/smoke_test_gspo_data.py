#!/usr/bin/env python
# coding=utf-8
"""Smoke test for online GSPO prompt data pipeline."""

import argparse
import os

from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src import WeDLMTrainingConfig
from src.data import WeDLMPromptDataset, gspo_prompt_collate_fn


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke test GSPO prompt pipeline")
    parser.add_argument("--config", type=str, default=None, help="Path to config yaml")
    parser.add_argument("--data_path", type=str, default=None, help="Path to prompt jsonl data")
    parser.add_argument("--batch_size", type=int, default=None, help="Override dataloader batch size")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    return parser.parse_args()


def main():
    args = parse_args()
    config = WeDLMTrainingConfig.from_yaml(args.config) if args.config else WeDLMTrainingConfig()

    data_path = args.data_path or config.gspo_train_data or config.dpo_train_data or config.train_data
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Prompt data not found: {data_path}")

    tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=config.trust_remote_code)

    dataset = WeDLMPromptDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_prompt_length=config.gspo_max_prompt_length,
    )
    if len(dataset) == 0:
        raise RuntimeError("No valid prompt samples loaded. Please verify your jsonl schema.")

    batch_size = args.batch_size or config.per_device_train_batch_size
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=gspo_prompt_collate_fn,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    batch = next(iter(dataloader))
    prompt_ids = batch["prompt_input_ids"]
    prompt_lens = [int(x.numel()) for x in prompt_ids]
    prompt_metadata = batch.get("prompt_metadata", [])

    print("=== GSPO Prompt Data Smoke Test ===")
    print(f"data_path: {data_path}")
    print(f"dataset_size: {len(dataset)}")
    print(f"batch_size: {batch_size}")
    print(f"prompt_count (collated): {int(batch['prompt_count'].item())}")
    print(f"prompt_lengths: {prompt_lens}")
    if len(prompt_metadata) > 0:
        first_meta = prompt_metadata[0]
        print(f"first_prompt_metadata_keys: {sorted(list(first_meta.keys()))}")
        if "ground_truth_answer" in first_meta:
            answer_preview = first_meta["ground_truth_answer"][:80].replace("\n", " ")
            print(f"first_ground_truth_answer_preview: {answer_preview}")

    print("GSPO prompt data smoke test passed.")


if __name__ == "__main__":
    main()
