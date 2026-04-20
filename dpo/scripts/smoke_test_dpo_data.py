#!/usr/bin/env python
# coding=utf-8
"""Smoke test for incremental DPO data pipeline.

This script verifies increment 1/2 changes:
1) DPO pairwise dataset loading
2) dpo_collate_fn packing outputs

Example:
    python scripts/smoke_test_dpo_data.py --config configs/example.yaml --data_path data/pairwise_train.jsonl
"""

import argparse
import os

from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src import WeDLMTrainingConfig
from src.data import WeDLMPairwiseDataset, dpo_collate_fn


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke test DPO data pipeline")
    parser.add_argument("--config", type=str, default=None, help="Path to config yaml")
    parser.add_argument("--data_path", type=str, default=None, help="Path to pairwise jsonl data")
    parser.add_argument("--batch_size", type=int, default=None, help="Override dataloader batch size")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    return parser.parse_args()


def _print_stream_stats(prefix: str, batch):
    packed_ids = batch[f"{prefix}_packed_input_ids"]
    packed_labels = batch[f"{prefix}_packed_labels"]
    cum = batch[f"{prefix}_cum_seqlens"]
    lengths = (cum[1:] - cum[:-1]).tolist()

    print(f"[{prefix}] packed_input_ids shape: {tuple(packed_ids.shape)}")
    print(f"[{prefix}] packed_labels shape: {tuple(packed_labels.shape)}")
    print(f"[{prefix}] num_sequences: {len(lengths)}")
    print(f"[{prefix}] sequence_lengths: {lengths}")

    if packed_labels.numel() > 0:
        valid = int((packed_labels != -100).sum().item())
        print(f"[{prefix}] valid_label_tokens: {valid}")


def main():
    args = parse_args()
    config = WeDLMTrainingConfig.from_yaml(args.config) if args.config else WeDLMTrainingConfig()

    data_path = args.data_path or config.dpo_train_data or config.train_data
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Pairwise data not found: {data_path}")

    tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=config.trust_remote_code)

    dataset = WeDLMPairwiseDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_seq_length=config.max_seq_length,
        num_learnable_im_end=config.num_learnable_im_end,
    )
    if len(dataset) == 0:
        raise RuntimeError("No valid pairwise samples loaded. Please verify your jsonl schema.")

    batch_size = args.batch_size or config.per_device_train_batch_size
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dpo_collate_fn,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    batch = next(iter(dataloader))

    print("=== DPO Data Pipeline Smoke Test ===")
    print(f"data_path: {data_path}")
    print(f"dataset_size: {len(dataset)}")
    print(f"batch_size: {batch_size}")
    print(f"pair_size (collated): {int(batch['pair_size'].item())}")

    _print_stream_stats("chosen", batch)
    _print_stream_stats("rejected", batch)

    print("Smoke test passed.")


if __name__ == "__main__":
    main()
