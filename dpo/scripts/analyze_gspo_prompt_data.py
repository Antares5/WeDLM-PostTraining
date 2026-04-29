#!/usr/bin/env python
# coding=utf-8
"""Data analysis script for GSPO prompt datasets (jsonl/parquet)."""

import argparse
import hashlib
import math
import os
from collections import Counter
from typing import Any, Dict, List, Optional

from transformers import AutoTokenizer

from src import WeDLMTrainingConfig
from src.data import WeDLMPromptDataset
from src.reward import extract_math_final_answer


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze GSPO prompt data quality")
    parser.add_argument("--config", type=str, default=None, help="Path to config yaml")
    parser.add_argument("--data_path", type=str, default=None, help="Path to prompt data (jsonl/parquet)")
    parser.add_argument(
        "--max_items",
        type=int,
        default=0,
        help="Max number of raw items to scan. 0 means scan all.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Top-k categories to print for each distribution",
    )
    parser.add_argument(
        "--risk_examples",
        type=int,
        default=5,
        help="Max number of high-risk sample previews per risk type",
    )
    return parser.parse_args()


def _percentile(sorted_vals: List[int], q: float) -> float:
    if len(sorted_vals) == 0:
        return 0.0

    idx = q * (len(sorted_vals) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return float(sorted_vals[lo])
    frac = idx - lo
    return float(sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac)


def _safe_ratio(num: int, den: int) -> float:
    if den <= 0:
        return 0.0
    return float(num) / float(den)


def _normalize_for_compare(text: str) -> str:
    normalized = text.strip()
    normalized = normalized.replace("$", "")
    normalized = normalized.replace(" ", "")
    normalized = normalized.replace("\\,", "")
    normalized = normalized.rstrip(".。")
    return normalized


def _preview_text(text: str, max_len: int = 120) -> str:
    single_line = text.replace("\n", " ").strip()
    if len(single_line) <= max_len:
        return single_line
    return single_line[:max_len] + "..."


def _item_keys_summary(item: Any, top_k: int = 12) -> str:
    if isinstance(item, dict):
        keys = sorted(str(k) for k in item.keys())
        if len(keys) <= top_k:
            return ", ".join(keys)
        return ", ".join(keys[:top_k]) + ", ..."
    if isinstance(item, list):
        return "list"
    return type(item).__name__


def _answer_type(answer_text: str) -> str:
    normalized = answer_text.strip().lower()
    if normalized == "":
        return "empty"

    yes_no_set = {"yes", "no", "true", "false", "y", "n"}
    if normalized in yes_no_set:
        return "yes_no"

    compact = normalized.replace(" ", "")
    numeric_candidate = compact.replace(",", "")
    if numeric_candidate.startswith("+") or numeric_candidate.startswith("-"):
        numeric_candidate = numeric_candidate[1:]
    if numeric_candidate.replace(".", "", 1).isdigit():
        return "numeric"

    if any(token in normalized for token in ["\\frac", "\\sqrt", "=", "/", "^", "pi", "\u221a"]):
        return "math_expression"

    if len(normalized) <= 16:
        return "short_text"

    return "long_text"


def _detect_prompt_source(item: Any, analyzer: WeDLMPromptDataset) -> str:
    if isinstance(item, list):
        messages = analyzer._to_messages(item, "user")
        if messages is not None:
            return "chat_list"
        return "chat_list_invalid"

    if not isinstance(item, dict):
        return "invalid_item"

    if "messages" in item:
        messages = analyzer._to_messages(item.get("messages"), "user")
        if messages is not None:
            return "messages"
        return "messages_invalid"

    if analyzer._normalize_text(item.get("question")) is not None:
        return "question"

    if analyzer._normalize_text(item.get("problem")) is not None:
        return "problem"

    if "prompt" in item:
        prompt_value = item.get("prompt")
        prompt_messages = analyzer._to_messages(prompt_value, "user")
        if prompt_messages is not None:
            if isinstance(prompt_value, str):
                return "prompt_text"
            return "prompt_messages"
        if prompt_value is not None:
            return "prompt_unrecognized"

    has_instruction = analyzer._normalize_text(item.get("instruction")) is not None
    has_input = analyzer._normalize_text(item.get("input")) is not None
    if has_instruction and has_input:
        return "instruction_plus_input"
    if has_instruction:
        return "instruction"
    if has_input:
        return "input"

    chosen = item.get("chosen")
    chosen_messages = analyzer._to_messages(chosen, "assistant")
    if chosen_messages is not None and len(chosen_messages) > 0:
        first = chosen_messages[0]
        if isinstance(first, dict) and first.get("role") in {"user", "system"}:
            return "chosen_fallback"
        return "chosen_assistant_only"

    if analyzer._extract_prompt_messages(item) is not None:
        return "extractor_resolved_other"

    return "unknown"


def _print_counter(title: str, counter: Counter, total: int, top_k: int):
    print(f"\n[{title}]")
    if total == 0:
        print("  (empty)")
        return

    for key, cnt in counter.most_common(top_k):
        ratio = 100.0 * _safe_ratio(cnt, total)
        print(f"  {key}: {cnt} ({ratio:.2f}%)")


def _print_examples(title: str, examples: List[Dict[str, Any]]):
    print(f"\n[{title}]")
    if len(examples) == 0:
        print("  (empty)")
        return

    for idx, ex in enumerate(examples, start=1):
        print(
            f"  {idx}. item={ex['item_idx']} source={ex['source']} "
            f"field={ex['field']} raw='{ex['raw']}' extracted='{ex['extracted']}'"
        )
        print(f"     keys: {ex['keys']}")


def main():
    args = parse_args()
    config = WeDLMTrainingConfig.from_yaml(args.config) if args.config else WeDLMTrainingConfig()

    data_path = args.data_path or config.gspo_train_data or config.dpo_train_data or config.train_data
    if not data_path or not os.path.exists(data_path):
        raise FileNotFoundError(f"Prompt data not found: {data_path}")

    tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=config.trust_remote_code)

    # Reuse the exact extraction/tokenization logic from WeDLMPromptDataset
    analyzer = WeDLMPromptDataset.__new__(WeDLMPromptDataset)
    analyzer.tokenizer = tokenizer
    analyzer.max_prompt_length = config.gspo_max_prompt_length

    raw_items = 0
    valid_items = 0
    prompt_lengths: List[int] = []
    truncated_count = 0
    duplicate_count = 0
    prompt_hashes = set()

    drop_reasons = Counter()
    prompt_source_counter = Counter()
    metadata_field_counter = Counter()
    answer_type_counter = Counter()
    extracted_answer_type_counter = Counter()

    with_ground_truth = 0
    extractable_ground_truth = 0
    empty_ground_truth_after_extract = 0
    extraction_changed = 0
    unknown_source_valid = 0

    risk_samples: Dict[str, List[Dict[str, Any]]] = {
        "unknown_source": [],
        "empty_extract": [],
        "long_raw_answer": [],
        "changed_extract": [],
    }

    for item_idx, item in analyzer._iter_data_items(data_path):
        if args.max_items > 0 and raw_items >= args.max_items:
            break

        _ = item_idx
        raw_items += 1

        prompt_source = _detect_prompt_source(item, analyzer)
        prompt_source_counter[prompt_source] += 1

        metadata = analyzer._extract_prompt_metadata(item)
        answer_text = ""
        extracted = ""
        if "ground_truth_field" in metadata:
            metadata_field_counter[metadata["ground_truth_field"]] += 1
        if "ground_truth_answer" in metadata:
            with_ground_truth += 1
            answer_text = metadata["ground_truth_answer"]
            answer_type_counter[_answer_type(answer_text)] += 1
            extracted = extract_math_final_answer(answer_text)
            extracted_answer_type_counter[_answer_type(extracted)] += 1
            if extracted == "":
                empty_ground_truth_after_extract += 1
            else:
                extractable_ground_truth += 1

            normalized_raw = _normalize_for_compare(answer_text)
            if extracted != normalized_raw:
                extraction_changed += 1

        prompt_messages = analyzer._extract_prompt_messages(item)
        if prompt_messages is None or len(prompt_messages) == 0:
            drop_reasons["prompt_extract_failed"] += 1
            if (
                prompt_source == "unknown"
                and len(risk_samples["unknown_source"]) < args.risk_examples
            ):
                risk_samples["unknown_source"].append(
                    {
                        "item_idx": item_idx,
                        "source": prompt_source,
                        "field": metadata.get("ground_truth_field", ""),
                        "raw": _preview_text(answer_text),
                        "extracted": _preview_text(extracted),
                        "keys": _item_keys_summary(item),
                    }
                )
            continue

        prompt_ids = analyzer._tokenize_prompt_messages(prompt_messages)
        if prompt_ids is None or int(prompt_ids.numel()) == 0:
            drop_reasons["tokenize_empty"] += 1
            continue

        valid_items += 1
        length = int(prompt_ids.numel())
        prompt_lengths.append(length)
        if length >= analyzer.max_prompt_length:
            truncated_count += 1

        if prompt_source == "unknown":
            unknown_source_valid += 1
            if len(risk_samples["unknown_source"]) < args.risk_examples:
                risk_samples["unknown_source"].append(
                    {
                        "item_idx": item_idx,
                        "source": prompt_source,
                        "field": metadata.get("ground_truth_field", ""),
                        "raw": _preview_text(answer_text),
                        "extracted": _preview_text(extracted),
                        "keys": _item_keys_summary(item),
                    }
                )

        if extracted == "" and len(risk_samples["empty_extract"]) < args.risk_examples:
            risk_samples["empty_extract"].append(
                {
                    "item_idx": item_idx,
                    "source": prompt_source,
                    "field": metadata.get("ground_truth_field", ""),
                    "raw": _preview_text(answer_text),
                    "extracted": _preview_text(extracted),
                    "keys": _item_keys_summary(item),
                }
            )

        if (
            _answer_type(answer_text) == "long_text"
            and len(risk_samples["long_raw_answer"]) < args.risk_examples
        ):
            risk_samples["long_raw_answer"].append(
                {
                    "item_idx": item_idx,
                    "source": prompt_source,
                    "field": metadata.get("ground_truth_field", ""),
                    "raw": _preview_text(answer_text),
                    "extracted": _preview_text(extracted),
                    "keys": _item_keys_summary(item),
                }
            )

        if (
            answer_text
            and extracted != _normalize_for_compare(answer_text)
            and len(risk_samples["changed_extract"]) < args.risk_examples
        ):
            risk_samples["changed_extract"].append(
                {
                    "item_idx": item_idx,
                    "source": prompt_source,
                    "field": metadata.get("ground_truth_field", ""),
                    "raw": _preview_text(answer_text),
                    "extracted": _preview_text(extracted),
                    "keys": _item_keys_summary(item),
                }
            )

        digest = hashlib.sha1(prompt_ids.detach().cpu().numpy().tobytes()).hexdigest()
        if digest in prompt_hashes:
            duplicate_count += 1
        else:
            prompt_hashes.add(digest)

    prompt_lengths_sorted = sorted(prompt_lengths)
    mean_len = float(sum(prompt_lengths) / len(prompt_lengths)) if prompt_lengths else 0.0
    std_len = 0.0
    if prompt_lengths:
        var = sum((x - mean_len) ** 2 for x in prompt_lengths) / len(prompt_lengths)
        std_len = math.sqrt(var)

    print("=== GSPO Prompt Data Analysis ===")
    print(f"data_path: {data_path}")
    print(f"raw_items_scanned: {raw_items}")
    print(f"valid_prompts: {valid_items}")
    print(f"valid_ratio: {_safe_ratio(valid_items, raw_items):.4f}")
    print(f"invalid_prompts: {raw_items - valid_items}")

    print("\n[Prompt Length Stats]")
    if len(prompt_lengths_sorted) == 0:
        print("  no valid prompts")
    else:
        print(f"  max_prompt_length: {analyzer.max_prompt_length}")
        print(f"  min: {prompt_lengths_sorted[0]}")
        print(f"  p50: {_percentile(prompt_lengths_sorted, 0.50):.2f}")
        print(f"  p90: {_percentile(prompt_lengths_sorted, 0.90):.2f}")
        print(f"  p95: {_percentile(prompt_lengths_sorted, 0.95):.2f}")
        print(f"  p99: {_percentile(prompt_lengths_sorted, 0.99):.2f}")
        print(f"  max: {prompt_lengths_sorted[-1]}")
        print(f"  mean: {mean_len:.2f}")
        print(f"  std: {std_len:.2f}")
        print(f"  truncated_at_max: {truncated_count} ({100.0 * _safe_ratio(truncated_count, valid_items):.2f}%)")

    print("\n[Prompt Duplicate Stats]")
    print(f"  duplicate_prompts: {duplicate_count}")
    print(f"  duplicate_ratio: {100.0 * _safe_ratio(duplicate_count, valid_items):.2f}%")

    print("\n[Ground Truth Coverage]")
    print(f"  with_ground_truth_answer: {with_ground_truth} ({100.0 * _safe_ratio(with_ground_truth, raw_items):.2f}%)")
    print(
        f"  extractable_ground_truth: {extractable_ground_truth} "
        f"({100.0 * _safe_ratio(extractable_ground_truth, with_ground_truth):.2f}% of items with ground truth)"
    )
    print(
        f"  empty_after_extract: {empty_ground_truth_after_extract} "
        f"({100.0 * _safe_ratio(empty_ground_truth_after_extract, with_ground_truth):.2f}% of items with ground truth)"
    )
    print(
        f"  extraction_changed_vs_raw: {extraction_changed} "
        f"({100.0 * _safe_ratio(extraction_changed, with_ground_truth):.2f}% of items with ground truth)"
    )
    print(
        f"  unknown_prompt_source_but_valid: {unknown_source_valid} "
        f"({100.0 * _safe_ratio(unknown_source_valid, valid_items):.2f}% of valid prompts)"
    )

    _print_counter("Drop Reasons", drop_reasons, sum(drop_reasons.values()), args.top_k)
    _print_counter("Prompt Source Distribution", prompt_source_counter, raw_items, args.top_k)
    _print_counter("Ground Truth Field Distribution", metadata_field_counter, with_ground_truth, args.top_k)
    _print_counter("Ground Truth Answer Type Distribution (Raw)", answer_type_counter, with_ground_truth, args.top_k)
    _print_counter(
        "Ground Truth Answer Type Distribution (Extracted)",
        extracted_answer_type_counter,
        with_ground_truth,
        args.top_k,
    )

    _print_examples("Risk Samples: Unknown Prompt Source", risk_samples["unknown_source"])
    _print_examples("Risk Samples: Empty Extracted Answer", risk_samples["empty_extract"])
    _print_examples("Risk Samples: Long Raw Answers", risk_samples["long_raw_answer"])
    _print_examples("Risk Samples: Changed After Extraction", risk_samples["changed_extract"])


if __name__ == "__main__":
    main()
