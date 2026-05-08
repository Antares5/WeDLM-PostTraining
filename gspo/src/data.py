# coding=utf-8
"""Prompt-only dataset for GSPO training.

Unlike ``WeDLMPairwiseDataset`` (chosen/rejected pairs) or
``WeDLMPackedDataset`` (full sequences), this dataset only loads
prompts and ground-truth answers.  Completions are generated online
by the GSPO generator during training.

Supported JSONL formats (per line):
  1) {"prompt": "question text", "ground_truth": "answer"}
  2) {"messages": [{"role":"user","content":"..."}], "ground_truth": "..."}
  3) {"question": "...", "answer": "..."}
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


def get_im_end_token_id(tokenizer) -> int:
    """Return <|im_end|> token id for a tokenizer, or a safe default."""
    try:
        return tokenizer.convert_tokens_to_ids("<|im_end|>")
    except Exception:
        return 151645


class GSPOPromptDataset(Dataset):
    """Dataset yielding (prompt_text, prompt_ids, ground_truth) tuples.

    This is a lightweight dataset: no tokenization beyond prompt encoding,
    and no pre-packing.  The trainer is responsible for parallel generation
    and dynamic scoring.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int = 2048,
        num_learnable_im_end: int = 0,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.im_end_token_id = get_im_end_token_id(tokenizer)
        self.num_learnable_im_end = num_learnable_im_end
        self.samples = self._load(data_path)
        logger.info("GSPOPromptDataset: %d prompts loaded from %s", len(self.samples), data_path)

    # ── Loading ──────────────────────────────────────────────────────
    def _load(self, path: str) -> List[Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Line %d: invalid JSON, skipped", line_no)
                    continue
                prompt_text, ground_truth = self._extract(item)
                if prompt_text is None or ground_truth is None:
                    logger.warning("Line %d: missing prompt or ground_truth, skipped", line_no)
                    continue
                samples.append({"prompt": prompt_text, "ground_truth": str(ground_truth)})
        return samples

    def _normalize(self, value) -> Optional[str]:
        if isinstance(value, str):
            s = value.strip()
            return s if s else None
        return None

    def _extract(self, item: Dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
        """Extract (prompt_text, ground_truth) from a JSONL item."""
        # Prompt: try several common field names.
        prompt = (
            self._normalize(item.get("prompt"))
            or self._normalize(item.get("question"))
            or self._normalize(item.get("instruction"))
        )
        # Also try messages format.
        if prompt is None and "messages" in item:
            msgs = item["messages"]
            if isinstance(msgs, list):
                # Use chat template to get prompt text.
                try:
                    prompt = self.tokenizer.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True,
                    )
                except Exception:
                    # Fallback: concat user contents.
                    parts = [m.get("content", "") for m in msgs if m.get("role") == "user"]
                    prompt = "\n".join(parts) if parts else None

        if prompt is None:
            return None, None

        # Ground truth: try several field names.
        truth = (
            self._normalize(item.get("ground_truth"))
            or self._normalize(item.get("answer"))
            or self._normalize(item.get("solution"))
            or self._normalize(item.get("target"))
        )
        return prompt, truth

    # ── Dataset API ──────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        prompt_ids = self.tokenizer.encode(sample["prompt"], add_special_tokens=False)
        # Append learnable im_end tokens if configured.
        reserved = max(0, self.num_learnable_im_end - 1)
        if reserved > 0:
            effective_max = self.max_seq_length - reserved
            if len(prompt_ids) > effective_max:
                prompt_ids = prompt_ids[:effective_max]
            prompt_ids = prompt_ids + [self.im_end_token_id] * reserved
        else:
            if len(prompt_ids) > self.max_seq_length:
                prompt_ids = prompt_ids[:self.max_seq_length]
        return {
            "prompt_text": sample["prompt"],
            "prompt_ids": prompt_ids,
            "ground_truth": sample["ground_truth"],
        }


def gspo_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for GSPO dataloader.

    Returns a dict of lists (not tensors), because completion generation
    happens inside the training step, before any tensor packing.
    """
    return {
        "prompt_text": [b["prompt_text"] for b in batch],
        "prompt_ids": [b["prompt_ids"] for b in batch],
        "ground_truth": [b["ground_truth"] for b in batch],
    }
