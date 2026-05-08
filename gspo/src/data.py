# coding=utf-8
"""Prompt-only dataset for GSPO training.

Unlike ``WeDLMPairwiseDataset`` (chosen/rejected pairs) or
``WeDLMPackedDataset`` (full sequences), this dataset only loads
prompts and ground-truth answers.  Completions are generated online
by the GSPO generator during training.

Supported formats:
  JSONL (.jsonl):
    {"prompt": "question text", "ground_truth": "answer"}
    {"messages": [{"role":"user","content":"..."}], "ground_truth": "..."}
    {"question": "...", "answer": "..."}

  Parquet (.parquet):
    Columns: prompt, solution (or answer, ground_truth)
    prompt can be: str, or list of {"role","content"} dicts
"""

from __future__ import annotations

import json
import logging
import os
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
        ext = os.path.splitext(path)[1].lower()
        if ext == ".parquet":
            return self._load_parquet(path)
        else:
            return self._load_jsonl(path)

    def _load_jsonl(self, path: str) -> List[Dict[str, Any]]:
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

    def _load_parquet(self, path: str) -> List[Dict[str, Any]]:
        """Load prompts and ground-truths from a parquet file.

        Expected columns: ``prompt`` (str or list-of-dicts), and
        ``solution`` / ``answer`` / ``ground_truth`` (str).
        """
        try:
            import pyarrow.parquet as pq
            table = pq.read_table(path)
            df = table.to_pandas()
        except ImportError:
            try:
                import pandas as pd
                df = pd.read_parquet(path)
            except ImportError:
                raise ImportError(
                    "Reading .parquet requires pyarrow or pandas. "
                    "Install with: pip install pyarrow"
                )

        samples: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            item = row.to_dict()
            # Parquet may store lists as numpy arrays; convert.
            prompt = item.get("prompt")
            if isinstance(prompt, (list,)) and not isinstance(prompt, str):
                # Could be [{'role':'user','content':'...'}].
                item["prompt"] = list(prompt) if hasattr(prompt, '__iter__') else prompt
            # Normalise column names.
            if "solution" in item and "ground_truth" not in item:
                item["ground_truth"] = item["solution"]
            if "answer" in item and "ground_truth" not in item:
                item["ground_truth"] = item["answer"]

            prompt_text, ground_truth = self._extract(item)
            if prompt_text is None or ground_truth is None:
                continue
            samples.append({"prompt": prompt_text, "ground_truth": str(ground_truth)})
        return samples

    def _normalize(self, value) -> Optional[str]:
        if isinstance(value, str):
            s = value.strip()
            return s if s else None
        return None

    def _extract(self, item: Dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
        """Extract (prompt_text, ground_truth) from a dict item."""
        # Prompt: try several common field names.
        prompt = (
            self._normalize(item.get("prompt"))
            or self._normalize(item.get("question"))
            or self._normalize(item.get("instruction"))
        )
        # If prompt is a list (messages format), convert to text.
        if prompt is None and "prompt" in item and isinstance(item["prompt"], list):
            msgs = item["prompt"]
            # Handle numpy-array-wrapped dicts from parquet.
            if hasattr(msgs, 'tolist'):
                msgs = msgs.tolist()
            if isinstance(msgs, list) and len(msgs) > 0:
                # Ensure each element is a plain dict.
                clean_msgs = []
                for m in msgs:
                    if isinstance(m, dict):
                        clean_msgs.append({"role": m.get("role", "user"),
                                           "content": m.get("content", "")})
                if clean_msgs:
                    try:
                        prompt = self.tokenizer.apply_chat_template(
                            clean_msgs, tokenize=False, add_generation_prompt=True,
                        )
                    except Exception:
                        parts = [m["content"] for m in clean_msgs if m.get("role") == "user"]
                        prompt = "\n".join(parts) if parts else None

        # Also try messages field.
        if prompt is None and "messages" in item:
            msgs = item["messages"]
            if isinstance(msgs, list):
                try:
                    prompt = self.tokenizer.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True,
                    )
                except Exception:
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
