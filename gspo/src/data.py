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

        # Determine column names.
        cols = list(df.columns)
        prompt_col = next((c for c in cols if c.lower() in ("prompt", "question", "instruction", "messages")), None)
        truth_col = next((c for c in cols if c.lower() in ("solution", "answer", "ground_truth", "target")), None)

        if prompt_col is None:
            raise ValueError(f"Could not find prompt column in parquet. Columns: {cols}")
        if truth_col is None:
            raise ValueError(f"Could not find ground-truth column in parquet. Columns: {cols}")

        logger.info("Parquet columns: prompt='%s', ground_truth='%s'", prompt_col, truth_col)

        samples: List[Dict[str, Any]] = []
        for idx, row in df.iterrows():
            # Extract raw values.
            raw_prompt = row[prompt_col]
            raw_truth = row[truth_col]

            # Normalise prompt to plain list-of-dicts.
            msgs = self._normalise_messages(raw_prompt)
            if msgs is None:
                logger.warning("Row %d: could not parse prompt, skipped", idx)
                continue

            # Build prompt text via chat template.
            try:
                prompt_text = self.tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True,
                )
            except Exception:
                # Fallback: concat user content.
                parts = [m.get("content", "") for m in msgs if m.get("role") == "user"]
                prompt_text = "\n".join(parts) if parts else None

            if not prompt_text:
                continue

            # Normalise ground truth.
            ground_truth = self._normalize(raw_truth)
            if ground_truth is None:
                continue

            samples.append({"prompt": prompt_text, "ground_truth": str(ground_truth)})
        return samples

    @staticmethod
    def _normalise_messages(value: Any) -> Optional[List[Dict[str, str]]]:
        """Convert a prompt value into a clean list of role/content dicts.

        Handles:
        - Plain string
        - list of dicts (may be wrapped in numpy array)
        - single dict
        """
        if value is None:
            return None

        # Plain string → wrap as single user message.
        if isinstance(value, str):
            s = value.strip()
            return [{"role": "user", "content": s}] if s else None

        # numpy array → convert to list recursively.
        if hasattr(value, "tolist"):
            value = value.tolist()

        # Single dict → wrap in list.
        if isinstance(value, dict):
            if "role" in value and "content" in value:
                return [{"role": str(value["role"]), "content": str(value["content"])}]
            return None

        # List of dicts (expected case).
        if isinstance(value, list):
            clean: List[Dict[str, str]] = []
            for item in value:
                if isinstance(item, dict):
                    role = str(item.get("role", "user"))
                    content = str(item.get("content", ""))
                    clean.append({"role": role, "content": content})
                elif isinstance(item, str):
                    clean.append({"role": "user", "content": item})
            return clean if clean else None

        return None

    def _normalize(self, value) -> Optional[str]:
        if isinstance(value, str):
            s = value.strip()
            return s if s else None
        return None

    def _extract(self, item: Dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
        """Extract (prompt_text, ground_truth) from a JSONL dict item.

        For parquet data, this is not used—see ``_load_parquet`` and
        ``_normalise_messages`` instead.
        """
        # Prompt: try string fields, then messages list.
        prompt = (
            self._normalize(item.get("prompt"))
            or self._normalize(item.get("question"))
            or self._normalize(item.get("instruction"))
        )
        if prompt is None and "messages" in item:
            msgs = self._normalise_messages(item["messages"])
            if msgs:
                try:
                    prompt = self.tokenizer.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True,
                    )
                except Exception:
                    parts = [m["content"] for m in msgs if m.get("role") == "user"]
                    prompt = "\n".join(parts) if parts else None

        if not prompt:
            return None, None

        # Ground truth.
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
