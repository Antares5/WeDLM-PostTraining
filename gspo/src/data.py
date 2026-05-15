# coding=utf-8
"""Data loading for GSPO on-policy training.

Provides:
- GSPOPromptDataset: loads prompt-only data (no labels/assistant responses)
- GSPOCollateFunction: left-pads prompt input_ids for batched generation
- get_im_end_token_id: helper to detect im_end token
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

DEFAULT_IM_END_TOKEN_ID = 151645


def get_im_end_token_id(tokenizer: PreTrainedTokenizer) -> int:
    """Get im_end token id from tokenizer."""
    if hasattr(tokenizer, 'im_end_id'):
        return tokenizer.im_end_id

    try:
        tokens = tokenizer.encode("<|im_end|>", add_special_tokens=False)
        if tokens:
            return tokens[0]
    except Exception:
        pass

    if hasattr(tokenizer, 'added_tokens_encoder'):
        if "<|im_end|>" in tokenizer.added_tokens_encoder:
            return tokenizer.added_tokens_encoder["<|im_end|>"]

    return DEFAULT_IM_END_TOKEN_ID


class GSPOPromptDataset(Dataset):
    """Dataset that loads prompt-only data for GSPO on-policy training.

    Each sample contains:
    - input_ids: tokenized prompt (no assistant response)
    - attention_mask: all ones
    - messages: original chat messages (for text reconstruction)
    - ground_truth: the correct answer (from 'solution' or 'answer' field)

    Supported data formats:
    - JSONL: {"messages": [...], "solution": "42"} or {"messages": [...], "answer": "42"}
    - DeepMath Parquet: columns containing prompt messages and solution/answer columns
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int = 2048,
        prompt_format: str = "messages",
        num_learnable_im_end: int = 0,
    ):
        """
        Args:
            data_path: Path to JSONL or Parquet file.
            tokenizer: HuggingFace tokenizer.
            max_seq_length: Maximum token length for prompt truncation.
            prompt_format: "messages" (standard chat format) or "deepmath" (parquet).
            num_learnable_im_end: Number of learnable im_end tokens to reserve.
        """
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.prompt_format = prompt_format
        self.num_learnable_im_end = num_learnable_im_end
        self.im_end_token_id = get_im_end_token_id(tokenizer)

        self.samples = self._load_data(data_path)
        logger.info(f"Loaded {len(self.samples)} prompt samples from {data_path}")

    def _load_data(self, data_path: str) -> List[Dict[str, Any]]:
        """Load prompt data from file."""
        if data_path.endswith(".jsonl"):
            return self._load_jsonl(data_path)
        elif data_path.endswith(".parquet"):
            return self._load_parquet(data_path)
        else:
            raise ValueError(f"Unsupported data format: {data_path}")

    def _load_jsonl(self, data_path: str) -> List[Dict[str, Any]]:
        """Load prompt data from JSONL file."""
        samples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    parsed = self._parse_item(item)
                    if parsed is not None:
                        samples.append(parsed)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping line {line_num}: JSON error - {e}")
                except Exception as e:
                    logger.warning(f"Skipping line {line_num}: {e}")
        return samples

    def _load_parquet(self, data_path: str) -> List[Dict[str, Any]]:
        """Load prompt data from Parquet file (DeepMath format)."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for Parquet loading. Install with: pip install pandas pyarrow")

        df = pd.read_parquet(data_path)
        samples = []

        for _, row in df.iterrows():
            item = row.to_dict()
            parsed = self._parse_item(item)
            if parsed is not None:
                samples.append(parsed)

        return samples

    def _parse_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse a single data item into standardized format.

        Returns:
            Dict with messages, ground_truth, or None if invalid.
        """
        if not isinstance(item, dict):
            return None

        # Extract messages
        messages = item.get("messages")

        # DeepMath parquet: 'prompt' column is already a list of {role, content} dicts
        if messages is None:
            prompt_val = item.get("prompt")
            if isinstance(prompt_val, list) and len(prompt_val) > 0:
                # Validate that entries look like message dicts
                if all(isinstance(m, dict) and "role" in m and "content" in m for m in prompt_val):
                    messages = prompt_val

        if messages is None:
            # Try flat-column format: build messages from columns
            messages = self._build_messages_from_flat(item)

        if not messages:
            return None

        # Validate messages format
        if not isinstance(messages, list) or len(messages) == 0:
            return None

        # Extract ground truth
        ground_truth = self._extract_ground_truth(item)
        if ground_truth is None:
            logger.warning("No ground truth found in item, skipping")
            return None

        return {
            "messages": messages,
            "ground_truth": ground_truth,
        }

    def _build_messages_from_flat(self, item: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
        """Build messages list from flat columns (DeepMath format)."""
        messages = []

        # System prompt
        system = item.get("system")
        if system and isinstance(system, str) and system.strip():
            messages.append({"role": "system", "content": system.strip()})

        # User prompt
        prompt = item.get("prompt")
        question = item.get("question")
        instruction = item.get("instruction")
        input_text = item.get("input")

        user_content = None
        if question and isinstance(question, str):
            user_content = question.strip()
        elif prompt and isinstance(prompt, str):
            user_content = prompt.strip()
        elif instruction and isinstance(instruction, str):
            parts = [instruction.strip()]
            if input_text and isinstance(input_text, str) and input_text.strip():
                parts.append(input_text.strip())
            user_content = "\n\n".join(parts)

        if user_content:
            messages.append({"role": "user", "content": user_content})

        return messages if messages else None

    def _extract_ground_truth(self, item: Dict[str, Any]) -> Optional[str]:
        """Extract ground truth answer from item.

        Priority: solution > answer > target > label
        """
        for key in ["solution", "answer", "target", "label"]:
            value = item.get(key)
            if value is not None:
                if isinstance(value, (int, float)):
                    return str(value)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return None

    def _tokenize_prompt(self, messages: List[Dict[str, str]]) -> Optional[torch.Tensor]:
        """Tokenize prompt messages into input_ids.

        Uses chat template to format messages, then encodes with truncation.

        Returns:
            input_ids tensor, or None if tokenization fails.
        """
        if not messages:
            return None

        # Apply chat template to get formatted text
        try:
            prompt_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # Fallback: manual formatting
            prompt_text = ""
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                prompt_text += f"{role}: {content}\n"
            prompt_text += "assistant: "

        # Reserve space for im_end tokens
        reserved_for_im_end = max(0, self.num_learnable_im_end - 1)
        effective_max_len = self.max_seq_length - reserved_for_im_end

        # Tokenize with truncation
        prompt_ids = self.tokenizer.encode(
            prompt_text,
            add_special_tokens=False,
            truncation=True,
            max_length=effective_max_len,
        )

        if len(prompt_ids) == 0:
            return None

        # Add learnable im_end tokens if configured
        if reserved_for_im_end > 0:
            prompt_ids = prompt_ids + [self.im_end_token_id] * reserved_for_im_end

        return torch.tensor(prompt_ids, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single prompt sample.

        Returns:
            Dict with:
                - input_ids: torch.Tensor [L_prompt]
                - attention_mask: torch.Tensor [L_prompt] (all 1s)
                - messages: List[Dict] original chat messages
                - ground_truth: str
        """
        sample = self.samples[idx]
        messages = sample["messages"]
        ground_truth = sample["ground_truth"]

        input_ids = self._tokenize_prompt(messages)
        if input_ids is None:
            # Return a minimal valid sample (will be skipped in collate)
            return {
                "input_ids": torch.tensor([self.im_end_token_id], dtype=torch.long),
                "attention_mask": torch.ones(1, dtype=torch.long),
                "messages": messages,
                "ground_truth": ground_truth,
            }

        attention_mask = torch.ones_like(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "messages": messages,
            "ground_truth": ground_truth,
        }

    def decode_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Decode messages back to prompt text for generation."""
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            text = ""
            for msg in messages:
                text += f"{msg.get('role', 'user')}: {msg.get('content', '')}\n"
            text += "assistant: "
            return text


class GSPOCollateFunction:
    """Collate function for GSPO prompt data.

    Left-pads prompt input_ids to the same length for batched processing.
    """

    def __init__(self, pad_token_id: int = DEFAULT_IM_END_TOKEN_ID):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate a batch of prompt samples.

        Args:
            batch: List of sample dicts from GSPOPromptDataset.

        Returns:
            Dict with:
                - input_ids: torch.Tensor [bs, max_len] (left-padded)
                - attention_mask: torch.Tensor [bs, max_len]
                - messages: List[List[Dict]]
                - ground_truths: List[str]
                - prompt_texts: List[str] (simple formatted prompt strings)
        """
        if len(batch) == 0:
            raise ValueError("Empty batch in GSPOCollateFunction")

        messages_list = [item["messages"] for item in batch]
        ground_truths = [item["ground_truth"] for item in batch]

        # Left-pad input_ids
        input_ids_list = [item["input_ids"] for item in batch]
        max_len = max(ids.size(0) for ids in input_ids_list)

        padded_input_ids = []
        padded_attention_mask = []

        for ids in input_ids_list:
            pad_len = max_len - ids.size(0)
            if pad_len > 0:
                padded = torch.cat([
                    torch.full((pad_len,), self.pad_token_id, dtype=ids.dtype),
                    ids,
                ])
                mask = torch.cat([
                    torch.zeros(pad_len, dtype=torch.long),
                    torch.ones(ids.size(0), dtype=torch.long),
                ])
            else:
                padded = ids
                mask = torch.ones(ids.size(0), dtype=torch.long)

            padded_input_ids.append(padded)
            padded_attention_mask.append(mask)

        # Generate simple prompt texts from messages
        prompt_texts = []
        for msgs in messages_list:
            text = ""
            for msg in msgs:
                text += f"{msg.get('role', 'user')}: {msg.get('content', '')}\n"
            text += "assistant: "
            prompt_texts.append(text)

        return {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_mask),
            "messages": messages_list,
            "ground_truths": ground_truths,
            "prompt_texts": prompt_texts,
        }
