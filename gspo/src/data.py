# coding=utf-8
"""Data loading for GSPO — prompt-only JSONL datasets.

GSPO requires prompts (no pre-existing responses), because responses
are generated on-policy during training. This module provides a minimal
dataset that loads prompts from chat-format JSONL files.
"""

import json
import logging
from typing import List, Optional

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


class GSPOPromptDataset(Dataset):
    """Loads prompts from a JSONL file, tokenizes them, returns prompt token ids.

    Each line is a chat message list:
        [{"role": "user", "content": "..."}, ...]

    The entire prompt is tokenized with the chat template.  No completion
    is expected in the data — the model generates completions on-policy.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_prompt_length: int = 2048,
    ):
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.prompts: List[torch.Tensor] = []
        self.prompt_texts: List[str] = []

        self._load(data_path)

        logger.info(f"GSPOPromptDataset: loaded {len(self)} prompts from {data_path}")

    def _load(self, data_path: str):
        with open(data_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    messages = json.loads(line)
                    result = self._tokenize_prompt(messages)
                    if result is not None:
                        self.prompts.append(result)
                        # Also keep raw text for debugging / reward model
                        self.prompt_texts.append(
                            self.tokenizer.apply_chat_template(
                                messages, tokenize=False, add_generation_prompt=True
                            )
                        )
                except json.JSONDecodeError:
                    logger.warning(f"GSPOPromptDataset: skipping line {line_num} (invalid JSON)")
                except Exception as e:
                    logger.warning(f"GSPOPromptDataset: skipping line {line_num}: {e}")

    def _tokenize_prompt(self, messages) -> Optional[torch.Tensor]:
        """Tokenize a chat prompt and return input_ids."""
        if not messages:
            return None
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        ids = self.tokenizer.encode(
            text, add_special_tokens=False, truncation=True,
            max_length=self.max_prompt_length,
        )
        if len(ids) == 0:
            return None
        return torch.tensor(ids, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.prompts[idx]

    def get_prompt_texts(self) -> List[str]:
        """Return the raw prompt text strings (useful for reward model API)."""
        return self.prompt_texts


def gspo_collate_fn(batch: List[torch.Tensor]) -> List[List[int]]:
    """Collate function for GSPO prompts — returns list of token-id lists.

    Unlike packed SFT collation, GSPO does not pack prompts into a single tensor
    because each prompt will be expanded G times by the generator.
    We simply return a list of token-id lists.
    """
    return [t.tolist() for t in batch]
