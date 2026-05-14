# coding=utf-8
"""Rollout Buffer for GSPO on-policy training.

The RolloutBuffer decouples generation from training:
- Every gen_every_n_steps, generate fresh responses and store them.
- In between, sample from the buffer to train without re-generating.

This dramatically reduces generation overhead (WeDLM block decoding is
much slower than a single forward/backward pass).
"""

import logging
import random
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class RolloutBuffer:
    """Fixed-size FIFO buffer storing generated responses with metadata.

    Each entry corresponds to one prompt and its K generated responses:
        {
            "prompt_text": str,
            "messages": List[Dict],
            "ground_truth": str,
            "responses": [
                {"input_ids": Tensor, "labels": Tensor, "text": str},
                ...  # K entries
            ],
            "rewards": Tensor [K],
            "ref_scores": Tensor [K],
        }

    The buffer supports two access patterns:
    - push: add a new batch of entries (during generation steps)
    - sample: randomly draw one entry (during training-only steps)
    - pop_oldest: FIFO dequeue when buffer is full
    """

    def __init__(self, max_size: int = 16):
        """
        Args:
            max_size: Maximum number of entries in the buffer.
                      Recommended: gen_every_n_steps * per_device_train_batch_size.
        """
        if max_size < 1:
            raise ValueError(f"max_size must be >= 1, got {max_size}")
        self.max_size = max_size
        self._entries: List[Dict[str, Any]] = []
        self._rng = random.Random()

    def __len__(self) -> int:
        return len(self._entries)

    def push_batch(
        self,
        prompt_texts: List[str],
        messages_list: List[List[Dict[str, str]]],
        ground_truths: List[str],
        all_responses: List[List[Dict[str, Any]]],
        rewards: torch.Tensor,      # [bs, K]
        ref_scores: torch.Tensor,   # [bs, K]
    ) -> int:
        """Push a full batch of entries into the buffer.

        Args:
            prompt_texts: List of prompt strings (length bs).
            messages_list: List of message dicts per prompt (length bs).
            ground_truths: List of ground truth strings (length bs).
            all_responses: List of K response dicts per prompt (length bs).
            rewards: Tensor [bs, K] of math rewards.
            ref_scores: Tensor [bs, K] of reference model scores.

        Returns:
            Number of entries evicted due to buffer overflow.
        """
        bs = len(prompt_texts)
        evicted = 0

        for i in range(bs):
            entry = {
                "prompt_text": prompt_texts[i],
                "messages": messages_list[i],
                "ground_truth": ground_truths[i],
                "responses": all_responses[i],
                "rewards": rewards[i].detach().cpu(),
                "ref_scores": ref_scores[i].detach().cpu(),
            }
            evicted += self._push_one(entry)

        logger.debug(
            f"RolloutBuffer: pushed {bs} entries, evicted {evicted}, "
            f"current size {len(self._entries)}/{self.max_size}"
        )
        return evicted

    def _push_one(self, entry: Dict[str, Any]) -> int:
        """Push a single entry, evicting oldest if full. Returns 1 if evicted."""
        if len(self._entries) >= self.max_size:
            self._entries.pop(0)  # FIFO eviction
            self._entries.append(entry)
            return 1
        else:
            self._entries.append(entry)
            return 0

    def sample(self) -> Dict[str, Any]:
        """Randomly sample one entry from the buffer.

        Returns:
            A single buffer entry dict.  If the buffer is empty, raises IndexError.

        Raises:
            IndexError: If the buffer is empty.
        """
        if not self._entries:
            raise IndexError("RolloutBuffer is empty — cannot sample")
        idx = self._rng.randint(0, len(self._entries) - 1)
        return self._entries[idx]

    def sample_batch(self, batch_size: int) -> Dict[str, Any]:
        """Sample a batch (with replacement) and collate into training format.

        Args:
            batch_size: Number of entries to sample.

        Returns:
            A dict suitable for train_step_gspo:
                {
                    "prompt_texts": List[str],
                    "messages": List[List[Dict]],
                    "ground_truths": List[str],
                    "all_responses": List[List[Dict]],
                    "rewards": Tensor [batch_size, K],
                    "ref_scores": Tensor [batch_size, K],
                }

        Raises:
            RuntimeError: If buffer has fewer entries than batch_size.
        """
        if len(self._entries) < batch_size:
            raise RuntimeError(
                f"RolloutBuffer has {len(self._entries)} entries, "
                f"need at least {batch_size} to sample without starvation"
            )

        entries = [self.sample() for _ in range(batch_size)]

        return {
            "prompt_texts": [e["prompt_text"] for e in entries],
            "messages": [e["messages"] for e in entries],
            "ground_truths": [e["ground_truth"] for e in entries],
            "all_responses": [e["responses"] for e in entries],
            "rewards": torch.stack([e["rewards"] for e in entries], dim=0),
            "ref_scores": torch.stack([e["ref_scores"] for e in entries], dim=0),
        }

    def clear(self):
        """Clear all entries."""
        self._entries.clear()

    def is_full(self) -> bool:
        return len(self._entries) >= self.max_size
