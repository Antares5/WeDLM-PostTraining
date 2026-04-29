# coding=utf-8
"""Data processing for WeDLM SFT with efficient packing."""

import json
import os
import pickle
import hashlib
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

DEFAULT_IM_END_TOKEN_ID = 151645
PACKING_ALGO_VERSION = "v2_no_discard"

DEEPMATH_ANSWER_FIELD_CANDIDATES = [
    "final_answer",
    "answer",
    "ground_truth",
    "target",
    "label",
    "solution",
]


@dataclass
class PackedBatch:
    """A pre-packed batch of samples."""
    packed_input_ids: torch.Tensor  # [total_length]
    packed_labels: torch.Tensor     # [total_length]
    cum_seqlens: torch.Tensor       # [num_samples + 1]
    num_samples: int
    total_tokens: int


class WeDLMPackedDataset(Dataset):
    """Dataset that pre-packs samples into fixed-length batches.
    
    Key design:
    - batch_seq_length = max_seq_length * per_device_train_batch_size
    - Each batch is packed to approximately batch_seq_length tokens
    - Samples can span across batches without dropping token remainders
    - All batches are pre-built and cached for fast loading
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int = 2048,
        per_device_train_batch_size: int = 2,
        num_learnable_im_end: int = 8,
        cache_dir: Optional[str] = None,
        seed: int = 42,
        rebuild_cache: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.per_device_train_batch_size = per_device_train_batch_size
        self.batch_seq_length = max_seq_length * per_device_train_batch_size
        self.num_learnable_im_end = num_learnable_im_end
        self.im_end_token_id = get_im_end_token_id(tokenizer)
        self.seed = seed
        
        # Setup cache directory
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(data_path), ".packed_cache")
        self.cache_dir = cache_dir
        
        # Compute cache filename based on config hash
        config_hash = self._compute_config_hash(data_path)
        self.cache_file = os.path.join(cache_dir, f"packed_{config_hash}.pkl")
        self.meta_file = os.path.join(cache_dir, f"meta_{config_hash}.json")
        
        # Load or build packed batches
        if rebuild_cache and os.path.exists(self.cache_file):
            os.remove(self.cache_file)
            if os.path.exists(self.meta_file):
                os.remove(self.meta_file)
        
        self.packed_batches, self.metadata = self._load_or_build_cache(data_path)
        
        logger.info(f"Loaded {len(self.packed_batches)} packed batches")
        logger.info(f"Total samples: {self.metadata['total_samples']}")
        logger.info(f"Total tokens: {self.metadata['total_tokens']}")
        logger.info(f"Batch seq length: {self.batch_seq_length}")
    
    def _compute_config_hash(self, data_path: str) -> str:
        """Compute a hash of the configuration for cache naming."""
        config_dict = {
            "data_path": data_path,
            "max_seq_length": self.max_seq_length,
            "batch_seq_length": self.batch_seq_length,
            "num_learnable_im_end": self.num_learnable_im_end,
            "seed": self.seed,
            "packing_algo_version": PACKING_ALGO_VERSION,
        }
        # Include file modification time for cache invalidation
        if os.path.exists(data_path):
            config_dict["mtime"] = os.path.getmtime(data_path)
        
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:16]
    
    def _load_or_build_cache(self, data_path: str) -> Tuple[List[PackedBatch], Dict]:
        """Load from cache or build packed batches."""
        if os.path.exists(self.cache_file) and os.path.exists(self.meta_file):
            logger.info(f"Loading cached packed batches from {self.cache_file}")
            try:
                with open(self.cache_file, "rb") as f:
                    packed_batches = pickle.load(f)
                with open(self.meta_file, "r") as f:
                    metadata = json.load(f)
                return packed_batches, metadata
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}, rebuilding...")
        
        logger.info("Building packed batches (this may take a while)...")
        packed_batches, metadata = self._build_packed_batches(data_path)
        
        # Save cache
        os.makedirs(self.cache_dir, exist_ok=True)
        with open(self.cache_file, "wb") as f:
            pickle.dump(packed_batches, f)
        with open(self.meta_file, "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved {len(packed_batches)} packed batches to {self.cache_file}")
        return packed_batches, metadata
    
    def _build_packed_batches(self, data_path: str) -> Tuple[List[PackedBatch], Dict]:
        """Build packed batches from raw data."""
        # 1. Load and tokenize all samples
        all_samples = self._load_and_tokenize_data(data_path)
        logger.info(f"Loaded {len(all_samples)} samples")
        
        # 2. Shuffle samples
        import random
        rng = random.Random(self.seed)
        rng.shuffle(all_samples)
        
        # 3. Pack into fixed-length batches
        packed_batches = self._pack_samples_into_batches(all_samples)
        
        # 4. Compute metadata
        total_samples = len(all_samples)
        total_sample_segments = sum(b.num_samples for b in packed_batches)
        total_tokens = sum(b.total_tokens for b in packed_batches)
        
        metadata = {
            "total_samples": total_samples,
            "total_sample_segments": total_sample_segments,
            "total_tokens": total_tokens,
            "num_batches": len(packed_batches),
            "batch_seq_length": self.batch_seq_length,
            "max_seq_length": self.max_seq_length,
            "per_device_train_batch_size": self.per_device_train_batch_size,
        }
        
        return packed_batches, metadata
    
    def _load_and_tokenize_data(self, data_path: str) -> List[Tuple[torch.Tensor, torch.Tensor, int]]:
        """Load and tokenize all samples.
        
        Returns list of (input_ids, labels, original_length) tuples.
        """
        samples = []
        
        with open(data_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    messages = json.loads(line)
                    result = self._tokenize_messages(messages)
                    if result is not None:
                        samples.append(result)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping line {line_num}: JSON decode error - {e}")
                except Exception as e:
                    logger.warning(f"Skipping line {line_num}: {e}")
        
        return samples
    
    def _tokenize_messages(self, messages: List[Dict[str, str]]) -> Optional[Tuple[torch.Tensor, torch.Tensor, int]]:
        """Tokenize chat messages into input_ids and labels.
        
        Returns (input_ids, labels, original_length) or None if invalid.
        """
        if not messages:
            return None
        
        # Apply chat template
        full_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        
        # Reserve space for learnable im_end tokens
        reserved_for_im_end = max(0, self.num_learnable_im_end - 1)
        effective_max_len = self.max_seq_length - reserved_for_im_end
        
        # Tokenize with truncation
        full_ids = self.tokenizer.encode(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=effective_max_len
        )
        
        if len(full_ids) == 0:
            return None
        
        # Compute prompt length for label masking
        if len(messages) > 1:
            prompt_messages = messages[:-1]
            prompt_text = self.tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
            prompt_len = min(len(prompt_ids), len(full_ids))
        else:
            # Single message - mask everything as prompt (no loss)
            prompt_len = len(full_ids)
        
        # Add learnable im_end tokens
        if reserved_for_im_end > 0:
            full_ids = full_ids + [self.im_end_token_id] * reserved_for_im_end
        
        input_ids = torch.tensor(full_ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[:prompt_len] = -100
        
        original_length = len(full_ids)
        
        return (input_ids, labels, original_length)
    
    def _pack_samples_into_batches(
        self, 
        samples: List[Tuple[torch.Tensor, torch.Tensor, int]]
    ) -> List[PackedBatch]:
        """Pack samples into fixed-length batches.
        
        Strategy:
        - Fill each batch up to batch_seq_length tokens
        - If a sample does not fit, continue packing its remainder in the next batch
        - Never drop sample token remainder during packing
        """
        packed_batches = []
        
        current_input_ids: List[torch.Tensor] = []
        current_labels: List[torch.Tensor] = []
        current_seqlens = [0]
        current_length = 0

        def _flush_current_batch() -> None:
            nonlocal current_input_ids, current_labels, current_seqlens, current_length

            if current_length == 0:
                return

            batch = self._create_packed_batch(current_input_ids, current_labels, current_seqlens)
            packed_batches.append(batch)

            current_input_ids = []
            current_labels = []
            current_seqlens = [0]
            current_length = 0
        
        for input_ids, labels, _ in samples:
            sample_len = input_ids.size(0)
            sample_start = 0

            while sample_start < sample_len:
                space_left = self.batch_seq_length - current_length
                if space_left == 0:
                    _flush_current_batch()
                    space_left = self.batch_seq_length

                take_len = min(space_left, sample_len - sample_start)
                sample_end = sample_start + take_len

                current_input_ids.append(input_ids[sample_start:sample_end])
                current_labels.append(labels[sample_start:sample_end])
                current_length += take_len
                current_seqlens.append(current_length)

                sample_start = sample_end

                if current_length == self.batch_seq_length:
                    _flush_current_batch()
        
        # Handle last batch (may not be full)
        _flush_current_batch()
        
        return packed_batches
    
    def _create_packed_batch(
        self,
        input_ids_list: List[torch.Tensor],
        labels_list: List[torch.Tensor],
        seqlens: List[int],
    ) -> PackedBatch:
        """Create a PackedBatch from lists of tensors."""
        packed_input_ids = torch.cat(input_ids_list, dim=0)
        packed_labels = torch.cat(labels_list, dim=0)
        cum_seqlens = torch.tensor(seqlens, dtype=torch.long)
        
        return PackedBatch(
            packed_input_ids=packed_input_ids,
            packed_labels=packed_labels,
            cum_seqlens=cum_seqlens,
            num_samples=len(input_ids_list),
            total_tokens=packed_input_ids.size(0),
        )
    
    def __len__(self) -> int:
        """Return number of batches (= number of training steps per epoch)."""
        return len(self.packed_batches)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a pre-packed batch."""
        batch = self.packed_batches[idx]
        return {
            "packed_input_ids": batch.packed_input_ids.clone(),
            "packed_labels": batch.packed_labels.clone(),
            "cum_seqlens": batch.cum_seqlens.clone(),
        }
    
    def get_total_samples(self) -> int:
        """Get total number of samples across all batches."""
        return self.metadata["total_samples"]
    
    def get_total_tokens(self) -> int:
        """Get total number of tokens across all batches."""
        return self.metadata["total_tokens"]
    
    def get_num_training_steps(self, num_epochs: int = 1) -> int:
        """Get total number of training steps for given epochs."""
        return len(self.packed_batches) * num_epochs


class WeDLMShuffledPackedDataset(Dataset):
    """A wrapper that shuffles batch order each epoch.
    
    Use this for multi-epoch training where you want different
    batch ordering each epoch.
    """
    
    def __init__(
        self,
        base_dataset: WeDLMPackedDataset,
        epoch: int = 0,
        seed: int = 42,
    ):
        self.base_dataset = base_dataset
        self.epoch = epoch
        self.seed = seed
        self._shuffle_indices()
    
    def _shuffle_indices(self):
        """Shuffle indices based on epoch and seed."""
        import random
        rng = random.Random(self.seed + self.epoch)
        self.indices = list(range(len(self.base_dataset)))
        rng.shuffle(self.indices)
    
    def set_epoch(self, epoch: int):
        """Set epoch for reshuffling."""
        self.epoch = epoch
        self._shuffle_indices()
    
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.base_dataset[self.indices[idx]]


class WeDLMPairwiseDataset(Dataset):
    """Pairwise dataset for DPO scaffolding.

    Supported JSONL formats per line:
    1) {"prompt": [...], "chosen": [...], "rejected": [...]}  # prompt + assistant branches
    2) {"chosen": [...], "rejected": [...]}                    # full conversations
    3) {"prompt": "...", "chosen": "...", "rejected": "..."}
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int = 2048,
        num_learnable_im_end: int = 8,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.num_learnable_im_end = num_learnable_im_end
        self.im_end_token_id = get_im_end_token_id(tokenizer)
        self.samples = self._load_and_tokenize_pairs(data_path)
        logger.info(f"Loaded {len(self.samples)} pairwise samples from {data_path}")

    def _to_messages(self, value: Any, default_role: str) -> Optional[List[Dict[str, str]]]:
        if isinstance(value, str):
            return [{"role": default_role, "content": value}]

        if isinstance(value, list):
            if all(isinstance(item, dict) and "role" in item and "content" in item for item in value):
                return value

        return None

    def _normalize_text(self, value: Any) -> Optional[str]:
        if isinstance(value, str):
            normalized = value.strip()
            return normalized if len(normalized) > 0 else None

        return None

    def _build_prompt_messages_from_item(self, item: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
        r"""Build prompt messages from common pairwise dataset fields.

        Priority for user-side prompt text:
        question > prompt(string) > instruction(+input) > input.
        """
        prompt_messages: List[Dict[str, str]] = []

        system_text = self._normalize_text(item.get("system"))
        if system_text is not None:
            prompt_messages.append({"role": "system", "content": system_text})

        user_text = self._normalize_text(item.get("question"))
        if user_text is None:
            user_text = self._normalize_text(item.get("prompt"))

        if user_text is None:
            instruction_text = self._normalize_text(item.get("instruction"))
            input_text = self._normalize_text(item.get("input"))
            if instruction_text is not None and input_text is not None:
                user_text = f"{instruction_text}\n\n{input_text}"
            elif instruction_text is not None:
                user_text = instruction_text
            elif input_text is not None:
                user_text = input_text

        if user_text is not None:
            prompt_messages.append({"role": "user", "content": user_text})

        return prompt_messages if len(prompt_messages) > 0 else None

    def _merge_prompt_and_answer(
        self,
        prompt_messages: Optional[List[Dict[str, str]]],
        answer_messages: Optional[List[Dict[str, str]]],
    ) -> Optional[List[Dict[str, str]]]:
        if answer_messages is None:
            return None

        if prompt_messages is None or len(prompt_messages) == 0:
            return answer_messages

        # If answer already starts from user/system, treat it as a full conversation.
        if len(answer_messages) > 0 and answer_messages[0].get("role") in {"user", "system"}:
            return answer_messages

        return prompt_messages + answer_messages

    def _tokenize_messages(self, messages: List[Dict[str, str]]) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        full_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        reserved_for_im_end = max(0, self.num_learnable_im_end - 1)
        effective_max_len = self.max_seq_length - reserved_for_im_end

        full_ids = self.tokenizer.encode(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=effective_max_len,
        )
        if len(full_ids) == 0:
            return None

        if len(messages) > 1:
            prompt_messages = messages[:-1]
            prompt_text = self.tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
            prompt_len = min(len(prompt_ids), len(full_ids))
        else:
            prompt_len = len(full_ids)

        if reserved_for_im_end > 0:
            full_ids = full_ids + [self.im_end_token_id] * reserved_for_im_end

        input_ids = torch.tensor(full_ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[:prompt_len] = -100
        return input_ids, labels

    def _parse_pair_item(
        self,
        item: Dict[str, Any],
    ) -> Optional[Tuple[List[Dict[str, str]], List[Dict[str, str]]]]:
        if not isinstance(item, dict):
            return None

        chosen_messages = self._to_messages(item.get("chosen"), "assistant")
        rejected_messages = self._to_messages(item.get("rejected"), "assistant")
        if chosen_messages is None or rejected_messages is None:
            return None

        prompt_messages = self._to_messages(item.get("prompt"), "user") if "prompt" in item else None
        if prompt_messages is None:
            prompt_messages = self._build_prompt_messages_from_item(item)

        chosen_messages = self._merge_prompt_and_answer(prompt_messages, chosen_messages)
        rejected_messages = self._merge_prompt_and_answer(prompt_messages, rejected_messages)
        if chosen_messages is None or rejected_messages is None:
            return None

        return chosen_messages, rejected_messages

    def _load_and_tokenize_pairs(self, data_path: str) -> List[Dict[str, torch.Tensor]]:
        samples: List[Dict[str, torch.Tensor]] = []

        with open(data_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                try:
                    item = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping line {line_num}: JSON decode error - {e}")
                    continue

                parsed_pair = self._parse_pair_item(item)
                if parsed_pair is None:
                    logger.warning(f"Skipping line {line_num}: invalid pairwise schema")
                    continue

                chosen_conv, rejected_conv = parsed_pair
                chosen_pair = self._tokenize_messages(chosen_conv)
                rejected_pair = self._tokenize_messages(rejected_conv)
                if chosen_pair is None or rejected_pair is None:
                    logger.warning(f"Skipping line {line_num}: empty chosen/rejected after tokenization")
                    continue

                chosen_input_ids, chosen_labels = chosen_pair
                rejected_input_ids, rejected_labels = rejected_pair
                chosen_valid = int((chosen_labels != -100).sum().item())
                rejected_valid = int((rejected_labels != -100).sum().item())
                if chosen_valid == 0 or rejected_valid == 0:
                    logger.warning(
                        "Skipping line %s: zero valid label tokens (chosen=%s, rejected=%s)",
                        line_num,
                        chosen_valid,
                        rejected_valid,
                    )
                    continue

                samples.append(
                    {
                        "chosen_input_ids": chosen_input_ids,
                        "chosen_labels": chosen_labels,
                        "rejected_input_ids": rejected_input_ids,
                        "rejected_labels": rejected_labels,
                    }
                )

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        return {
            "chosen_input_ids": sample["chosen_input_ids"].clone(),
            "chosen_labels": sample["chosen_labels"].clone(),
            "rejected_input_ids": sample["rejected_input_ids"].clone(),
            "rejected_labels": sample["rejected_labels"].clone(),
        }


class WeDLMPromptDataset(Dataset):
    """Prompt-only dataset for online GSPO rollouts.

    Supported input formats:
    - JSONL: one JSON item per line
    - Parquet: one sample per row

    Supported JSONL item formats per line:
    1) [{"role": ..., "content": ...}, ...]              # chat conversation
    2) {"messages": [...]}                                 # wrapped conversation
    3) {"prompt": ..., "chosen": ..., "rejected": ...}  # pairwise-style item
    4) {"question"|"instruction"|"input"|"system": ...}
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_prompt_length: int = 1536,
    ):
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.samples = self._load_and_tokenize_prompts(data_path)
        logger.info(f"Loaded {len(self.samples)} prompt samples from {data_path}")

    def _to_messages(self, value: Any, default_role: str) -> Optional[List[Dict[str, str]]]:
        if isinstance(value, str):
            return [{"role": default_role, "content": value}]

        if isinstance(value, list):
            if all(isinstance(item, dict) and "role" in item and "content" in item for item in value):
                return value

        return None

    def _normalize_text(self, value: Any) -> Optional[str]:
        if isinstance(value, str):
            normalized = value.strip()
            return normalized if len(normalized) > 0 else None

        return None

    def _build_prompt_messages_from_item(self, item: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
        prompt_messages: List[Dict[str, str]] = []

        system_text = self._normalize_text(item.get("system"))
        if system_text is not None:
            prompt_messages.append({"role": "system", "content": system_text})

        user_text = self._normalize_text(item.get("question"))
        if user_text is None:
            user_text = self._normalize_text(item.get("problem"))
        if user_text is None:
            user_text = self._normalize_text(item.get("prompt"))

        if user_text is None:
            instruction_text = self._normalize_text(item.get("instruction"))
            input_text = self._normalize_text(item.get("input"))
            if instruction_text is not None and input_text is not None:
                user_text = f"{instruction_text}\n\n{input_text}"
            elif instruction_text is not None:
                user_text = instruction_text
            elif input_text is not None:
                user_text = input_text

        if user_text is not None:
            prompt_messages.append({"role": "user", "content": user_text})

        return prompt_messages if len(prompt_messages) > 0 else None

    def _extract_prompt_metadata(self, item: Any) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {}
        if not isinstance(item, dict):
            return metadata

        for field in DEEPMATH_ANSWER_FIELD_CANDIDATES:
            value = self._normalize_text(item.get(field))
            if value is not None:
                metadata["ground_truth_answer"] = value
                metadata["ground_truth_field"] = field
                break

        for field in ["dataset", "source", "subject", "type", "level"]:
            value = item.get(field)
            if isinstance(value, str) and value.strip():
                metadata[field] = value.strip()

        return metadata

    def _extract_prompt_messages(self, item: Any) -> Optional[List[Dict[str, str]]]:
        # Pure chat-format conversation lines.
        if isinstance(item, list):
            messages = self._to_messages(item, "user")
            if messages is None:
                return None
            if len(messages) > 1:
                return messages[:-1]
            return messages

        if not isinstance(item, dict):
            return None

        # Wrapped conversation under "messages".
        if "messages" in item:
            messages = self._to_messages(item.get("messages"), "user")
            if messages is not None:
                if len(messages) > 1:
                    return messages[:-1]
                return messages

        # Direct prompt fields.
        prompt_messages = self._to_messages(item.get("prompt"), "user") if "prompt" in item else None
        if prompt_messages is None:
            prompt_messages = self._build_prompt_messages_from_item(item)
        if prompt_messages is not None and len(prompt_messages) > 0:
            return prompt_messages

        # Fallback: pairwise-style "chosen" full conversation.
        chosen_messages = self._to_messages(item.get("chosen"), "assistant")
        if chosen_messages is not None and len(chosen_messages) > 0:
            if chosen_messages[0].get("role") in {"user", "system"}:
                if len(chosen_messages) > 1:
                    return chosen_messages[:-1]
                return chosen_messages

        return None

    def _tokenize_prompt_messages(self, messages: List[Dict[str, str]]) -> Optional[torch.Tensor]:
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = self.tokenizer.encode(
            prompt_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_prompt_length,
        )
        if len(prompt_ids) == 0:
            return None
        return torch.tensor(prompt_ids, dtype=torch.long)

    def _iter_data_items(self, data_path: str):
        suffix = os.path.splitext(data_path)[1].lower()
        if suffix == ".parquet":
            yield from self._iter_parquet_items(data_path)
            return

        with open(data_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                try:
                    item = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping line {line_num}: JSON decode error - {e}")
                    continue

                yield line_num, item

    def _iter_parquet_items(self, data_path: str):
        row_idx = 0

        try:
            import pyarrow.parquet as pq

            parquet_file = pq.ParquetFile(data_path)
            for batch in parquet_file.iter_batches():
                for item in batch.to_pylist():
                    yield row_idx, item
                    row_idx += 1
            return
        except ImportError:
            logger.warning(
                "pyarrow is not installed. Falling back to pandas for parquet reading."
            )

        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "Reading parquet data requires pyarrow or pandas. "
                "Please install one of them, e.g. `pip install pyarrow`."
            ) from e

        dataframe = pd.read_parquet(data_path)
        for item in dataframe.to_dict(orient="records"):
            yield row_idx, item
            row_idx += 1

    def _load_and_tokenize_prompts(self, data_path: str) -> List[Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []

        for item_idx, item in self._iter_data_items(data_path):
            prompt_messages = self._extract_prompt_messages(item)
            if prompt_messages is None or len(prompt_messages) == 0:
                logger.warning(f"Skipping item {item_idx}: unable to extract prompt messages")
                continue

            prompt_ids = self._tokenize_prompt_messages(prompt_messages)
            if prompt_ids is None:
                logger.warning(f"Skipping item {item_idx}: empty prompt after tokenization")
                continue

            samples.append(
                {
                    "prompt_input_ids": prompt_ids,
                    "prompt_metadata": self._extract_prompt_metadata(item),
                }
            )

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        return {
            "prompt_input_ids": sample["prompt_input_ids"].clone(),
            "prompt_metadata": dict(sample["prompt_metadata"]),
        }


def packed_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function for packed dataset.
    
    Since each item is already a complete packed batch, we just return it directly.
    DataLoader should use batch_size=1 with this dataset.
    """
    if len(batch) == 1:
        return batch[0]
    
    # If somehow batch_size > 1, we need to concatenate
    # This shouldn't happen in normal usage
    all_input_ids = []
    all_labels = []
    all_seqlens = [0]
    
    for item in batch:
        all_input_ids.append(item["packed_input_ids"])
        all_labels.append(item["packed_labels"])
        
        # Adjust cum_seqlens offsets
        offset = all_seqlens[-1]
        item_seqlens = item["cum_seqlens"][1:] + offset
        all_seqlens.extend(item_seqlens.tolist())
    
    return {
        "packed_input_ids": torch.cat(all_input_ids, dim=0),
        "packed_labels": torch.cat(all_labels, dim=0),
        "cum_seqlens": torch.tensor(all_seqlens, dtype=torch.long),
    }


def dpo_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function for pairwise DPO data.

    It packs chosen and rejected streams separately and emits independent cumulative
    sequence offsets so downstream logic can build WeDLM batches for each stream.
    """
    if len(batch) == 0:
        raise ValueError("Empty batch in dpo_collate_fn")

    def _pack_stream(items: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        all_items: List[torch.Tensor] = []
        cum_seqlens = [0]
        for tensor in items:
            all_items.append(tensor)
            cum_seqlens.append(cum_seqlens[-1] + tensor.numel())

        return torch.cat(all_items, dim=0), torch.tensor(cum_seqlens, dtype=torch.long)

    chosen_input_ids = [item["chosen_input_ids"] for item in batch]
    chosen_labels = [item["chosen_labels"] for item in batch]
    rejected_input_ids = [item["rejected_input_ids"] for item in batch]
    rejected_labels = [item["rejected_labels"] for item in batch]

    chosen_packed_input_ids, chosen_cum_seqlens = _pack_stream(chosen_input_ids)
    chosen_packed_labels, _ = _pack_stream(chosen_labels)
    rejected_packed_input_ids, rejected_cum_seqlens = _pack_stream(rejected_input_ids)
    rejected_packed_labels, _ = _pack_stream(rejected_labels)

    return {
        "chosen_packed_input_ids": chosen_packed_input_ids,
        "chosen_packed_labels": chosen_packed_labels,
        "chosen_cum_seqlens": chosen_cum_seqlens,
        "rejected_packed_input_ids": rejected_packed_input_ids,
        "rejected_packed_labels": rejected_packed_labels,
        "rejected_cum_seqlens": rejected_cum_seqlens,
        "pair_size": torch.tensor(len(batch), dtype=torch.long),
    }


def gspo_prompt_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for online GSPO prompt batches."""
    if len(batch) == 0:
        raise ValueError("Empty batch in gspo_prompt_collate_fn")

    return {
        "prompt_input_ids": [item["prompt_input_ids"] for item in batch],
        "prompt_metadata": [item.get("prompt_metadata", {}) for item in batch],
        "prompt_count": torch.tensor(len(batch), dtype=torch.long),
    }


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


# Legacy compatibility
@dataclass
class SFTSample:
    """A single SFT sample (for backward compatibility)."""
    input_ids: torch.Tensor
    labels: torch.Tensor


def collate_fn(
    batch: List[SFTSample],
    pad_token_id: int = DEFAULT_IM_END_TOKEN_ID,
) -> Dict[str, torch.Tensor]:
    """Legacy collate function (for backward compatibility)."""
    input_ids_list = []
    labels_list = []
    seq_lens = [0]
    
    for sample in batch:
        input_ids_list.append(sample.input_ids)
        labels_list.append(sample.labels)
        seq_lens.append(seq_lens[-1] + sample.input_ids.size(0))
    
    return {
        "packed_input_ids": torch.cat(input_ids_list, dim=0),
        "packed_labels": torch.cat(labels_list, dim=0),
        "cum_seqlens": torch.tensor(seq_lens, dtype=torch.long),
    }

