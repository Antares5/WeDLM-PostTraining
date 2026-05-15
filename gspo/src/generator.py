# coding=utf-8
"""On-policy generation engine using stochastic WeDLM block decoding.

Uses a manual decode loop so K rollouts can diverge under different seeds.
This keeps GSPO sampling stochastic even when the model exposes a greedy
`generate_wedlm()` helper.
"""

import logging
from typing import List, Dict, Optional, Any

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

MASK_TOKEN_ID = 151665
EOS_TOKEN_ID = 151645  # <|im_end|> for WeDLM tokenizer


class WeDLMGenerator:
    """WeDLM block-decoding generator for on-policy response generation.

    The generator uses a manual block-decoding loop with stochastic token
    sampling so repeated calls with different seeds can produce diverse K
    responses for GSPO.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        generation_config: Dict[str, Any],
        device: torch.device,
        model_path: str = None,
    ):
        """
        Args:
            model: Unwrapped policy model (from accelerator.unwrap_model).
            tokenizer: Tokenizer for encoding/decoding.
            generation_config: Dict with max_new_tokens, temperature, top_p, top_k,
                               wedlm_entropy_threshold, wedlm_pos_penalty_factor.
            device: Target device.
            model_path: Path to model directory (unused, kept for compatibility).
        """
        self.model = model
        self.tokenizer = tokenizer
        self.gen_config = generation_config
        self.device = device
        self._model_path = model_path

        # Resolve mask_token_id and eos_token_id
        self.mask_token_id = MASK_TOKEN_ID
        self.eos_token_id = getattr(
            self.tokenizer, "eos_token_id", EOS_TOKEN_ID
        )
        if self.eos_token_id is None:
            self.eos_token_id = EOS_TOKEN_ID

        # Block size for WeDLM decoding
        self.block_size = 32

    @torch.no_grad()
    def generate(
        self, prompt_text: str, seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """Generate a single response using WeDLM block decoding.

        Args:
            prompt_text: The prompt text (already formatted with chat template).
            seed: Random seed for reproducibility.

        Returns:
            Dict with:
                - "input_ids": torch.Tensor [L_prompt + L_response]
                - "labels": torch.Tensor (prompt part = -100)
                - "text": str (the generated response text only)
        """
        if seed is not None:
            torch.manual_seed(seed)

        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        if not prompt_ids:
            raise ValueError("Empty prompt after tokenization")

        max_new_tokens = self.gen_config.get("max_new_tokens", 512)
        temperature = self.gen_config.get("temperature", 1.0)
        confidence_threshold = self.gen_config.get(
            "wedlm_entropy_threshold", 0.4
        )
        top_p = float(self.gen_config.get("top_p", 1.0))
        top_k = int(self.gen_config.get("top_k", 0))

        # Use the manual loop so sampling stays stochastic even if the model
        # exposes a deterministic generate_wedlm() helper.
        response_ids = self._generate_via_loop(
            prompt_ids,
            max_new_tokens,
            temperature,
            confidence_threshold,
            top_p,
            top_k,
        )

        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        return {
            "input_ids": torch.cat([
                torch.tensor(prompt_ids, dtype=torch.long),
                torch.tensor(response_ids, dtype=torch.long),
            ]),
            "labels": torch.cat([
                torch.full((len(prompt_ids),), -100, dtype=torch.long),
                torch.tensor(response_ids, dtype=torch.long),
            ]),
            "text": response_text,
        }

    def _generate_via_builtin(
        self,
        prompt_ids: List[int],
        max_new_tokens: int,
        temperature: float,
        confidence_threshold: float,
    ) -> List[int]:
        """Use model.generate_wedlm() if available."""
        prompt_tensor = torch.tensor(
            prompt_ids, dtype=torch.long, device=self.device
        ).unsqueeze(0)

        result = self.model.generate_wedlm(
            input_ids=prompt_tensor,
            max_new_tokens=max_new_tokens,
            block_size=self.block_size,
            mask_token_id=self.mask_token_id,
            confidence_threshold=confidence_threshold,
            temperature=temperature,
            return_stats=False,
        )

        if isinstance(result, dict):
            result = result.get("sequences", result.get("generated_ids", result))

        if isinstance(result, torch.Tensor):
            full_ids = result[0].tolist()
            # Strip prompt
            response_ids = full_ids[len(prompt_ids):]
            # Remove trailing pad tokens and EOS
            pad_id = getattr(self.tokenizer, "pad_token_id", None)
            eos_id = self.eos_token_id
            while response_ids:
                last = response_ids[-1]
                if last == pad_id or last == eos_id:
                    response_ids.pop()
                else:
                    break
            return response_ids

        return []

    def _sample_token_ids(
        self,
        logits: torch.Tensor,
        greedy_ids: torch.Tensor,
        top_k: int,
        top_p: float,
    ) -> torch.Tensor:
        """Sample one token per masked position from a filtered distribution."""
        filtered_logits = logits.float().clone()
        vocab_size = filtered_logits.size(-1)

        if top_k > 0 and top_k < vocab_size:
            top_k = min(top_k, vocab_size)
            topk_values = torch.topk(filtered_logits, top_k, dim=-1).values[..., -1, None]
            filtered_logits = filtered_logits.masked_fill(filtered_logits < topk_values, float("-inf"))

        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(filtered_logits, descending=True, dim=-1)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = sorted_probs.cumsum(dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False

            remove_mask = torch.zeros_like(filtered_logits, dtype=torch.bool)
            remove_mask.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
            filtered_logits = filtered_logits.masked_fill(remove_mask, float("-inf"))

        probs = F.softmax(filtered_logits, dim=-1)

        if not torch.isfinite(probs).all() or (probs.sum(dim=-1) <= 0).any():
            return greedy_ids

        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def _generate_via_loop(
        self,
        prompt_ids: List[int],
        max_new_tokens: int,
        temperature: float,
        confidence_threshold: float,
        top_p: float,
        top_k: int,
    ) -> List[int]:
        """Manual WeDLM block-decoding loop using model.forward().

        Implements the block-wise mask-predict decoding:
        1. Append a block of MASK tokens
        2. Reorder (unmasked first, masked last) for causal mask
        3. Forward pass → predict and fill confident positions
        4. Repeat until max_new_tokens or EOS
        """
        device = self.device
        block_size = self.block_size
        eos_id = self.eos_token_id

        current_ids = torch.tensor(prompt_ids, dtype=torch.long, device=device)
        prefix_len = len(prompt_ids)
        # Track ORIGINAL position for each token (needed for RoPE)
        orig_positions = torch.arange(prefix_len, dtype=torch.long, device=device)
        num_blocks = (max_new_tokens + block_size - 1) // block_size
        next_pos = prefix_len

        for block_idx in range(num_blocks):
            remaining = max_new_tokens - block_idx * block_size
            cur_block_size = min(block_size, remaining)

            # 1. Append MASK tokens with their future positions
            mask_tensor = torch.full(
                (cur_block_size,), self.mask_token_id,
                dtype=torch.long, device=device
            )
            mask_positions = torch.arange(
                next_pos, next_pos + cur_block_size,
                dtype=torch.long, device=device
            )
            current_ids = torch.cat([current_ids, mask_tensor])
            orig_positions = torch.cat([orig_positions, mask_positions])
            next_pos += cur_block_size

            # 2. WeDLM iteration within this block
            is_mask = (current_ids == self.mask_token_id)

            for _ in range(cur_block_size):
                if not is_mask.any():
                    break

                # Reorder: unmasked first, masked last (both tokens and positions)
                reordered_ids = torch.cat([
                    current_ids[~is_mask],
                    current_ids[is_mask],
                ])
                reordered_positions = torch.cat([
                    orig_positions[~is_mask],
                    orig_positions[is_mask],
                ])
                input_ids = reordered_ids.unsqueeze(0)  # [1, L]
                position_ids = reordered_positions.unsqueeze(0)  # [1, L]

                # Forward pass with explicit position_ids for correct RoPE
                outputs = self.model(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    use_cache=False,
                )
                logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

                # Get logits for masked positions (at the end of reordered sequence)
                num_unmasked = (~is_mask).sum().item()
                mask_logits = logits[0, num_unmasked:]  # [num_masked, V]

                if mask_logits.size(0) == 0:
                    break

                # Apply temperature and score confidence before stochastic sampling.
                mask_logits = mask_logits / max(temperature, 1e-8)
                probs = F.softmax(mask_logits, dim=-1)
                max_probs, greedy_ids = probs.max(dim=-1)

                sampled_ids = self._sample_token_ids(
                    mask_logits,
                    greedy_ids,
                    top_k=top_k,
                    top_p=top_p,
                )

                # Confidence-based selection
                if confidence_threshold > 0.0:
                    confident = max_probs >= confidence_threshold
                    if confident.any():
                        fill_indices = confident.nonzero(as_tuple=True)[0]
                    else:
                        fill_indices = max_probs.argmax().unsqueeze(0)
                else:
                    fill_indices = max_probs.argmax().unsqueeze(0)

                # Fill predicted tokens at original positions
                mask_positions_orig = is_mask.nonzero(as_tuple=True)[0]
                for idx in fill_indices:
                    pos = mask_positions_orig[idx].item()
                    current_ids[pos] = sampled_ids[idx].item()
                    is_mask[pos] = False

            # Check for EOS in generated tokens (any position after prefix)
            if eos_id is not None:
                new_tokens = current_ids[prefix_len:]
                eos_positions = (new_tokens == eos_id).nonzero(as_tuple=True)
                if eos_positions[0].numel() > 0:
                    cutoff = eos_positions[0][0].item()
                    response_ids = current_ids[prefix_len:prefix_len + cutoff]
                    return response_ids.tolist()

        # Return all generated tokens (strip trailing mask/EOS tokens)
        response_ids_full = current_ids[prefix_len:].tolist()
        while response_ids_full:
            last = response_ids_full[-1]
            if last == self.mask_token_id or last == eos_id:
                response_ids_full.pop()
            else:
                break
        return response_ids_full

    @torch.no_grad()
    def generate_batch(
        self,
        prompts: List[str],
        K: int,
        base_seed: int = 42,
    ) -> List[List[Dict[str, Any]]]:
        """Generate K responses for each prompt in the batch.

        Args:
            prompts: List of prompt texts.
            K: Number of responses per prompt.
            base_seed: Base random seed (each response uses base_seed + k).

        Returns:
            List of shape [batch_size, K], each element is a response dict.
        """
        all_responses = []
        for prompt_text in prompts:
            prompt_responses = []
            for k in range(K):
                seed = base_seed + k if base_seed is not None else None
                response = self.generate(prompt_text, seed=seed)
                prompt_responses.append(response)
            all_responses.append(prompt_responses)
        return all_responses
