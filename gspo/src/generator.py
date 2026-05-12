# coding=utf-8
"""On-policy generation engine wrapping wedlm.engine.LLMEngine."""

import logging
from typing import List, Dict, Optional, Any
import torch
from wedlm.engine.llm_engine import LLMEngine
from wedlm.sampling_params import SamplingParams

logger = logging.getLogger(__name__)


class WeDLMGenerator:
    """Thin wrapper around wedlm.engine.LLMEngine for on-policy response generation.

    This class handles the WeDLM iterative decoding process during GSPO training,
    generating K responses per prompt with diverse random seeds.
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
            model_path: Path to model directory (required for LLMEngine initialization).
        """
        self.model = model
        self.tokenizer = tokenizer
        self.gen_config = generation_config
        self.device = device
        self.model_path = model_path

    def _build_sampling_params(self, seed: Optional[int] = None) -> SamplingParams:
        """Build SamplingParams from generation config."""
        return SamplingParams(
            temperature=self.gen_config.get("temperature", 1.0),
            top_p=self.gen_config.get("top_p", 1.0),
            top_k=self.gen_config.get("top_k", 0),
            max_tokens=self.gen_config.get("max_new_tokens", 512),
            wedlm_entropy_threshold=self.gen_config.get(
                "wedlm_entropy_threshold", 0.4
            ),
            wedlm_pos_penalty_factor=self.gen_config.get(
                "wedlm_pos_penalty_factor", 0.02
            ),
        )

    @torch.no_grad()
    def generate(
        self, prompt_text: str, seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """Generate a single response using the high-level LLMEngine.generate API.

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

        sampling_params = self._build_sampling_params(seed)
        engine = LLMEngine(self.model_path or self.model)

        try:
            results = engine.generate(
                prompts=[prompt_text],
                sampling_params=sampling_params,
                use_tqdm=False,
            )
        finally:
            engine.exit()

        if not results:
            raise RuntimeError("Generation returned no results")

        result = results[0]
        response_text = result.get("text", "")
        response_ids = result.get("token_ids", [])

        return self._build_response(prompt_ids, response_ids, response_text)

    def _build_response(
        self,
        prompt_ids: List[int],
        response_ids: List[int],
        response_text: str,
    ) -> Dict[str, Any]:
        """Build the output dict from prompt and response token IDs.

        Args:
            prompt_ids: Token IDs of the prompt.
            response_ids: Token IDs of the generated response.
            response_text: Decoded response text.

        Returns:
            Dict with input_ids, labels, text.
        """
        prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long)
        response_tensor = torch.tensor(response_ids, dtype=torch.long)

        full_input_ids = torch.cat([prompt_tensor, response_tensor], dim=0)
        labels = full_input_ids.clone()
        labels[: len(prompt_ids)] = -100  # mask prompt tokens

        return {
            "input_ids": full_input_ids,
            "labels": labels,
            "text": response_text,
        }

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
