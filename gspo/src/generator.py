# coding=utf-8
"""WeDLM generator interface for GSPO on-policy generation.

Provides:
  - WeDLMGenerator: wraps wedlm.engine.LLMEngine for batch generation.
  - MockGenerator: returns synthetic token IDs for smoke testing.
"""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class BaseGenerator:
    """Abstract interface for response generation.

    GSPO training needs to sample G responses per prompt from the current
    policy.  The generator is responsible for:
      - Loading / updating model weights
      - Running WeDLM sliding-window decoding
      - Returning token-id lists
    """

    def generate(
        self,
        prompts: List[List[int]],
        sampling_params: "SamplingParams",
    ) -> List[List[int]]:
        """Generate one response per prompt.

        Args:
            prompts: List of tokenized prompts (list of int lists).
            sampling_params: wedlm.sampling_params.SamplingParams instance.

        Returns:
            List of completion token-id lists (one per prompt).
        """
        raise NotImplementedError

    def update_weights(self, model_path: str):
        """Reload model weights from a checkpoint directory."""
        raise NotImplementedError

    def release_memory(self):
        """Free GPU memory used by the generator (KV cache, etc.)."""
        raise NotImplementedError


class WeDLMGenerator(BaseGenerator):
    """Real generator using wedlm.engine.LLMEngine."""

    def __init__(
        self,
        model_path: str,
        block_size: int = 4096,
        window_size: int = 16,
        gpu_memory_utilization: float = 0.3,
        max_num_seqs: int = 16,
        max_model_len: int = 4096,
        tensor_parallel_size: int = 1,
    ):
        from wedlm.engine.llm_engine import LLMEngine

        self.model_path = model_path
        self.block_size = block_size
        self.window_size = window_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_num_seqs = max_num_seqs
        self.max_model_len = max_model_len
        self.tensor_parallel_size = tensor_parallel_size
        self._engine: Optional[LLMEngine] = None

    def _get_engine(self) -> "LLMEngine":
        if self._engine is None:
            from wedlm.engine.llm_engine import LLMEngine
            logger.info(f"Initializing WeDLMGenerator engine from {self.model_path}")
            self._engine = LLMEngine(
                model=self.model_path,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                wedlm_window_size=self.window_size,
                kvcache_block_size=self.block_size,
                max_num_seqs=self.max_num_seqs,
                max_model_len=self.max_model_len,
            )
        return self._engine

    def generate(
        self,
        prompts: List[List[int]],
        sampling_params: "SamplingParams",
    ) -> List[List[int]]:
        engine = self._get_engine()
        results = engine.generate(
            prompts=prompts,
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        # results is list of dicts, each with "token_ids" key
        return [r["token_ids"] for r in results]

    def update_weights(self, model_path: str):
        """Destroy and recreate the engine to pick up new weights."""
        if self._engine is not None:
            logger.info("Releasing old WeDLM engine for weight update")
            self._engine.exit()
            self._engine = None
        self.model_path = model_path
        # Engine will be lazily recreated on next generate()

    def release_memory(self):
        if self._engine is not None:
            logger.info("Releasing WeDLM engine memory")
            self._engine.exit()
            self._engine = None
            import torch
            torch.cuda.empty_cache()


class MockGenerator(BaseGenerator):
    """Mock generator that returns prompt IDs as fake completions.

    Used for smoke testing the training pipeline without a real WeDLM engine.
    """

    def __init__(self, fixed_length: int = 32):
        self.fixed_length = fixed_length

    def generate(
        self,
        prompts: List[List[int]],
        sampling_params: "SamplingParams",
    ) -> List[List[int]]:
        import random
        random.seed(42)
        results = []
        for prompt_ids in prompts:
            # Generate fake completion of fixed_length random token IDs
            fake_completion = [random.randint(100, 50000) for _ in range(self.fixed_length)]
            results.append(fake_completion)
        return results

    def update_weights(self, model_path: str):
        pass  # no-op for mock

    def release_memory(self):
        pass
