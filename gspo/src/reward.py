# coding=utf-8
"""Math rule-based reward computation for GSPO training (Phase 1).

Phase 1 uses rule-based binary reward from math problem ground truth answers.
No independent Reward Model is loaded.
"""

import re
import logging
from typing import List, Optional
import torch

logger = logging.getLogger(__name__)


class MathReward:
    """Compute binary rewards by comparing generated answers with ground truth.

    Supports multiple math dataset formats:
    - GSM8K: answers after '####'
    - MATH: answers inside \\boxed{...}
    - Generic: last numeric value
    """

    def __init__(
        self,
        reward_type: str = "math_verify",
        tokenizer: Optional[object] = None,
    ):
        """
        Args:
            reward_type: "math_verify" (Phase 1) or "model" (Phase 3).
            tokenizer: Tokenizer (reserved for future answer extraction needs).
        """
        self.reward_type = reward_type
        self.tokenizer = tokenizer

        if reward_type not in ["math_verify", "model"]:
            raise ValueError(f"Unknown reward_type: {reward_type}")

    def compute_rewards(
        self,
        prompts: List[str],
        responses: List[str],
        ground_truths: List[str],
    ) -> torch.Tensor:
        """Compute binary rewards for a list of (prompt, response, ground_truth).

        Args:
            prompts: List of prompt texts.
            responses: List of response texts.
            ground_truths: List of ground truth answer strings.

        Returns:
            rewards: torch.Tensor [N], each element is 1.0 (correct) or 0.0 (incorrect).
        """
        if len(responses) != len(ground_truths):
            raise ValueError(
                f"Mismatched lengths: responses={len(responses)}, ground_truths={len(ground_truths)}"
            )

        rewards = []
        for response, gt in zip(responses, ground_truths):
            reward = self._compute_single_reward(response, gt)
            rewards.append(reward)

        return torch.tensor(rewards, dtype=torch.float32)

    def _compute_single_reward(self, response: str, ground_truth: str) -> float:
        """Compute reward for a single response.

        Args:
            response: Generated response text.
            ground_truth: Ground truth answer string.

        Returns:
            1.0 if the answer is correct, 0.0 otherwise.
        """
        if not response or not response.strip():
            return 0.0

        if not ground_truth or not ground_truth.strip():
            return 0.0

        extracted = self.extract_answer(response)
        if extracted is None:
            return 0.0

        return 1.0 if self.verify_answer(extracted, ground_truth.strip()) else 0.0

    def extract_answer(self, text: str) -> Optional[str]:
        """Extract the final answer from a generated response.

        Attempts multiple extraction strategies in order:
        1. GSM8K-style: '#### <number>'
        2. MATH-style: '\\boxed{...}'
        3. 'The answer is ...' / 'Answer: ...'
        4. Last numeric value in the text

        Args:
            text: Generated response text.

        Returns:
            Extracted answer string, or None if no answer found.
        """
        if not text or not text.strip():
            return None

        text = text.strip()

        # Strategy 1: GSM8K format - "#### <number>"
        match = re.search(r"####\s*(-?[\d,.\/]+)", text)
        if match:
            return self._normalize_numeric(match.group(1))

        # Strategy 2: MATH format - "\boxed{...}"
        match = re.findall(r"\\boxed\{([^}]*)\}", text)
        if match:
            # Take the last boxed expression
            return match[-1].strip()

        # Strategy 3: "The answer is ..." / "Answer: ..."
        patterns = [
            r"(?:the\s+)?answer\s+is\s*:?\s*([^\n\.]+)",
            r"answer\s*:\s*([^\n\.]+)",
            r"(?:=\s*)(-?[\d,.\/]+)\s*$",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                candidate = match.group(1).strip()
                # Remove trailing punctuation
                candidate = re.sub(r"[.,;:!?]+$", "", candidate)
                if candidate:
                    return candidate

        # Strategy 4: Last number in the text
        numbers = re.findall(r"-?[\d,]+\.?\d*", text)
        if numbers:
            return numbers[-1]

        return None

    def _normalize_numeric(self, s: str) -> str:
        """Normalize a numeric string by removing commas and extra spaces."""
        s = s.replace(",", "").replace(" ", "")
        return s

    def verify_answer(self, extracted: str, ground_truth: str) -> bool:
        """Verify if the extracted answer matches the ground truth.

        Args:
            extracted: Extracted answer string.
            ground_truth: Ground truth answer string.

        Returns:
            True if answers match, False otherwise.
        """
        if not extracted or not ground_truth:
            return False

        extracted = extracted.strip()
        ground_truth = ground_truth.strip()

        # Exact string match (case-insensitive)
        if extracted.lower() == ground_truth.lower():
            return True

        # Try numeric comparison
        try:
            ext_num = float(self._normalize_numeric(extracted))
            gt_num = float(self._normalize_numeric(ground_truth))

            # Tolerance-based comparison
            if gt_num == 0.0:
                return abs(ext_num) < 1e-5
            rel_err = abs(ext_num - gt_num) / max(abs(gt_num), 1e-8)
            abs_err = abs(ext_num - gt_num)
            return rel_err < 1e-3 or abs_err < 1e-5
        except (ValueError, TypeError):
            pass

        # Try SymPy symbolic comparison (for expression-type answers)
        try:
            import sympy as sp

            ext_expr = sp.simplify(extracted)
            gt_expr = sp.simplify(ground_truth)
            return sp.simplify(ext_expr - gt_expr) == 0
        except (ImportError, Exception):
            pass

        return False

    def compute_batch_rewards(
        self,
        prompts: List[str],
        responses_per_prompt: List[List[str]],
        ground_truths: List[str],
    ) -> torch.Tensor:
        """Compute rewards for a batch of prompts, each with K responses.

        Args:
            prompts: List of prompt texts, length [bs].
            responses_per_prompt: List of [K] responses per prompt, shape [bs, K].
            ground_truths: List of ground truths, length [bs].

        Returns:
            rewards: torch.Tensor [bs, K].
        """
        all_rewards = []
        for prompt, responses, gt in zip(prompts, responses_per_prompt, ground_truths):
            prompt_rewards = self.compute_rewards(
                [prompt] * len(responses), responses, [gt] * len(responses)
            )
            all_rewards.append(prompt_rewards)
        return torch.stack(all_rewards, dim=0)
