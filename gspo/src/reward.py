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


class ModelReward:
    """Compute rewards using an independent Reward Model (Phase 3).

    Loads a separate RM (e.g., a sequence-classification or causal-LM head)
    and scores (prompt, response) pairs.  The model is kept on GPU and used
    in no-grad mode during the reward phase of each GSPO training step.

    Supported RM architectures:
    - "sequence_classification": AutoModelForSequenceClassification
    - "causal_lm": AutoModelForCausalLM (uses the last-token logit as score)
    - "auto": tries to infer from model config
    """

    def __init__(
        self,
        model_path: str,
        tokenizer: object,
        device: torch.device,
        model_type: str = "auto",
        torch_dtype: torch.dtype = torch.bfloat16,
        trust_remote_code: bool = True,
    ):
        """
        Args:
            model_path: Path to the Reward Model checkpoint.
            tokenizer: Tokenizer matching the RM.
            device: Target device (GPU).
            model_type: RM architecture ("auto", "sequence_classification", "causal_lm").
            torch_dtype: Data type for the RM.
            trust_remote_code: Whether to trust remote code in model config.
        """
        from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM, AutoConfig

        self.reward_type = "model"
        self.tokenizer = tokenizer
        self.device = device

        # Set pad_token if not configured
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Detect model architecture
        if model_type == "auto":
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
            if hasattr(config, "num_labels") and config.num_labels == 1:
                model_type = "sequence_classification"
            elif hasattr(config, "architectures") and any(
                "ForSequenceClassification" in a for a in config.architectures
            ):
                model_type = "sequence_classification"
            else:
                model_type = "causal_lm"

        logger.info(f"Loading Reward Model from {model_path} (type={model_type})")

        if model_type == "sequence_classification":
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
                device_map=None,  # manual placement
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
                device_map=None,
            )

        self.model = self.model.to(device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self._model_type = model_type
        logger.info(
            f"Reward Model loaded: {sum(p.numel() for p in self.model.parameters()) / 1e9:.2f}B params"
        )

    @torch.no_grad()
    def compute_rewards(
        self,
        prompts: List[str],
        responses: List[str],
        ground_truths: List[str],  # unused for model-based reward, kept for API compatibility
    ) -> torch.Tensor:
        """Compute rewards from RM for a list of (prompt, response) pairs.

        Args:
            prompts: List of prompt texts.
            responses: List of response texts.
            ground_truths: Unused (kept for API compatibility with MathReward).

        Returns:
            rewards: torch.Tensor [N], continuous reward scores.
        """
        if len(prompts) != len(responses):
            raise ValueError(
                f"Mismatched lengths: prompts={len(prompts)}, responses={len(responses)}"
            )

        rewards = []
        # Process in pairs to avoid OOM (tokenize each prompt+response together)
        for prompt, response in zip(prompts, responses):
            reward = self._score_single(prompt, response)
            rewards.append(reward)

        return torch.tensor(rewards, dtype=torch.float32)

    def _score_single(self, prompt: str, response: str) -> float:
        """Score a single (prompt, response) pair with the RM.

        For sequence_classification RM: returns the scalar output.
        For causal_lm RM: uses the last non-padding token's logit as score.
        """
        # Build the full text
        full_text = prompt + response

        # Tokenize
        inputs = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
            padding=False,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Forward
        outputs = self.model(**inputs)

        if self._model_type == "sequence_classification":
            # Sequence classification head returns scalar logit
            score = outputs.logits.squeeze(-1)  # [1] or [1, 1]
            if score.dim() > 0:
                score = score[0]
            return float(score.item())
        else:
            # Causal LM: use the score of the last token (common for AR reward models)
            logits = outputs.logits  # [1, T, V]
            last_idx = inputs["attention_mask"].sum(dim=1) - 1  # [1]
            last_logit = logits[0, last_idx[0], :]  # [V]
            # Use the max logit as the reward signal (or mean of top-k)
            score = last_logit.max()
            return float(score.item())

    @torch.no_grad()
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
            ground_truths: Unused (API compatibility).

        Returns:
            rewards: torch.Tensor [bs, K].
        """
        all_rewards = []
        for prompt, responses in zip(prompts, responses_per_prompt):
            prompt_rewards = self.compute_rewards(
                [prompt] * len(responses), responses, [""] * len(responses)
            )
            all_rewards.append(prompt_rewards)
        return torch.stack(all_rewards, dim=0)
