# coding=utf-8
"""DeepMath reward combining margin shaping and exact-match correctness.
Migrated from dpo/src/reward.py (DeepMathCorrectnessMarginReward + answer extraction)."""

from fractions import Fraction
import math
import re
from typing import List, Optional, Tuple

import torch
from .base import BaseRewardFunction, RewardInputs


# ── Answer extraction utilities ──────────────────────────────

_BOOL_TRUE_VALUES = {"yes", "true", "y", "1"}
_BOOL_FALSE_VALUES = {"no", "false", "n", "0"}


def _extract_boxed_answer(text: str) -> Optional[str]:
    marker = "\\boxed{"
    matches: List[str] = []
    search_start = 0

    while True:
        pos = text.find(marker, search_start)
        if pos == -1:
            break

        idx = pos + len(marker)
        depth = 1
        content_start = idx
        while idx < len(text):
            ch = text[idx]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    matches.append(text[content_start:idx])
                    idx += 1
                    break
            idx += 1

        search_start = pos + 1

    if len(matches) > 0:
        return matches[-1]

    return None


def _strip_enclosing_math_fences(text: str) -> str:
    value = text.strip()
    pairs = [("$$", "$$"), ("$", "$"), ("\\[", "\\]"), ("\\(", "\\)")]

    changed = True
    while changed:
        changed = False
        for left, right in pairs:
            if value.startswith(left) and value.endswith(right):
                inner = value[len(left): len(value) - len(right)].strip()
                if inner:
                    value = inner
                    changed = True

    return value


def _split_candidate_lines(text: str) -> List[str]:
    candidates: List[str] = []
    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        for segment in re.split(r"\\\\", line):
            cleaned = segment.strip()
            if cleaned:
                candidates.append(cleaned)
    return candidates


def _strip_latex_environment_tokens(text: str) -> str:
    value = text.strip()
    value = re.sub(r"^\\begin\{[^{}]+\}\s*", "", value)
    while True:
        stripped = re.sub(r"\\end\{[^{}]+\}\s*$", "", value).strip()
        if stripped == value:
            break
        value = stripped
    return value


def _is_structural_only_line(text: str) -> bool:
    value = _strip_enclosing_math_fences(text)
    value = value.strip()
    if value == "":
        return True

    if re.fullmatch(r"\\(begin|end)\{[^{}]+\}", value):
        return True

    if value in {"\\[", "\\]", "\\(", "\\)", "$$", "$"}:
        return True

    if re.fullmatch(r"[{}]+", value):
        return True

    return False


def _normalize_math_answer(text: str) -> str:
    normalized = _strip_enclosing_math_fences(text)
    normalized = normalized.strip()
    normalized = normalized.replace("$", "")
    normalized = normalized.replace("\\left", "")
    normalized = normalized.replace("\\right", "")
    normalized = normalized.replace("\\!", "")
    normalized = normalized.replace("\u2212", "-")
    normalized = re.sub(r"\s+", "", normalized)
    normalized = normalized.replace("\\,", "")
    normalized = normalized.rstrip(".。")
    return normalized


def _parse_bool_answer(answer: str) -> Optional[bool]:
    lowered = re.sub(r"[^a-z0-9]+", "", answer.strip().lower())
    if lowered in _BOOL_TRUE_VALUES:
        return True
    if lowered in _BOOL_FALSE_VALUES:
        return False
    return None


def _parse_numeric_answer(answer: str) -> Optional[float]:
    value = answer.strip()
    if value == "":
        return None

    try:
        if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", value):
            return float(value)
    except ValueError:
        return None

    fraction_match = re.fullmatch(r"([+-]?\d+)\/([+-]?\d+)", value)
    if fraction_match is not None:
        den = int(fraction_match.group(2))
        if den == 0:
            return None
        num = int(fraction_match.group(1))
        return float(Fraction(num, den))

    latex_fraction_match = re.fullmatch(
        r"\\d?frac\{([+-]?\d+(?:\.\d+)?)\}\{([+-]?\d+(?:\.\d+)?)\}",
        value,
    )
    if latex_fraction_match is not None:
        try:
            num = float(latex_fraction_match.group(1))
            den = float(latex_fraction_match.group(2))
            if den == 0:
                return None
            return num / den
        except ValueError:
            return None

    return None


def _is_simple_symbolic_answer(answer: str) -> bool:
    if len(answer) == 0 or len(answer) > 32:
        return False
    return re.fullmatch(r"[a-zA-Z0-9_+\-]+", answer) is not None


def _answers_match(
    pred_answer: str,
    gold_answer: str,
    numeric_atol: float,
    numeric_rtol: float,
) -> Tuple[bool, bool]:
    pred = _normalize_math_answer(pred_answer)
    gold = _normalize_math_answer(gold_answer)

    if pred == "" or gold == "":
        return False, False

    if pred == gold:
        return True, True

    pred_bool = _parse_bool_answer(pred)
    gold_bool = _parse_bool_answer(gold)
    if pred_bool is not None and gold_bool is not None:
        return pred_bool == gold_bool, True

    pred_num = _parse_numeric_answer(pred)
    gold_num = _parse_numeric_answer(gold)
    if pred_num is not None and gold_num is not None:
        matched = math.isclose(pred_num, gold_num, abs_tol=numeric_atol, rel_tol=numeric_rtol)
        return matched, True

    if _is_simple_symbolic_answer(pred) and _is_simple_symbolic_answer(gold):
        return False, True

    return False, False


def extract_math_final_answer(text: str) -> str:
    """Extract a final-answer string from free-form math solution text."""
    boxed = _extract_boxed_answer(text)
    if boxed is not None:
        return _normalize_math_answer(boxed)

    marker_patterns = [
        r"####\s*(.+)$",
        r"Final\s*Answer\s*[:：]\s*(.+)$",
        r"Answer\s*[:：]\s*(.+)$",
        r"(?:Final\s*)?Answer\s*(?:is|=)\s*(.+)$",
        r"答案\s*[:：]\s*(.+)$",
    ]
    for pattern in marker_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if match is not None:
            return _normalize_math_answer(match.group(1))

    lines = _split_candidate_lines(text)
    for line in reversed(lines):
        if _is_structural_only_line(line):
            continue

        cleaned = _strip_latex_environment_tokens(line)
        cleaned = _strip_enclosing_math_fences(cleaned)
        cleaned = cleaned.strip()
        if cleaned == "" or _is_structural_only_line(cleaned):
            continue

        normalized = _normalize_math_answer(cleaned)
        if normalized != "" and not normalized.startswith("\\end{"):
            return normalized

    fallback = _strip_latex_environment_tokens(_strip_enclosing_math_fences(text))
    return _normalize_math_answer(fallback)


# ── DeepMath reward function ────────────────────────────────

class DeepMathCorrectnessMarginReward(BaseRewardFunction):
    """DeepMath reward combining margin shaping and exact-match correctness."""

    def __init__(
        self,
        beta: float,
        correct_bonus: float,
        wrong_penalty: float = 0.0,
        length_penalty: float = 0.0,
        numeric_atol: float = 1e-6,
        numeric_rtol: float = 1e-5,
        penalize_only_when_confident: bool = True,
    ):
        self.beta = float(beta)
        self.correct_bonus = float(correct_bonus)
        self.wrong_penalty = float(wrong_penalty)
        self.length_penalty = float(length_penalty)
        self.numeric_atol = float(numeric_atol)
        self.numeric_rtol = float(numeric_rtol)
        self.penalize_only_when_confident = bool(penalize_only_when_confident)

    def _resolve_completion_texts(self, inputs: RewardInputs) -> Optional[List[str]]:
        if inputs.candidate_completion_texts is not None:
            return inputs.candidate_completion_texts

        if inputs.candidate_completion_ids is None:
            return None

        if inputs.tokenizer is None:
            return None

        texts: List[str] = []
        for completion_ids in inputs.candidate_completion_ids:
            texts.append(
                inputs.tokenizer.decode(completion_ids.tolist(), skip_special_tokens=True)
            )
        return texts

    def __call__(self, inputs: RewardInputs) -> torch.Tensor:
        rewards = self.beta * (
            inputs.policy_scores.detach() - inputs.reference_scores.detach()
        )

        completion_texts = self._resolve_completion_texts(inputs)
        prompt_metadata = inputs.prompt_metadata
        if completion_texts is not None and prompt_metadata is not None:
            for i, completion_text in enumerate(completion_texts):
                group_idx = int(inputs.group_ids[i].item())
                if group_idx < 0 or group_idx >= len(prompt_metadata):
                    continue

                metadata = prompt_metadata[group_idx] or {}
                gold_text = metadata.get("ground_truth_answer")
                if not isinstance(gold_text, str) or len(gold_text.strip()) == 0:
                    continue

                pred_answer = extract_math_final_answer(completion_text)
                gold_answer = extract_math_final_answer(gold_text)

                matched, confident = _answers_match(
                    pred_answer=pred_answer,
                    gold_answer=gold_answer,
                    numeric_atol=self.numeric_atol,
                    numeric_rtol=self.numeric_rtol,
                )

                if matched:
                    rewards[i] = rewards[i] + self.correct_bonus
                elif self.wrong_penalty > 0 and (
                    (not self.penalize_only_when_confident) or confident
                ):
                    rewards[i] = rewards[i] - self.wrong_penalty

        if self.length_penalty > 0:
            lengths = torch.tensor(
                inputs.completion_lengths,
                dtype=rewards.dtype,
                device=rewards.device,
            )
            rewards = rewards - self.length_penalty * lengths

        return rewards
