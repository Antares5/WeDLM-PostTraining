#!/usr/bin/env python
# coding=utf-8
"""Smoke test for GSPO generator and reward modules.

Validates:
- MathReward answer extraction and verification
- (generator test requires model loading, skipped in smoke test)
"""

import os
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from src.reward import MathReward


def test_math_reward_gsm8k():
    """Test MathReward with GSM8K format answers."""
    reward = MathReward(reward_type="math_verify")

    # GSM8K style: answer follows "####"
    response = """Let me solve this step by step.
Aaron runs 2 miles in 16 minutes.
Pace = 16/2 = 8 minutes per mile.
Vanessa is twice as slow, so 8 * 2 = 16 minutes per mile.
She runs 4 miles, so 4 * 16 = 64 minutes.
#### 64"""

    ground_truth = "64"
    r = reward._compute_single_reward(response, ground_truth)
    logger.info(f"GSM8K test: extracted answer, reward={r}")
    assert r == 1.0, f"Expected 1.0, got {r}"

    # Test wrong answer
    r_wrong = reward._compute_single_reward(response, "65")
    assert r_wrong == 0.0, f"Expected 0.0 for wrong answer, got {r_wrong}"

    logger.info("✓ GSM8K reward test PASSED")


def test_math_reward_boxed():
    """Test MathReward with MATH format (\\boxed{...}) answers."""
    reward = MathReward(reward_type="math_verify")

    # MATH style: answer inside \boxed{}
    response = """Solution:
We have $\\det(AB) = (\\det A)(\\det B) = (2)(12) = \\boxed{24}$.
Final Answer: $\\boxed{24}$."""

    ground_truth = "24"
    r = reward._compute_single_reward(response, ground_truth)
    logger.info(f"MATH boxed test: reward={r}")
    assert r == 1.0, f"Expected 1.0, got {r}"

    # Test with fraction
    response2 = """The answer is \\boxed{\\frac{3}{4}}."""
    r2 = reward._compute_single_reward(response2, "3/4")
    logger.info(f"MATH fraction test: reward={r2}")

    logger.info("✓ MATH boxed reward test PASSED")


def test_math_reward_answer_is():
    """Test MathReward with 'answer is' format."""
    reward = MathReward(reward_type="math_verify")

    response = "The answer is 42."
    r = reward._compute_single_reward(response, "42")
    logger.info(f"'Answer is' test: reward={r}")
    assert r == 1.0, f"Expected 1.0, got {r}"

    response2 = "I think the answer is 3.14 approximately"
    r2 = reward._compute_single_reward(response2, "3.14")
    logger.info(f"'Answer is' float test: reward={r2}")
    assert r2 == 1.0

    logger.info("✓ 'Answer is' reward test PASSED")


def test_math_reward_numeric_tolerance():
    """Test numeric tolerance comparison."""
    reward = MathReward(reward_type="math_verify")

    # Slightly different but within tolerance
    r = reward._compute_single_reward("The answer is 3.14159", "3.1416")
    logger.info(f"Numeric tolerance test: reward={r}")
    # 3.14159 vs 3.1416: rel_err = |diff|/3.1416 ≈ 0.00001/3.1416 ≈ 3e-6 < 1e-3
    assert r == 1.0, f"Expected 1.0 within tolerance, got {r}"

    logger.info("✓ Numeric tolerance test PASSED")


def test_math_reward_empty():
    """Test edge case handling."""
    reward = MathReward(reward_type="math_verify")

    # Empty response
    r = reward._compute_single_reward("", "42")
    assert r == 0.0, f"Empty response should be 0, got {r}"

    # Empty ground truth
    r = reward._compute_single_reward("answer is 42", "")
    assert r == 0.0

    # No extractable answer
    r = reward._compute_single_reward("This is a response without any answer", "42")
    assert r == 0.0

    logger.info("✓ Edge case tests PASSED")


def test_batch_rewards():
    """Test batch reward computation."""
    reward = MathReward(reward_type="math_verify")

    prompts = ["What is 1+1?", "What is 2+2?"]
    responses = [["The answer is 2.", "I think 3."], ["#### 4", "The answer is 5"]]
    ground_truths = ["2", "4"]

    rewards_tensor = reward.compute_batch_rewards(prompts, responses, ground_truths)
    logger.info(f"Batch rewards shape: {rewards_tensor.shape}")
    logger.info(f"Batch rewards:\n{rewards_tensor}")
    assert rewards_tensor.shape == (2, 2)
    assert rewards_tensor[0, 0].item() == 1.0  # correct
    assert rewards_tensor[0, 1].item() == 0.0  # wrong
    assert rewards_tensor[1, 0].item() == 1.0  # correct
    assert rewards_tensor[1, 1].item() == 0.0  # wrong

    logger.info("✓ Batch rewards test PASSED")


if __name__ == "__main__":
    test_math_reward_gsm8k()
    test_math_reward_boxed()
    test_math_reward_answer_is()
    test_math_reward_numeric_tolerance()
    test_math_reward_empty()
    test_batch_rewards()
    logger.info("All generator/reward tests passed!")
