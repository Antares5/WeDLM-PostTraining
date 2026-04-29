# reward - 奖励函数体系
from .base import BaseRewardFunction, RewardInputs
from .margin import PolicyRefMarginReward, LengthPenalizedMarginReward
from .deepmath import DeepMathCorrectnessMarginReward, extract_math_final_answer
from .callable import CallableReward
from .clipped import ClippedReward
from .registry import build_reward_function

__all__ = [
    "BaseRewardFunction",
    "RewardInputs",
    "PolicyRefMarginReward",
    "LengthPenalizedMarginReward",
    "DeepMathCorrectnessMarginReward",
    "CallableReward",
    "ClippedReward",
    "build_reward_function",
    "extract_math_final_answer",
]
