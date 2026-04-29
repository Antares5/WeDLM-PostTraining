# WeDLM PostTraining

> WeDLM 后训练统一框架 — 支持 SFT / DPO / GSPO 三大训练范式。

## 目录结构

```
posttraining/
├── wedlm_train/          # 核心训练库 (pip install -e .)
│   ├── config/           # 分层配置 (base/sft/dpo/gspo)
│   ├── data/             # 数据集 (packed/pairwise/prompt)
│   ├── batch/            # WeDLMBatch 构建与 masking
│   ├── model/            # 模型 forward / rotary embedding
│   ├── attention/        # Dense / Magi attention wrapper
│   ├── loss/             # MLM / AR / DPO / GSPO 损失函数
│   ├── reward/           # 奖励函数体系
│   ├── trainer/          # SFT / DPO / GSPO Trainer
│   └── utils/            # logging / seed / device
├── scripts/              # 工具与测试脚本
│   ├── train.py          # 统一训练入口
│   └── smoke_tests/      # 冒烟测试
├── configs/              # 预设配置文件
│   ├── sft_example.yaml
│   ├── dpo_example.yaml
│   └── gspo_deepmath_robust.yaml
├── data/                 # 示例数据
├── tests/                # 单元测试
└── README.md
```

## 快速开始

### 安装依赖

```bash
pip install torch transformers accelerate pyyaml tqdm wandb
```

### 运行冒烟测试

```bash
cd posttraining
python scripts/smoke_tests/test_masking.py
python scripts/smoke_tests/test_dpo_loss.py
python scripts/smoke_tests/test_gspo_loss.py
```

### 运行单元测试

```bash
cd posttraining
pip install pytest
pytest tests/ -v
```

### 启动训练

```bash
# SFT
accelerate launch --multi_gpu --num_processes 4 --mixed_precision bf16 \
    scripts/train.py --config configs/sft_example.yaml

# DPO
accelerate launch --multi_gpu --num_processes 4 --mixed_precision bf16 \
    scripts/train.py --config configs/dpo_example.yaml

# GSPO
accelerate launch --multi_gpu --num_processes 4 --mixed_precision bf16 \
    scripts/train.py --config configs/gspo_deepmath_robust.yaml

# 使用 CLI 覆盖配置
accelerate launch scripts/train.py --config configs/dpo_example.yaml \
    --override dpo.beta=0.5 --override learning_rate=1e-6
```

## 配置说明

- **SFT** (`training_mode: sft`): 使用 MLM + AR loss 的 WeDLM 自监督微调
- **DPO** (`training_mode: dpo`): 使用成对偏好数据的块级 DPO 训练
- **GSPO** (`training_mode: gspo`): 在线采样的组级序列偏好优化

### 自定义 Reward（GSPO）

```python
# my_rewards.py
from wedlm_train.reward import RewardInputs
import torch

def my_custom_reward(inputs: RewardInputs) -> torch.Tensor:
    """自定义奖励函数"""
    return inputs.policy_scores.detach() - 0.1 * torch.tensor(
        inputs.completion_lengths, dtype=torch.float32
    )
```

然后在配置中指定：
```yaml
gspo_reward_source: "callable"
gspo_reward_callable: "my_rewards:my_custom_reward"
```

## 架构关系

```
config ──────────────────────────────────────────────────────────┐
  ↑                                                              │
data ←── tokenization ──→ batch ←── masking                     │
  │                          ↑                                   │
  ↓                          │                                   │
trainer ────→ model ←── attention ←── registry                  │
  │  │          │                ↑                               │
  │  │          ↓                │                               │
  │  └──→ loss ←── block_scores │                               │
  │          ↑                  │                               │
  └──→ reward ──────────────────┘                               │
```

## 迁移说明

此项目将原有的 `dpo/` 和 `finetune/` 目录重构为统一的 `posttraining/wedlm_train/` 包结构。
原有代码保持不变，新代码位于 `posttraining/` 下。

- 原 `dpo/src/config.py` → `wedlm_train/config/{base,sft,dpo,gspo,registry}.py`
- 原 `dpo/src/data.py` → `wedlm_train/data/{packed,pairwise,prompt}_dataset.py + collate.py + utils.py`
- 原 `dpo/src/trainer.py` → `wedlm_train/trainer/{base,sft,dpo,gspo}_trainer.py`
- 原 `dpo/src/loss.py` → `wedlm_train/loss/{mlm,ar,block_scores,dpo,gspo,token_logps}.py`
- 原 `dpo/src/reward.py` → `wedlm_train/reward/{base,margin,deepmath,callable,clipped,registry}.py`
