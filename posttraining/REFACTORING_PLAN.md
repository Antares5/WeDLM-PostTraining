# WeDLM-PostTraining 重构计划

> 目标：将分散在 `dpo/`、`finetune/` 的 SFT/DPO/GSPO 后训练代码整合为规范化工程结构。

---

## 一、现状诊断

### 1.1 当前文件布局

```
WeDLM-PostTraining/
├── dpo/                          # 主力训练代码（SFT+DPO+GSPO，持续迭代）
│   ├── train.py                  # 🔴 入口脚本，40+ CLI 参数硬编码
│   ├── src/
│   │   ├── config.py             # 🔴 80+ 字段单一扁平 dataclass
│   │   ├── data.py               # 🔴 3 个 Dataset + collate 混在单文件
│   │   ├── trainer.py            # 🔴 SFT/DPO/GSPO 三种模式耦合在一个类
│   │   ├── loss.py               # 🟡 部分函数未在主线使用
│   │   ├── batch.py / model.py / masking.py / attention.py / reward.py
│   │   └── __init__.py
│   ├── configs/
│   ├── scripts/                  # 🟡 smoke tests + 分析脚本混放
│   └── data/
│
├── finetune/                     # 🟠 早期 SFT 代码，与 dpo/ 大量重复
│   ├── train.py
│   └── src/（与 dpo/src 结构镜像但功能子集）
│
├── evaluation/                   # 🟢 评估套件，相对独立
├── wedlm/                        # 🟢 核心推理引擎，不改动
├── hf_compat/                    # 🟢 HuggingFace 兼容层
└── images/ / paper/
```

### 1.2 核心问题清单

| # | 问题 | 严重度 | 所在文件 |
|---|------|--------|---------|
| 1 | **代码重复** — `finetune/` 与 `dpo/` 中 SFT 逻辑几乎一致，各维护一份 | 🔴 高 | finetune/*, dpo/* |
| 2 | **巨型单文件** — `trainer.py` (~800 行)、`data.py` (~750 行)、`config.py` (~300 行) | 🔴 高 | dpo/src/* |
| 3 | **配置扁平化** — 单一 `WeDLMTrainingConfig` dataclass 容纳 SFT/DPO/GSPO 所有参数（80+ 字段），不同模式的字段互相可见 | 🔴 高 | dpo/src/config.py |
| 4 | **模式耦合** — `WeDLMTrainer.__init__` 中用 if/elif 三分支初始化不同模式的数据/模型 | 🔴 高 | dpo/src/trainer.py |
| 5 | **CLI 参数爆炸** — `train.py` 的 `parse_args()` 硬编码 40+ 个 `--gspo_*` / `--dpo_*` 参数 | 🟡 中 | dpo/train.py |
| 6 | **命名不一致** — `WeDLMPackedDataset` 在两处有不同实现；`compute_masked_token_logps` 定义但主线未用 | 🟡 中 | dpo/src/data.py, dpo/src/loss.py |
| 7 | **脚本分类不清** — smoke tests、数据下载、数据分析脚本混在 `scripts/` | 🟡 中 | dpo/scripts/ |
| 8 | **缺乏测试体系** — 仅有 smoke tests，无单元测试 | 🟡 中 | dpo/scripts/smoke_test_*.py |
| 9 | **文档缺失** — `TODO` 文件仅一行 "merge train_dataset code" | 🟡 中 | dpo/TODO |
| 10 | **包导出不完整** — `__init__.py` 使用显式列表但未覆盖所有公开符号 | 🟢 低 | dpo/src/__init__.py |

---

## 二、目标架构

### 2.1 目录结构

```
posttraining/
├── pyproject.toml                 # 独立包配置（可选）
├── README.md                      # 重构后整体说明
│
├── wedlm_train/                   # 核心训练库（pip install -e .）
│   ├── __init__.py                # 顶层公开 API
│   │
│   ├── config/
│   │   ├── __init__.py
│   │   ├── base.py                # BaseTrainingConfig（共用参数）
│   │   ├── sft.py                 # SFTConfig
│   │   ├── dpo.py                 # DPOConfig
│   │   ├── gspo.py                # GSPOConfig
│   │   └── registry.py            # from_yaml 工厂（按 training_mode 分发）
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── tokenization.py        # 公用 tokenization 工具
│   │   ├── packed_dataset.py      # WeDLMPackedDataset（SFT 用）
│   │   ├── pairwise_dataset.py    # WeDLMPairwiseDataset（DPO 用）
│   │   ├── prompt_dataset.py      # WeDLMPromptDataset（GSPO 用）
│   │   ├── collate.py             # 各模式 collate_fn
│   │   └── utils.py               # im_end_token_id 等
│   │
│   ├── batch/
│   │   ├── __init__.py
│   │   ├── wedlm_batch.py         # WeDLMBatch + build_wedlm_batch
│   │   └── masking.py             # sample_block_mask_ratios, reorder_block 等
│   │
│   ├── model/
│   │   ├── __init__.py
│   │   ├── forward.py             # wedlm_forward, wedlm_attention_forward
│   │   └── rotary.py              # apply_rotary_pos_emb
│   │
│   ├── attention/
│   │   ├── __init__.py
│   │   ├── base.py                # 抽象 wrapper 接口
│   │   ├── dense.py               # DenseAttentionWrapper
│   │   ├── magi.py                # MagiAttentionWrapper
│   │   └── registry.py            # check_backend_available, get_* 工厂
│   │
│   ├── loss/
│   │   ├── __init__.py
│   │   ├── mlm.py                 # compute_mlm_loss
│   │   ├── ar.py                  # compute_ar_loss
│   │   ├── block_scores.py        # compute_block_scores（公有）
│   │   ├── dpo.py                 # compute_dpo_loss（仅公式）
│   │   └── gspo.py                # compute_gspo_loss（仅公式）
│   │
│   ├── reward/
│   │   ├── __init__.py
│   │   ├── base.py                # RewardInputs, BaseRewardFunction
│   │   ├── margin.py              # PolicyRefMargin, LengthPenalizedMargin
│   │   ├── deepmath.py            # DeepMathCorrectnessMargin + 答案抽取工具
│   │   ├── callable.py            # CallableReward
│   │   ├── clipped.py             # ClippedReward wrapper
│   │   └── registry.py            # build_reward_function 工厂
│   │
│   ├── trainer/
│   │   ├── __init__.py
│   │   ├── base.py                # BaseTrainer（优化器/调度器/save/logging）
│   │   ├── sft_trainer.py         # SFTTrainer(BaseTrainer)
│   │   ├── dpo_trainer.py         # DPOTrainer(BaseTrainer)
│   │   ├── gspo_trainer.py        # GSPOTrainer(BaseTrainer)
│   │   └── callbacks.py           # wandb / checkpoint / lr logging
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logging.py             # logger 配置
│       ├── seed.py                # 随机种子
│       └── device.py              # accelerator 辅助
│
├── scripts/                       # 工具与测试脚本
│   ├── train.py                   # 统一入口（轻量 CLI）
│   ├── download/
│   │   └── download_alpaca_sft.py
│   ├── analysis/
│   │   └── analyze_gspo_prompt_data.py
│   └── smoke_tests/
│       ├── test_dpo_data.py
│       ├── test_dpo_loss.py
│       ├── test_dpo_train_step.py
│       ├── test_gspo_data.py
│       ├── test_gspo_loss.py
│       ├── test_gspo_reward.py
│       └── test_gspo_train_step.py
│
├── configs/                       # 预设配置文件
│   ├── sft_example.yaml
│   ├── dpo_example.yaml
│   └── gspo_deepmath_robust.yaml
│
├── data/                          # 示例数据（clone 自原 dpo/data/）
│   ├── train.jsonl
│   └── orca_rlhf.jsonl
│
└── tests/                         # 单元测试
    ├── __init__.py
    ├── test_config.py
    ├── test_masking.py
    ├── test_loss_mlm.py
    ├── test_loss_dpo.py
    ├── test_loss_gspo.py
    ├── test_reward.py
    └── conftest.py
```

### 2.2 模块依赖关系

```
config ─────────────────────────────────────────────────────────────┐
  ↑                                                                 │
data ←── tokenization ──→ batch ←── masking                        │
  │                          ↑                                      │
  ↓                          │                                      │
trainer ────→ model ←── attention ←── registry                    │
  │  │          │                ↑                                  │
  │  │          ↓                │                                  │
  │  └──→ loss ←── block_scores │                                  │
  │          ↑                  │                                  │
  └──→ reward ──────────────────┘                                  │
```

依赖方向：trainer → loss / reward / model；loss → model 输出；config 被所有模块引用（构造注入，避免全局单例）。

---

## 三、分阶段实施计划

### Phase 1: 基础骨架搭建（预计 1-2 天）

**目标**：建立新目录结构，消除 `finetune/` 与 `dpo/` 的重复。

| 步骤 | 任务 | 产出 |
|------|------|------|
| 1.1 | 创建 `posttraining/wedlm_train/` 目录骨架（所有 `__init__.py`） | 空包结构 |
| 1.2 | 从 `dpo/src/config.py` 拆分出 `base.py` / `sft.py` / `dpo.py` / `gspo.py` | 分层 config |
| 1.3 | 迁移共用模块（masking, batch, model, attention 等）到新位置，保持接口不变 | 核心无业务逻辑模块就位 |
| 1.4 | 合并 `finetune/src/` 与 `dpo/src/data.py` 中重复逻辑，统一到新 `data/` 包 | 单一数据源 |

**关键决策**：
- `finetune/src/data.py` 中有 `WeDLMShuffledPackedDataset` 包装类，保留到新位置
- `dpo/src/loss.py` 中 `compute_masked_token_logps` 在 `compute_block_scores` 内部已被内联，但保留为公开工具函数供 smoke tests 使用

---

### Phase 2: 业务逻辑拆分（预计 2-3 天）

**目标**：将巨型 `trainer.py` 按训练模式拆分为独立类，降低耦合。

| 步骤 | 任务 | 产出 |
|------|------|------|
| 2.1 | 提取 `BaseTrainer`：优化器/调度器初始化、`train()` 主循环、checkpoint、wandb logging | `trainer/base.py` |
| 2.2 | 从 `WeDLMTrainer` 中提取 `SFTTrainer`，继承 `BaseTrainer` | `trainer/sft_trainer.py` |
| 2.3 | 提取 `DPOTrainer`，独立管理 reference model + DPO 特有逻辑 | `trainer/dpo_trainer.py` |
| 2.4 | 提取 `GSPOTrainer`，独立管理 online rollout + reward 打分 | `trainer/gspo_trainer.py` |
| 2.5 | 验证每种 trainer 可独立实例化和进行单步训练 | smoke tests 全绿 |

**关键接口约定**：

```python
# BaseTrainer 核心签名
class BaseTrainer:
    def __init__(self, config: BaseTrainingConfig, accelerator: Accelerator):
        ...
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        raise NotImplementedError
    def train(self) -> None:
        # 标准循环：dataloader → train_step → backward → optimizer.step → logging
```

---

### Phase 3: CLI 与配置体系优化（预计 1 天）

**目标**：精简 CLI，使配置管理更清晰。

| 步骤 | 任务 | 产出 |
|------|------|------|
| 3.1 | 重写 `scripts/train.py`：仅保留 `--config` / `--training_mode` / 高频 override 参数（~10 个） | 精简入口 |
| 3.2 | 实现配置覆盖逻辑：`--override key=value` 通用覆盖（替代 40+ 独立 CLI 参数） | 通用 CLI |
| 3.3 | 实现 `config/registry.py`：按 `training_mode` 自动选择对应 Config 子类 | 配置工厂 |
| 3.4 | 为每个模式提供最佳实践的 example yaml | `configs/*.yaml` |

**新 CLI 设计**：

```bash
# 精简入口
python scripts/train.py --config configs/gspo_deepmath_robust.yaml
python scripts/train.py --config configs/dpo_example.yaml --override dpo_beta=0.5 gspo_group_size=8
```

---

### Phase 4: 测试与文档（预计 1-2 天）

**目标**：建立测试体系，补全文档。

| 步骤 | 任务 | 产出 |
|------|------|------|
| 4.1 | 迁移现有 smoke tests 到 `scripts/smoke_tests/`，适配新导入路径 | 冒烟测试可用 |
| 4.2 | 为核心函数编写单元测试：loss 函数、masking 逻辑、reward 计算 | `tests/` |
| 4.3 | 编写 `posttraining/README.md`：快速开始、配置说明、架构图 | 文档 |

---

### Phase 5: 清理与收尾（预计 0.5 天）

| 步骤 | 任务 |
|------|------|
| 5.2 | 更新根 `pyproject.toml`，将 `posttraining/wedlm_train` 加入包路径 |
| 5.3 | 运行全量 smoke tests 确认端到端可用 |

---

## 四、重构原则

1. **不改 wedlm 核心引擎**：`wedlm/`、`hf_compat/` 保持不动，仅引用其接口。
2. **向后兼容**：旧 `dpo/` 目录在重构期间保留，新增 `posttraining/` 逐步接管。
3. **配置分层**：`base.py` 含共用参数 → `sft.py` / `dpo.py` / `gspo.py` 各自扩展，避免单一 80 字段 dataclass。
4. **依赖注入**：Trainer 不直接 import config 全局实例，通过构造函数注入。
5. **单一职责**：每个文件不超过 300 行；loss 函数只做数学计算，不涉及 batch 组装。
6. **测试优先**：先迁 smoke tests，再补 unit tests。
7. **类型注解**：所有公开函数使用 typing 标注参数和返回值。
8. **代码保留**：不对任何旧代码进行改动，只将重构后的代码保存至新目录下。
---

## 五、风险与注意事项

| 风险 | 缓解措施 |
|------|---------|
| `WeDLMPackedDataset` 两处实现的微妙差异导致行为变化 | Phase 1.4 中对比 diff，保留 dpo 版本为主，合入 finetune 独有特性（如 `WeDLMShuffledPackedDataset`） |
| MagiAttention 安装依赖可能导致 smoke tests 失败 | smoke tests 优先使用 `dense` backend |
| DeepSpeed 配置兼容性 | 保持 `config/base.py` 中 deepspeed 字段完整不变 |

---

## 六、迁移对照表

| 旧路径 | 新路径 |
|--------|--------|
| `dpo/src/config.py` → | `wedlm_train/config/{base,sft,dpo,gspo}.py` |
| `dpo/src/data.py` → | `wedlm_train/data/{packed,pairwise,prompt}_dataset.py + collate.py` |
| `dpo/src/trainer.py` → | `wedlm_train/trainer/{base,sft,dpo,gspo}_trainer.py` |
| `dpo/src/loss.py` → | `wedlm_train/loss/{mlm,ar,block_scores,dpo,gspo}.py` |
| `dpo/src/batch.py` → | `wedlm_train/batch/wedlm_batch.py` |
| `dpo/src/masking.py` → | `wedlm_train/batch/masking.py` |
| `dpo/src/model.py` → | `wedlm_train/model/forward.py + rotary.py` |
| `dpo/src/attention.py` → | `wedlm_train/attention/{dense,magi,registry}.py` |
| `dpo/src/reward.py` → | `wedlm_train/reward/{base,margin,deepmath,callable,clipped,registry}.py` |
| `dpo/train.py` → | `scripts/train.py` |
| `dpo/scripts/*.py` → | `scripts/{smoke_tests,analysis,download}/` |
| `dpo/configs/*.yaml` → | `configs/` |
---

## 七、命名规范

- 类名：`WeDLM` 前缀保留（如 `WeDLMPackedDataset`），新增类保持一致性
- 函数名：`snake_case`，loss 函数以 `compute_` 开头（如 `compute_dpo_loss`）
- 配置字段：原样保留 yaml 字段名（如 `gspo_group_size`），不做重命名
- 文件命名：`snake_case`（如 `packed_dataset.py`）
- 模块 `__init__.py` 使用 `__all__` 显式控制导出
