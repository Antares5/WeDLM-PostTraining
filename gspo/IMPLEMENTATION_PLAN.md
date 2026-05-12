# GSPO On-Policy Training for WeDLM Block Diffusion Language Model

## 实施文档

---

## 目录

1. [概述与目标](#1-概述与目标)
2. [理论基础](#2-理论基础)
3. [目录结构设计](#3-目录结构设计)
4. [组件实现清单](#4-组件实现清单)
5. [数据流设计](#5-数据流设计)
6. [配置设计](#6-配置设计)
7. [训练循环设计](#7-训练循环设计)
8. [内存优化策略](#8-内存优化策略)
9. [Math Reward 设计](#9-math-reward-设计)
10. [测试与验证](#10-测试与验证)
11. [分阶段实施路线](#11-分阶段实施路线)

---

## 1. 概述与目标

### 1.1 目标

在 WeDLM block diffusion language model 上实现 GSPO (Group Sampling Policy Optimization) 的 on-policy 训练。使模型能够在训练过程中自行采样多个 response，通过 math rule-based reward 打分后进行组内对比优化。

### 1.2 核心特征

- **On-policy 采样**：每个训练步从当前 policy 模型采样 K 个 response
- **Group-wise 对比**：best response vs 其余 responses 的 log-mean-exp
- **Block diffusion 适配**：使用 block-level pseudo-log-likelihood 作为概率的代理
- **低内存设计**：延续现有 DPO 的逐分支 backward 策略
- **自包含**：所有代码复制到 gspo 目录，不从 dpo/finetune 导入

### 1.3 与现有 DPO 的关系

现有的 `dpo/` 目录实现了 **offline block-wise DPO**，使用预先收集的 pairwise 数据（chosen/rejected）。GSPO 在此基础上增加了：
- **在线生成**：训练过程中由模型自身生成 response
- **多路采样**：K > 2 个 response 进行组内比较
- **Reward signal**：Phase 1 使用 math rule-based binary reward（无需独立 RM 模型）

### 1.4 关键架构决策（2026-05-11 确认）

以下决策已与实施者确认，作为代码编写的依据：

**决策 1 — 生成引擎**：
- ✅ **复用 `wedlm/engine/LLMEngine`**（项目内部推理引擎），不从零实现 WeDLM 迭代解码
- `gspo/src/generator.py` 作为薄封装层，构造 `SamplingParams` 并调用 `LLMEngine.add_request` / `step`
- 依赖关系：`gspo/` → `wedlm/` ✔️（允许），`gspo/` → `dpo/` ✖️（禁止），`gspo/` → `finetune/` ✖️（禁止）

**决策 2 — Reward 来源**：
- ✅ **Phase 1 不加载独立 Reward Model**，使用 math rule-based binary reward
- 训练数据为数学题，具有客观 ground truth（`solution`/`answer` 字段）
- Reward = 1.0（模型正确解答）或 0.0（错误/无法判断）
- 后续 Phase 3 可扩展为 RM 模型打分（通过 `gspo_reward_type` 切换）

**决策 3 — 训练流程**：
- ✅ **Phase 1 每步执行完整的 生成→打分→ref scoring→训练 流程**
- 暂不实现 Rollout Buffer（留到 Phase 2）
- 目标：先验证端到端流程正确性（K=2, 少量 prompt），再优化效率

**决策 4 — 内存布局**：
- ✅ **双模型常驻 GPU**：Policy Model + Ref Model 同时加载（各 ~16 GB）
- 无需 CPU↔GPU 切换，无独立 RM 模型
- 逐分支 backward 保证任意时刻只有 1 个 response 的激活在 GPU 上

---

## 2. 理论基础

### 2.1 核心挑战：Block Diffusion 没有精确的 log π(y|x)

WeDLM 是 block diffusion language model，其生成过程是**迭代去噪**：

1. 初始化窗口为 MASK tokens
2. 每步：前向传播 → 计算 entropy → 选低 entropy 位置填充 → 补充 MASK
3. 涉及随机采样、entropy threshold、temperature

而训练时的打分函数 `compute_block_scores` 使用的是**随机 mask 下的单步预测**，得到的是 ELBO 的下界估计 s_θ(y)，而非精确的 log π_θ(y|x)。

### 2.2 三条理论路径

#### 路径 A：Block Score 直接替代 Log-Probability（启发式方案）

**思想**：直接用 s_θ(y) 替代 log π_θ(y|x)，代入 GSPO loss。

**损失形式**：对每个 prompt x，生成 K 个 response {y_1,...,y_K}，RM 给出奖励 {R_1,...,R_K}。设 y_best 为奖励最高的 response：

L = -log σ( β · [ (s_θ(y_best) - s_old(y_best)) - log( (1/(K-1)) · Σ_{j≠best} exp(s_θ(y_j) - s_old(y_j)) ) ] )

**理论评估**：
- 优点：代码复用度最高，直接复用 `compute_block_scores` 和 `compute_dpo_loss` 的结构
- 缺点：s_θ 是 log π_θ 的有偏代理，重要性比率的校正性质不严格成立
- 实践判断：s_θ - s_old 仍是有效的"偏好变化"信号——若模型变得更偏好某个 response，该差值会上升。本质上是一种 block-level 的 contrastive representation learning

#### 路径 B：REINFORCE 策略梯度视角（严谨方案）

**思想**：策略梯度定理

∇_θ E_{y~π_θ}[R(y)] = E_{y~π_θ}[R(y) · ∇_θ log π_θ(y)]

利用 ∇_θ s_θ(y) ≈ ∇_θ log π_θ(y)（ELBO 性质）:

∇_θ E[R] ≈ -E_{y~π_θ}[R(y) · ∇_θ s_θ(y)]

等价于最小化加权 block score:

L = -Σ_i A_i · s_θ(y_i),  其中 A_i = R(y_i) - (1/K) Σ_j R(y_j)

**理论评估**：
- 优点：不要求 s_θ 是精确概率，只要求 ∇s_θ ≈ ∇log π_θ，在 block diffusion 框架下是最理论正确的 RL 形式
- 缺点：REINFORCE 高方差问题，需要 group baseline 缓解；缺少 KL 约束可能 reward hacking

#### 路径 C：混合方案（推荐）

**思想**：结合路径 A 的 ratio 约束和路径 B 的梯度结构。

L = -Σ_i stop_grad(w_i) · s_θ(y_i) + α · KL(s_θ || s_old)

其中 w_i = softmax(R(y_i)/τ) · s_θ(y_i) / s_old(y_i)（重要性采样修正）。

**Phase 1 采用路径 A，Phase 2 升级到路径 C。**

### 2.3 收敛性质分析

**On-policy 数据分布漂移**：随着训练进行，π_θ 不断变化，采样的 response 分布也随之改变。Group baseline 能在一定程度上适应这种漂移。

**Block diffusion 的额外不稳定性**：
- 随机 mask ratio 导致 s_θ(y) 本身就是随机变量（即使 y 固定）
- 通过 `dpo_num_mask_samples` 多次采样取平均来降低方差
- 同一批次的 chosen/rejected 使用相同的 mask pattern 可提高信噪比

**KL 约束的必要性**：无 KL 约束时，模型可能学会利用 block scoring 的盲区（即 s_θ 高但实际 log π_θ 不高的区域）进行 reward hacking。因此 s_old 的 reference score 作为 anchor 是必要的。

---

## 3. 目录结构设计

### 3.1 目标结构

```
gspo/
├── IMPLEMENTATION_PLAN.md          # 本文档
├── train.py                         # 训练入口（从 dpo/train.py 复制并修改）
├── configs/
│   └── example.yaml                 # 示例配置（从 dpo/configs/example.yaml 复制并修改）
├── data/
│   └── (prompt 数据文件，不含 label)
├── src/
│   ├── __init__.py                  # 模块导出
│   ├── config.py                    # GSPO 训练配置（从 dpo/src/config.py 复制并扩展）
│   ├── data.py                      # Prompt dataset + collate（从 dpo/src/data.py 复制并改造）
│   ├── batch.py                     # WeDLM batch 构建（从 dpo/src/batch.py 复制）
│   ├── masking.py                   # Mask 采样与重排（从 dpo/src/masking.py 复制）
│   ├── model.py                     # 前向传播（从 dpo/src/model.py 复制）
│   ├── attention.py                 # 注意力后端（从 dpo/src/attention.py 复制）
│   ├── loss.py                      # 损失函数（从 dpo/src/loss.py 复制并新增 GSPO loss）
│   ├── trainer.py                   # 训练器（从 dpo/src/trainer.py 复制并重写训练步）
│   ├── generator.py                 # 【新增】On-policy 生成引擎（封装 wedlm.engine.LLMEngine）
│   └── reward.py                    # 【新增】Math reward 计算（rule-based answer verification）
├── scripts/
│   ├── smoke_test_gspo_data.py      # 数据加载测试
│   ├── smoke_test_gspo_gen.py       # 生成流程测试
│   └── smoke_test_gspo_loss.py      # 损失函数测试
└── (train-deepmath-*.parquet)       # 已有数据文件
```

### 3.2 文件来源标注

| 目标文件 | 来源 | 改动幅度 |
|:---------|:-----|:---------|
| `src/batch.py` | `dpo/src/batch.py` | **直接复制**，无需改动 |
| `src/masking.py` | `dpo/src/masking.py` | **直接复制**，无需改动 |
| `src/model.py` | `dpo/src/model.py` | **直接复制**，无需改动 |
| `src/attention.py` | `dpo/src/attention.py` | **直接复制**，无需改动 |
| `src/config.py` | `dpo/src/config.py` | 中等改动（新增 GSPO 配置项） |
| `src/data.py` | `dpo/src/data.py` | 大改动（改造为 prompt dataset） |
| `src/loss.py` | `dpo/src/loss.py` | 中等改动（新增 group-wise loss） |
| `src/trainer.py` | `dpo/src/trainer.py` | 大改动（重写 train_step_gspo） |
| `src/generator.py` | 无 | **全新文件**（封装 `wedlm.engine.LLMEngine`） |
| `src/reward.py` | 无 | **全新文件**（rule-based math verification） |
| `train.py` | `dpo/train.py` | 少量改动（入口 + CLI 参数） |

### 3.3 核心原则

- **绝对不 import dpo 或 finetune 目录下的任何模块**
- 所有被依赖的通用代码直接复制到 gspo/src/ 下
- 这样即使 dpo 目录被修改或删除，gspo 依然可独立运行

---

## 4. 组件实现清单

### 4.1 文件：`src/batch.py` — WeDLM Batch 构建（复制）

**来源**：`dpo/src/batch.py`

**改动**：无。直接复制全部内容。

**职责**：
- 定义 `WeDLMBatch` dataclass，包含 packed_input_ids、masked_indices、p_mask、attention masks 等
- 实现 `build_wedlm_batch` 函数，完成 packed tokens → 双流重排 + 随机 mask + attention plan 构建

---

### 4.2 文件：`src/masking.py` — Mask 采样与重排（复制）

**来源**：`dpo/src/masking.py`

**改动**：无。直接复制全部内容。

**职责**：
- `sample_block_mask_ratios`：为每个 block 采样随机 mask ratio p ~ U(0,1)
- `sample_mask_indices`：在可 mask 位置上按比例随机选中 mask 位置
- `reorder_block`：重排为 unmasked tokens 在前、masked tokens 在后
- `build_2d_attention_mask`：构建 dense 后端的 2D attention mask
- `build_magi_plan`：构建 Magi 后端的 attention plan

---

### 4.3 文件：`src/model.py` — 前向传播（复制）

**来源**：`dpo/src/model.py`

**改动**：无。直接复制全部内容。

**职责**：
- `apply_rotary_pos_emb`：RoPE 位置编码
- `wedlm_attention_forward`：单层 attention 前向（含 mask/plan 路由）
- `wedlm_forward`：完整模型前向，返回 logits

---

### 4.4 文件：`src/attention.py` — 注意力后端（复制）

**来源**：`dpo/src/attention.py`

**改动**：无。直接复制全部内容。

**职责**：
- `check_backend_available`：检查后端是否可用
- `get_available_backend`：自动选择可用后端（magi > dense）
- `MagiAttentionWrapper`：Magi flex flash attention 封装
- `DenseAttentionWrapper`：PyTorch SDPA dense attention 封装
- `get_attention_wrapper`：工厂函数返回对应 wrapper

---

### 4.5 文件：`src/config.py` — GSPO 训练配置（复制+扩展）

**来源**：`dpo/src/config.py`

**改动描述**：

**保留的配置项**（来自 dpo/config.py）：
- model_path, trust_remote_code, max_seq_length, block_size
- mask_per_block, loss_weighting_scheme, mask_eps, num_learnable_im_end
- enable_ar_loss, ar_loss_weight, attention_backend
- output_dir, num_train_epochs, per_device_train_batch_size, gradient_accumulation_steps
- learning_rate, lr_scheduler_type, warmup_ratio, weight_decay, max_grad_norm
- rebuild_cache, use_deepspeed + 所有 deepspeed_* 配置
- logging_steps, save_steps, save_total_limit, bf16, seed
- use_wandb + wandb_* 配置

**移除的配置项**：
- training_mode（固定为 gspo，不需要）
- dpo_train_data（GSPO 只有 prompt 数据）
- dpo_ref_model_path（改为 ref 路径在 GSPO 专属字段）
- dpo_beta, dpo_length_norm, dpo_block_reduce, dpo_seq_reduce, dpo_num_mask_samples（改为 GSPO 前缀字段）

**新增的 GSPO 专属配置项**：
- **gspo_prompt_data**：prompt 数据路径（仅含 prompt，无需 label 的 jsonl）
- **gspo_num_samples**：每个 prompt 采样的 response 数量 K（默认 4）
- **gspo_beta**：GSPO loss 的温度系数（默认 0.1）
- **gspo_block_reduce**：block 内 score reduce 方式（默认 "mean"）
- **gspo_seq_reduce**：sequence 内 score reduce 方式（默认 "mean"）
- **gspo_num_mask_samples**：score 估计的 MC 采样数（默认 1，on-policy 场景建议 1~2 以控制内存）
- **gspo_length_norm**：是否启用长度归一化（默认 true）
- **gspo_ref_model_path**：reference model 路径（用于 s_old 计算，null 表示用初始化模型）
- **gspo_reward_type**：reward 类型（Phase 1: "math_verify"，Phase 3: "model"）
- **gspo_reward_model_path**：reward model 路径（仅 gspo_reward_type="model" 时使用，Phase 3）
- **gspo_generation_config**：生成参数嵌套配置，包含：
  - `max_new_tokens`：最大生成 token 数（默认 512）
  - `temperature`：采样温度（默认 1.0）
  - `top_p`：nucleus sampling 参数（默认 1.0）
  - `top_k`：top-k sampling 参数（默认 0，即不使用）
  - `wedlm_entropy_threshold`：WeDLM 并行解码 entropy 阈值（默认 0.4）
  - `wedlm_pos_penalty_factor`：位置惩罚因子（默认 0.02）
  - `wedlm_window_size`：解码窗口大小（默认 256）

**新增方法**：
- `get_generation_config`：返回生成参数的字典

---

### 4.6 文件：`src/data.py` — Prompt Dataset + Collate（复制+改造）

**来源**：`dpo/src/data.py`

**保留内容**：
- `get_im_end_token_id` 辅助函数（直接复制）
- `DEFAULT_IM_END_TOKEN_ID` 常量

**新增类**：`GSPOPromptDataset`

**职责**：
- 从 JSONL/Parquet 文件加载 prompt 数据
- 每行是一个标准的 messages 列表（与 `dpo/src/data.py` 的 `WeDLMPackedDataset` 相同的消息格式）
- 不做 SFT 相关的 label 构建（无需区分 prompt 和 response 部分的 loss mask）
- tokenize + 截断后存储 input_ids（不做 packing，每个样本独立）
- 额外提取 `ground_truth` 字段（来自数据的 `solution`/`answer` 字段），用于 math reward 计算
- 返回格式：`{"input_ids": tensor, "attention_mask": tensor, "messages": list, "ground_truth": str}`

**支持的数据格式**：
- JSONL 每行是一个 JSON 对象：`{"messages": [{"role": "user", "content": "..."}, ...], "solution": "42"}`
- Parquet 格式（如 DeepMath）：列包含 prompt 消息和 `solution`/`answer` 列
- 支持 system/user/assistant 角色的消息
- 使用 `tokenizer.apply_chat_template` 构建文本后 encode
- `ground_truth` 从 `solution` 或 `answer` 字段提取（优先级：`solution` > `answer`）

**新增类**：`GSPOCollateFunction`

**职责**：
- 作为 DataLoader 的 collate_fn
- 对变长 input_ids 做 padding（左侧 padding 用于生成）
- 返回 `{"input_ids": [bs, max_len], "attention_mask": [bs, max_len], "messages": list}`

**DataLoader 配置**：
- batch_size 由 `per_device_train_batch_size` 控制（在 config 中）
- shuffle=True，无 DistributedSampler（每个 GPU 独立采样不同 prompt）

---

### 4.7 文件：`src/loss.py` — 损失函数（复制+新增）

**来源**：`dpo/src/loss.py`

**保留的函数**（直接复制）：
- `compute_masked_token_logps`：单 token log-prob 计算
- `compute_block_scores`：block-level sequence score 计算（核心打分函数）
- `compute_mlm_loss`：MLM 训练 loss
- `compute_ar_loss`：自回归训练 loss

**移除的函数**：
- `compute_dpo_loss`：GSPO 不需要 pairwise DPO loss，但其结构可作为参考

**新增函数**：`compute_gspo_loss`

**职责**：
- 输入：policy_scores [K], reference_scores [K], rewards [K], beta
- 逻辑：
  1. 找到 reward 最高的 index（best_idx）
  2. 计算 pi_diff = policy_scores - reference_scores，shape [K]
  3. best = pi_diff[best_idx]
  4. others = pi_diff[非 best_idx 的所有 index]，shape [K-1]
  5. others_logmeanexp = logsumexp(others) - log(K-1)
  6. logits = beta * (best - others_logmeanexp)
  7. loss = -logsigmoid(logits)
- 返回：loss（标量）和 logs（字典，包含 rewards/chosen、rewards/rejected、rewards/margin、rewards/accuracy、logits 等指标）

**额外说明**：
- 支持 K=0（空 batch）的安全处理，返回 zero loss
- 当 K=1 时，others 为空，退化为仅使用 best 的 loss（此时效果不佳，但不应 crash）
- 与 `compute_dpo_loss` 的输出格式一致，便于复用 trainer 中的 logging 逻辑

---

### 4.8 Block Scoring（集成在 trainer 中）

**设计决策**：Block scoring 逻辑**不抽取为独立文件**，而是作为 `trainer.py` 中的辅助方法。

原因：
- `build_wedlm_batch` + `wedlm_forward` + `compute_block_scores` 的调用模式在 DPO trainer 中已有成熟实现
- GSPO 的 scoring 需求与 DPO 高度一致（构建 WeDLMBatch → forward → compute_block_scores）
- 直接复用 `trainer.py` 中的 `_forward_wedlm_logits` 和 `_compute_block_scores_for_batch` 辅助方法即可
- 减少文件数量和模块间耦合

**trainer 中的相关辅助方法**：
- `_forward_wedlm_logits(model, batch)` → 返回 logits
- `_compute_block_scores_for_batch(logits, batch)` → 返回 sequence scores
- 以上方法直接从 `dpo/src/trainer.py` 复制并适配 GSPO 配置参数名

**用于 GSPO 的新增辅助方法**：
- `_score_responses(model, responses, prompt_labels)` → 对 K 个 response 批量 build batch + forward + score，返回 [K] scores
- `_score_responses_no_grad(model, ...)` → 同上但全程 no_grad（用于 ref model 和 policy no-grad pass）

---

### 4.9 文件：`src/generator.py` — On-Policy 生成引擎（全新）

**职责**：在训练循环中使用当前 policy 模型生成 K 个 response。

**核心设计决策**：**复用 `wedlm/engine/LLMEngine`**（`wedlm/` 目录下的推理引擎），而非从零实现 WeDLM 迭代解码。原因：
- `wedlm/engine/` 已有成熟的 `LLMEngine`、`Scheduler`、`ModelRunner`、`Sequence` 实现
- `wedlm/sampling_params.py` 已有 `SamplingParams` dataclass，字段与 GSPO 所需的生成参数高度一致（`temperature`, `top_p`, `top_k`, `max_tokens`, `wedlm_entropy_threshold`, `wedlm_pos_penalty_factor`）
- 避免重复实现复杂的 WeDLM 并行解码逻辑（entropy threshold、位置惩罚、window refill 等）

> **依赖说明**：`gspo/` 可以依赖 `wedlm/`（这是项目内部的推理引擎），只是不依赖 `dpo/` 和 `finetune/`。

**核心类**：`WeDLMGenerator`（薄封装层）

**初始化参数**：
- `model`：当前 policy 模型（已 unwrap，即 `accelerator.unwrap_model(model)` 的结果）
- `tokenizer`：用于 encode/decode
- `generation_config`：字典，包含 `max_new_tokens`, `temperature`, `top_p`, `top_k`, `wedlm_entropy_threshold`, `wedlm_pos_penalty_factor`
- `device`：计算设备

**内部实现**：
1. 使用 `wedlm.engine.LLMEngine`（传入 `model` 作为初始化参数）
2. 构建 `SamplingParams` 对象，字段映射：
   - `max_tokens` ← `generation_config.max_new_tokens`
   - `temperature` ← `generation_config.temperature`
   - `top_p` ← `generation_config.top_p`
   - `top_k` ← `generation_config.top_k`
   - `wedlm_entropy_threshold` ← `generation_config.wedlm_entropy_threshold`
   - `wedlm_pos_penalty_factor` ← `generation_config.wedlm_pos_penalty_factor`
3. 调用 `engine.add_request(prompt_text, sampling_params)` 提交生成请求
4. 调用 `engine.step()` 循环进行迭代解码
5. 收集生成的 token ids 和 text

**核心方法**：`generate_single`

**输入**：`prompt_text: str`（单个 prompt 的文本）
**输出**：`{"input_ids": tensor [L_prompt+L_response], "labels": tensor (prompt部分=-100), "text": str}`

**内部流程**：
1. 将 prompt_text tokenize → `prompt_ids`
2. 通过 `LLMEngine` 迭代生成 response tokens
3. 将 `prompt_ids + response_ids` 拼接
4. 构建 labels：prompt 部分全为 -100，response 部分为真实 token id
5. 返回结果字典

**核心方法**：`generate_batch`

**输入**：`prompts: List[str]`，K: int（每个 prompt 生成的 response 数量）
**输出**：`List[List[Dict]]`，形状为 `[batch_size, K]`，每个 dict 同上

**实现要点**：
- 对每个 prompt，使用不同的 random seed（`base_seed + k`）初始化 `LLMEngine` 的采样状态，保证 K 个 response 的多样性
- 当前版本逐个 prompt 串行生成（一个 prompt 的 K 个 response 可以并行提交给 engine）
- 所有生成操作在 `torch.no_grad()` 下执行
- 使用 `model.eval()` 模式
- 生成前保存模型的随机状态，生成后恢复

**关于 `wedlm_window_size`**：
- `LLMEngine` 内部通过 `Config` 管理 window/block 大小，不需要在 generator 层额外指定
- 如果需要在训练时调整 window 行为，通过 `LLMEngine` 构造参数传入

---

### 4.10 文件：`src/reward.py` — Math Reward 计算（全新）

**设计决策**：**Phase 1 不使用独立 Reward Model**。训练数据以数学题为主（GSM8K、MATH 等），奖励信号来源于模型生成的 response 是否包含正确答案。Reward 为 binary 值：1.0（正确）或 0.0（错误）。

> **原因**：
> - 无需额外的 RM 模型加载，大幅降低内存压力和实现复杂度
> - 数学题有客观的 ground truth，binary reward 信号明确、无歧义
> - 后续 Phase 3 可扩展为 RM 模型打分（通过 `gspo_reward_type` 切换）

**核心类**：`MathReward`

**初始化参数**：
- `reward_type: str`：奖励类型，Phase 1 固定为 `"math_verify"`
- `tokenizer`：tokenizer（用于可能的 answer extraction）

**核心方法**：`compute_rewards`

**输入**：
- `prompts: List[str]`：prompt 文本列表
- `responses: List[str]`：response 文本列表（与 prompts 一一对应）
- `ground_truths: List[str]`：正确答案列表（从 prompt 数据的 `solution` 或 `answer` 字段提取）

**输出**：`rewards: torch.Tensor [K]`，每个元素为 1.0 或 0.0

**内部流程**：
1. 对每个 (response, ground_truth) 对：
   a. 调用 `extract_answer(response)` 从生成文本中提取最终答案
   b. 调用 `verify_answer(extracted, ground_truth)` 进行答案比对
   c. 正确 → 1.0，错误 → 0.0
2. 返回 `[K]` 的 float tensor

**答案提取策略**（`extract_answer`）：
- 支持 GSM8K 风格：提取 `#### <number>` 之后的数字
- 支持 MATH 风格：提取 `\boxed{...}` 中的内容
- 支持通用格式：提取最后一行中的数字/表达式
- 参考现有 `evaluation/evaluators/` 目录中的 evaluator 实现（如 `gsm8k_evaluator.py`、`math_evaluator.py`）

**答案比对策略**（`verify_answer`）：
- 数值型答案：使用数值容差比较（相对误差 < 1e-3 或绝对误差 < 1e-5）
- 表达式型答案：SymPy 等价性检查（如 MATH 数据集）
- 多选型答案：精确字符串匹配（如 ARC 数据集）

**辅助方法**：`compute_batch_rewards`

**输入**：`prompts [bs]`，每个 prompt 对应 `[K]` 个 responses，以及对应的 `[K]` 个 ground_truths（同一个 prompt 的 K 个 response 共享同一个 ground_truth）
**输出**：`rewards [bs, K]` tensor

**数据格式要求**：
- Prompt 数据（JSONL 或 Parquet）每行必须包含 `solution` 或 `answer` 字段作为 ground truth
- 示例：`{"messages": [...], "solution": "42", "answer_type": "numeric"}`

**边界情况处理**：
- 无法从 response 中提取答案 → reward = 0.0
- ground_truth 为空或缺失 → reward = 0.0（保守处理）
- response 为空字符串 → reward = 0.0

---

### 4.11 文件：`src/trainer.py` — GSPO 训练器（复制+重写）

**来源**：`dpo/src/trainer.py`

**保留内容**：
- `_init_wandb`：wandb 初始化（直接复制）
- `WeDLMTrainer.__init__`：基础初始化逻辑（保留 model + tokenizer 加载部分）
- `_setup`：model + tokenizer + dataset + dataloader 初始化（改造）
- `_prepare_training`：optimizer + scheduler + accelerator 准备（直接复制）
- `train`：主训练循环（保留框架，step 内部 dispatch 改掉）
- `_log_metrics`, `_save_checkpoint`：日志和保存（直接复制）
- `_compute_ar_loss`：AR loss 辅助计算（保留，用于可选的 AR loss）
- `_forward_wedlm_logits`：前向辅助函数（保留）

**改造的 `_setup` 方法**：
- 不再加载 `WeDLMPackedDataset` 或 `WeDLMPairwiseDataset`
- 改为加载 `GSPOPromptDataset`（来自 `src/data.py`）
- DataLoader 使用 `GSPOCollateFunction`
- 加载 ref model（用于计算 s_old，ref model 保持在 GPU 上）
- 初始化 `WeDLMGenerator`（wrapper around `wedlm.engine.LLMEngine`）
- 初始化 `MathReward`（无需模型加载，纯规则匹配）
- 无需初始化 `Scorer`（scoring 逻辑直接在 trainer 中通过 `build_wedlm_batch` + `compute_block_scores` 完成）

**新增方法**：`train_step_gspo`

**输入**：batch（dict，由 GSPOPromptDataset + collate 返回，包含 `input_ids`, `attention_mask`, `messages`, `ground_truths`）

**流程概要**：
1. **生成阶段**（no_grad）：
   - 对 batch 中每个 prompt，调用 `generator.generate_single` K 次（通过 `LLMEngine`），得到 K 个 (input_ids, labels, text)
   - 使用不同的 random seed 保证 K 个 response 的多样性
2. **Math Reward 打分阶段**（no_grad）：
   - 调用 `math_reward.compute_rewards(prompts_text, responses_text, ground_truths)`
   - 得到 K 个 binary reward（1.0 或 0.0）
   - 归一化 rewards（可选，减均值除标准差，对 binary reward 仍有意义）
3. **Reference scoring 阶段**（no_grad）：
   - 对 K 个 response 分别构建 WeDLMBatch
   - 使用 ref model 做前向 + `compute_block_scores`，得到 [K] 个 reference scores
4. **Policy scoring + Gradient 阶段**：
   - 采用多轮逐分支 backward 策略（详见内存优化章节）
   - 先做 K 次 no_grad forward 获取所有 policy scores（计算 GSPO loss 系数）
   - 再逐 response backward：每个 response forward → compute_block_scores → 乘以系数 → backward → del
5. **日志收集**：
   - 收集 loss、rewards、scores、margins、K 个 response 中正确数等指标

**与现有 `train_step_dpo` 的关键区别**：
- 不再有固定的 chosen/rejected pairs
- 每次都要做 K 次生成（耗时）
- reference scores 是对 K 个 response 分别计算的
- loss 是 group-wise 而非 pairwise
- 使用 policy model 自身的生成结果（true on-policy）而非离线数据

**训练循环改造**：
- 在 `train()` 方法中，生成阶段不是每个 step 都做——引入 `gen_every_n_steps` 配置项
- 每 N 步重新生成一次 response，中间的 step 复用上一次生成的 responses（类似 PPO 的 rollout buffer）
- 这样可以大幅降低生成开销（生成通常比训练慢很多）

---

### 4.12 文件：`train.py` — 训练入口（复制+修改）

**来源**：`dpo/train.py`

**改动**：
- 移除 `--training_mode` CLI 参数
- 移除 DPO 专属 CLI 参数（`--dpo_train_data`, `--dpo_ref_model_path` 等）
- 新增 GSPO 专属 CLI 参数：
  - `--gspo_prompt_data`：prompt 数据路径
  - `--gspo_num_samples`：每 prompt 采样数 K
  - `--gspo_beta`：GSPO beta
  - `--gspo_ref_model_path`：ref 模型路径
  - `--gspo_reward_model_path`：reward 模型路径
  - `--gen_max_new_tokens`：最大生成 token 数
  - `--gen_temperature`：生成温度
- 导入来自 `gspo.src` 而非 `dpo.src`

---

### 4.13 文件：`configs/example.yaml` — 示例配置

**来源**：`dpo/configs/example.yaml`

**改动**：
- 使用 GSPO 专属字段
- 新增生成参数、math reward 参数
- 移除 training_mode 和 DPO 专属字段

---

## 5. 数据流设计

### 5.1 单步训练数据流

```
Prompt JSONL
    │
    ▼
GSPOPromptDataset.__getitem__
    │  返回: input_ids, attention_mask, messages, ground_truth
    ▼
DataLoader (GSPOCollateFunction)
    │  返回: padded batch {input_ids, attention_mask}
    ▼
train_step_gspo (trainer.py)
    │
    ├─► [1] WeDLMGenerator.generate_batch (no_grad)
    │       每个 prompt → K 个 response
    │       输出: [{input_ids, labels, text}, ...] × K
    │
    ├─► [2] MathReward.compute_rewards (no_grad)
    │       K 个 (prompt, response, ground_truth) → K 个 binary reward
    │       输出: rewards [K], 值为 1.0 或 0.0
    │
    ├─► [3] Ref Model Scoring (no_grad)
    │       对每个 response:
    │         tokenize → WeDLMBatch → wedlm_forward → compute_block_scores
    │       输出: reference_scores [K]
    │
    ├─► [4] Policy Model Scoring + GSPO Loss (grad)
    │       对每个 response 逐分支 backward:
    │         tokenize → WeDLMBatch → wedlm_forward → compute_block_scores
    │         → 计算该分支 loss → backward → del
    │       输出: loss, logs
    │
    ▼
optimizer.step() → lr_scheduler.step() → zero_grad()
```

### 5.2 生成复用（Rollout Buffer）策略

由于 WeDLM 生成比 AR 模型更慢（需要多步迭代去噪），在每个 step 都重新生成是不现实的。引入 **Rollout Buffer** 模式：

1. 维护一个 buffer，存储最近的 N_gen 个 prompt 及其 K 个 responses + rewards
2. 每 `gen_every_n_steps` 步，从 dataloader 取一个新 batch 的 prompts：
   - 生成 K 个 responses
   - 打分
   - 存入 buffer
3. 在非生成 step 中，从 buffer 随机采样一个 prompt 的 responses 进行训练
4. Buffer 大小 = gen_every_n_steps（保证不会 data starvation）

**Buffer 结构**：
- 每个 entry：`{prompt_input_ids, prompt_labels, prompts_text, responses: [{input_ids, labels, text}] × K, rewards: [K], ref_scores: [K]}`
- 多维护 ref_scores 是因为：如果 gen_every_n_steps > 1，ref scores 可以在生成时一次性计算并缓存（ref model 不变）

---

## 6. 配置设计

### 6.1 完整配置项清单

```yaml
# ========== 模型 ==========
model_path: "tencent/WeDLM-8B-Base"
trust_remote_code: true

# ========== 数据 ==========
gspo_prompt_data: "data/prompts.jsonl"    # prompt 数据（含 ground_truth）
gspo_prompt_format: "messages"             # "messages"（标准 chat format）或 "deepmath"（DeepMath parquet）
max_seq_length: 2048

# ========== GSPO 核心 ==========
gspo_num_samples: 4                       # K: 每 prompt 采样数
gspo_beta: 0.1                            # GSPO 温度系数
gspo_block_reduce: "mean"                 # block 内 reduce: mean/sum
gspo_seq_reduce: "mean"                   # sequence 内 reduce: mean/sum
gspo_num_mask_samples: 1                  # score 估计 MC 采样数（on-policy 建议 1）
gspo_length_norm: true                    # 是否启用长度归一化
gspo_ref_model_path: null                 # null=用 model_path 加载 ref

# ========== Math Reward（Phase 1：rule-based）==========
gspo_reward_type: "math_verify"           # "math_verify"（Phase 1）/ "model"（Phase 3）
# gspo_reward_model_path: null            # 仅 reward_type="model" 时使用（Phase 3）

# ========== 生成参数 ==========
gen_max_new_tokens: 512
gen_temperature: 1.0
gen_top_p: 1.0
gen_top_k: 0
gen_wedlm_entropy_threshold: 0.4
gen_wedlm_pos_penalty_factor: 0.02

# ========== WeDLM 结构 ==========
block_size: 32
mask_per_block: true
loss_weighting_scheme: "weighted"
mask_eps: 1.0e-8
num_learnable_im_end: 0

# ========== AR loss（可选） ==========
enable_ar_loss: false                     # on-policy 阶段通常关掉
ar_loss_weight: 1.0

# ========== 注意力 ==========
attention_backend: "magi"

# ========== 训练 ==========
output_dir: "outputs/gspo"
num_train_epochs: 1
per_device_train_batch_size: 1            # 每个 GPU 的 prompt 数
gradient_accumulation_steps: 8
learning_rate: 1.0e-6
lr_scheduler_type: "cosine"
warmup_ratio: 0.1
weight_decay: 0.01
max_grad_norm: 1.0

# ========== 缓存 ==========
rebuild_cache: false

# ========== DeepSpeed ==========
use_deepspeed: false
deepspeed_zero_stage: 2
deepspeed_offload_optimizer: false
# ... （其他 deepspeed 配置与 dpo 相同）

# ========== 日志 & 保存 ==========
logging_steps: 10
save_steps: 500
save_total_limit: 3

# ========== 设备 & 随机 ==========
bf16: true
seed: 42

# ========== WandB（可选） ==========
use_wandb: false
wandb_project: "wedlm-gspo"
wandb_team: null
wandb_group: null
wandb_host: null
wandb_key: null
```

---

## 7. 训练循环设计

### 7.1 主训练循环伪逻辑（Phase 1：每步完整流程）

Phase 1 不做 Rollout Buffer，**每个 training step 都执行完整的 生成→打分→ref scoring→训练 流程**。

```
for epoch in range(num_train_epochs):
    for step, batch in enumerate(train_dataloader):
        # === 阶段 0：从 batch 提取 prompt 和 ground_truth ===
        prompt_texts = batch["prompt_texts"]        # List[str]
        ground_truths = batch["ground_truths"]      # List[str]

        # === 阶段 1：生成（no_grad）===
        model.eval()
        with torch.no_grad():
            for each prompt:
                for k in range(K):  # 采样 K 个 response
                    response = generator.generate_single(prompt_text, seed=base_seed + k)
            # 结果: responses = [{input_ids, labels, text}, ...] × K
        model.train()

        # === 阶段 2：Math Reward 打分（no_grad）===
        rewards = math_reward.compute_rewards(prompt_text, [r["text"] for r in responses], ground_truth)
        # rewards: [K] tensor, 每个为 1.0 或 0.0

        # === 阶段 3：Ref Model Scoring（no_grad）===
        with torch.no_grad():
            for each response:
                wedlm_batch = build_wedlm_batch(response["input_ids"], response["labels"], ...)
                ref_logits = ref_model.forward(wedlm_batch)
                ref_scores[i] = compute_block_scores(ref_logits, ...)
            # ref_scores: [K] tensor

        # === 阶段 4：Policy Scoring + Backward（grad）===
        with accelerator.accumulate(model):
            # 4a. No-grad pass: 获取所有 K 个 policy scores（用于计算 GSPO 系数）
            with torch.no_grad():
                for each response i:
                    policy_logits_ng = model.forward(wedlm_batch_i)
                    policy_scores_ng[i] = compute_block_scores(policy_logits_ng, ...)
                    del policy_logits_ng

            # 4b. 计算 GSPO loss 和 per-response 系数
            best_idx = argmax(rewards)
            # pi_diff_i = policy_scores_ng[i] - ref_scores[i]
            loss_coeffs = compute_gspo_coefficients(policy_scores_ng, ref_scores, rewards, beta)
            # loss_coeffs: [K] tensor，每个元素是 ∂L/∂(policy_score_i) 的系数

            # 4c. 逐分支 backward（每个 response 独立 forward + backward）
            for each response i:
                policy_logits = model.forward(wedlm_batch_i)
                policy_score_i = compute_block_scores(policy_logits, ...)
                loss_i = loss_coeffs[i].detach() * policy_score_i
                accelerator.backward(loss_i)
                del policy_logits, policy_score_i, loss_i

            clip_grad_norm
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
```

**关键设计要点**：
- 阶段 4a 的 no-grad pass：一次性获取所有 K 个 policy score，用于计算 GSPO loss 中每个 response 的梯度系数
- 阶段 4c 的逐分支 backward：每个 response 独立 forward（构建计算图）+ backward 后立即释放，确保 GPU 内存中只有 1 个 response 的激活
- 这种"先算系数，再逐分支 backward"的策略与现有 DPO 的 `coeff_chosen`/`coeff_rejected` 设计一致，只是从 2 个分支扩展到 K 个分支

### 7.2 关键设计决策

**Q: Phase 1 为什么每步都生成？**
A: 简化实现，确保训练流程正确后再优化。虽然 WeDLM 生成比 AR 模型慢，但 Phase 1 的目标是**验证端到端流程**（K=2, 少量 prompt），而非吞吐。Phase 2 引入 Rollout Buffer 解决效率问题。

**Q: gen_every_n_steps 在 Phase 1 如何处理？**
A: Phase 1 暂不实现 `gen_every_n_steps`，每步都生成。Phase 2 再引入 Rollout Buffer 机制。

**Q: 生成时使用什么 sampling 策略？**
A: Phase 1 使用 temperature = 1.0 + top_p = 1.0（纯随机采样），保证 response 多样性。后续可调低 temperature 以聚焦 high-reward 区域。

**Q: 为什么不需要独立 RM 模型？**
A: 训练数据为数学题，有客观 ground truth。Binary reward（1/0）信号明确，无需训练 RM。这大幅降低了内存压力和实现复杂度。

---

## 8. 内存优化策略

### 8.1 内存预算分析（Phase 1：无独立 RM）

以 WeDLM-8B (bf16) 为例，Phase 1 **不加载独立 Reward Model**（使用 math rule-based reward）：

| 组件 | 估计内存 |
|:-----|:---------|
| Policy model (bf16) | ~16 GB |
| Ref model (bf16) | ~16 GB |
| 单个 response 的 activations | ~2-8 GB（取决于序列长度） |
| Optimizer states (AdamW, fp32) | ~32 GB |

**总计**：Policy + Ref + 1 个 response 的激活 + Optimizer ≈ **66-72 GB**。单张 A100-80GB 刚好可以容纳。

### 8.2 优化策略清单

**策略 1 — 双模型常驻 GPU**（Phase 1 推荐）：
- Policy model 和 Ref model 同时驻留在 GPU 上（各 ~16 GB）
- 不需要 CPU↔GPU 切换（避免 DeepSpeed ZeRO 下的复杂性）
- 如果显存不足（< 80GB），可将 Ref model offload 到 CPU

**策略 2 — 逐分支 backward（延续 DPO 设计）**：
- 对 K 个 response 逐一做 policy forward + backward
- 先做一轮 no-grad forward 获取所有 K 个 score（用于计算系数）
- 再逐 response backward，每个 backward 后立即 `del graph`
- 确保任意时刻 GPU 上只有 1 个 response 的激活

**策略 3 — Reference scores 在同一 no-grad 阶段计算**：
- Ref model 的 forward 在 no-grad 下一次性完成所有 K 个 response
- Scores 存入临时变量，训练阶段直接使用标量
- Ref model 不需要在 backward 阶段参与

**策略 4 — 使用 DeepSpeed ZeRO-3 时卸载优化器状态**：
- `deepspeed_offload_optimizer: true`
- `deepspeed_offload_param: true`（ZeRO-3 only）
- 可将 optimizer states 和参数卸载到 CPU/NVMe

**策略 5 — 降低 num_mask_samples**：
- GSPO on-policy 场景建议 `gspo_num_mask_samples = 1`
- 相比 DPO 的 `4` 大幅降低
- on-policy 场景下多次生成已经提供了足够的 exploration，score 估计不需要太多 MC 采样

**策略 6 — 使用 gradient checkpointing**：
- 在 model 上启用 gradient checkpointing
- 用计算换内存

### 8.3 推荐内存布局（每 GPU，Phase 1）

```
GPU (A100-80GB):
  ├── Policy Model (bf16)          ~16 GB  ← 常驻
  ├── Ref Model (bf16)             ~16 GB  ← 常驻
  ├── Optimizer States (fp32)      ~32 GB  ← 常驻
  └── 临时: 1 个 response 的激活   ~2-8 GB  ← 逐分支创建/释放
  ─────────────────────────────────────────
  总计:                              ~66-72 GB

CPU:
  └── (无大型模型，MathReward 为纯规则计算)
```

**Phase 2 升级**（如需引入独立 RM 模型时）：
- 将 Ref model offload 到 CPU，GPU 上只保留 Policy + RM
- 或使用多 GPU 分模型部署（Policy 在 GPU0，Ref+RM 在 GPU1）

---

## 9. Math Reward 设计

### 9.1 Phase 1 方案：Rule-based Binary Reward

**核心思想**：训练数据为数学题，每道题有明确的 ground truth 答案。生成的 response 若能正确解答，则 reward = 1.0；否则 reward = 0.0。

**不需要训练/加载任何 Reward Model**，reward 完全由规则匹配计算。

### 9.2 答案提取（`extract_answer`）

从 WeDLM 生成的 response 文本中提取最终答案。支持多种数学数据集的答案格式：

| 数据集 | 答案格式 | 提取策略 |
|:-------|:---------|:---------|
| GSM8K | `#### 42` | 正则匹配 `####\\s*(-?[\\d,.]+)` |
| MATH | `\\boxed{42}` | 正则匹配 `\\boxed{([^}]*)}`，取最后一个 |
| DeepMath | `The answer is 42` | 匹配 "answer is" 后的数值/表达式 |
| 通用数值 | 文本末尾数字 | 提取最后一行的最后一个数值 |

**实现参考**：可复用 `evaluation/evaluators/` 中已有的提取逻辑：
- `evaluation/evaluators/gsm8k_evaluator.py` — GSM8K 答案提取
- `evaluation/evaluators/math_evaluator.py` — MATH 答案提取

### 9.3 答案比对（`verify_answer`）

| 答案类型 | 比对策略 |
|:---------|:---------|
| 数值型 | 容差比较：相对误差 < 1e-3 或绝对误差 < 1e-5 |
| 表达式型 | SymPy `simplify(extracted - ground_truth) == 0` |
| 字符串型 | `extracted.strip().lower() == ground_truth.strip().lower()` |

### 9.4 Reward 归一化

在计算 GSPO loss 前，对 K 个 binary rewards 做归一化：
- 减均值、除标准差（在 K 个 response 内部做）
- 对于 binary reward，归一化后值为正（高于均值）或负（低于均值）
- 当所有 K 个 reward 相同（全 0 或全 1）时，标准差为 0，此时跳过归一化（rewards 全 0）

### 9.5 后续升级方案

- **Phase 3**：支持 `gspo_reward_type = "model"` 切换到独立 RM 模型打分（适用于无 ground truth 的通用偏好数据）
- **Phase 3**：支持 partial credit（非 binary reward），例如根据解题步骤的部分正确性给出 [0, 1] 之间的连续值
- **Phase 3**：支持 `gspo_reward_model_path` 配置外部 RM 模型路径

---

## 10. 测试与验证

### 10.1 Smoke Test 脚本

#### `scripts/smoke_test_gspo_data.py`

**测试目标**：验证 GSPOPromptDataset 能正确加载和 tokenize 数据。

**测试项**：
- 加载配置 + tokenizer + 数据集
- 检查返回的 input_ids shape 是否合理
- 检查 attention_mask 是否正确
- 检查 collate 后的 batch 格式
- 检查 DataLoader 迭代是否正常

#### `scripts/smoke_test_gspo_gen.py`

**测试目标**：验证 WeDLMGenerator 能正确生成 response。

**测试项**：
- 加载 model + tokenizer
- 用固定 prompt 生成 1 个 response
- 检查生成的 token 数量是否在合理范围
- 检查 response 解码后是否为合法文本
- 重复生成 K 次，检查 response 多样性（不应完全相同）
- 验证生成的 labels（prompt 部分应为 -100）

#### `scripts/smoke_test_gspo_loss.py`

**测试目标**：验证 GSPO loss 计算正确。

**测试项**：
- 用随机 scores 和 rewards 计算 loss
- 验证 loss 值在合理范围
- 验证梯度不为 None
- 边界情况：K=1, K=2, rewards 全部相同
- 与手动计算的 GSPO loss 进行数值对比

### 10.2 单元测试覆盖的边界情况

| 边界情况 | 预期行为 |
|:---------|:---------|
| K=0（空 batch） | 返回 zero loss，不 crash |
| K=1（只有一个 response） | 返回 zero loss（无法对比） |
| 所有 reward 相同 | loss 接近 -log σ(0) = log 2 ≈ 0.693 |
| response 长度为 0 | 跳过，不影响其他 response |
| prompt 长度为 0 | 跳过该 prompt |

### 10.3 集成测试

在单 GPU 上运行 1 个 epoch 的 mini 训练：
- 使用 10 个 prompt
- K=2
- 1 步生成 + 1 步训练
- 验证 loss 下降
- 验证 checkpoint 可正常保存和加载

---

## 11. 分阶段实施路线

### Phase 0：基础设施搭建（1-2 天）

**目标**：gspo 目录可运行，无功能但结构完整。

**任务清单**：
1. 从 `dpo/src/` 复制以下文件到 `gspo/src/`：
   - `batch.py`（直接复制）
   - `masking.py`（直接复制）
   - `model.py`（直接复制）
   - `attention.py`（直接复制）
   - `loss.py`（复制，后续增量修改）
   - `config.py`（复制，后续增量修改）
   - `data.py`（复制，后续增量修改）
   - `trainer.py`（复制，后续重写 train_step）
   - `__init__.py`（复制，调整 export）
2. 创建 `gspo/configs/` 目录和 `example.yaml`
3. 创建 `gspo/src/generator.py`（空壳）
4. 创建 `gspo/src/reward.py`（空壳）
5. 创建 `gspo/src/scorer.py`（空壳）
6. 创建 `gspo/scripts/` 目录和 smoke test 空壳
7. 运行 `python -c "from gspo.src.config import GSPOTrainingConfig"` 验证导入

### Phase 1：最小可行 GSPO（3-5 天）

**目标**：端到端可运行，K=2（退化为 pairwise on-policy），math reward，每步完整流程。

**任务清单**：
1. 实现 `src/config.py` 的 GSPO 配置项扩展（新增 `gspo_*`、`gen_*`、`gspo_reward_type` 等字段）
2. 实现 `src/data.py` 的 `GSPOPromptDataset` 和 `GSPOCollateFunction`（支持 messages 格式 JSONL + DeepMath parquet，提取 ground_truth）
3. 实现 `src/generator.py` 的 `WeDLMGenerator`（薄封装 `wedlm.engine.LLMEngine`，支持单 prompt 生成 K 个 response）
4. 实现 `src/reward.py` 的 `MathReward`（rule-based answer extraction + verification）
5. 实现 `src/loss.py` 的 `compute_gspo_loss`（group-wise GSPO loss + 系数计算函数 `compute_gspo_coefficients`）
6. 实现 `src/trainer.py` 的 `train_step_gspo`（完整四阶段流程：生成→打分→ref scoring→逐分支 backward）
7. 实现 `train.py` 入口（GSPO 专用 CLI，导入 `gspo.src`）
8. 编写 `scripts/smoke_test_gspo_data.py`（数据加载 + ground_truth 提取测试）
9. 编写 `scripts/smoke_test_gspo_gen.py`（LLMEngine 生成测试，K 个 response 多样性验证）
10. 编写 `scripts/smoke_test_gspo_loss.py`（GSPO loss 数值验证 + 梯度流验证）
11. 端到端单 GPU 测试（10 prompt, K=2, 5 step，验证 loss 下降和 checkpoint 保存）

### Phase 2：性能优化（2-3 天）

**目标**：扩展到 K=4~8，内存可控，速度可接受。

**任务清单**：
1. 实现 Rollout Buffer（`gen_every_n_steps` 机制）
2. 实现模型分时加载（ref/RM 的 GPU ↔ CPU 切换）
3. 实现逐分支 backward（K 个 response 逐一 backward）
4. 多 GPU 适配（DistributedSampler + 每 GPU 独立生成）
5. 实现生成阶段的 batch 处理（多个 prompt 同时生成，利用 GPU 并行）

### Phase 3：功能完善（2-3 天）

**目标**：生产就绪。

**任务清单**：
1. DeepSpeed ZeRO-3 适配测试
2. WandB 日志完善（生成质量指标：avg response length、vocab diversity、reward distribution）
3. Checkpoint 恢复训练（断点续训）
4. 支持 `gspo_reward_type = "model"`（切换到独立 RM 模型打分，适用于无 ground truth 的通用偏好数据）
5. KL 约束（路径 C 的 Phase 2 升级）
6. 学习率 warmup + cosine decay 的 GSPO 适配
7. 训练监控：每 N 步采样一个 prompt 展示生成的 response

### Phase 4：消融实验 & 调优（持续）

**目标**：验证各设计选择的有效性。

**实验维度**：
- K 的取值：{2, 4, 8, 16}
- gspo_beta 的取值：{0.01, 0.05, 0.1, 0.5, 1.0}
- gen_every_n_steps 的取值：{1, 4, 8, 16}
- gspo_num_mask_samples 的取值：{1, 2, 4}
- block_reduce: {"mean", "sum"}
- 生成 temperature: {0.8, 1.0, 1.2}
- 有/无 reference model 的 s_old anchor
- 有/无 reward 归一化

---

## 附录 A：关键数学符号对照

| 符号 | 含义 |
|:-----|:-----|
| $x$ | prompt |
| $y$ | response |
| $\pi_\theta$ | 当前 policy 模型 |
| $\pi_{\text{old}}$ | reference 模型 |
| $s_\theta(y)$ | block score（pseudo-log-likelihood） |
| $R(y)$ | math rule-based binary reward（Phase 1）或 RM 模型打分（Phase 3） |
| $K$ | 每个 prompt 采样的 response 数量 |
| $\beta$ | GSPO 温度系数 |
| $p_i$ | block i 的随机 mask ratio |
| $\sigma$ | sigmoid 函数 |

## 附录 B：与现有 DPO 实现的差异总结

| 维度 | DPO (dpo/) | GSPO (gspo/) |
|:-----|:-----------|:-------------|
| 数据来源 | 离线 pairwise JSONL | 在线生成 |
| 数据格式 | chosen + rejected pairs | prompt only |
| 采样方式 | 无（直接使用数据） | WeDLM iterative decoding |
| 比较方式 | pairwise (1 vs 1) | group-wise (best vs K-1 others) |
| 信号来源 | 人工标注的 preference | Math rule-based binary reward（Phase 1）；可扩展为 RM 模型（Phase 3） |
| Reference | 固定 ref model | 固定 ref model（同 DPO） |
| Loss | pairwise DPO | GSPO group contrastive |
| 生成频率 | 无 | 每 N 步一次 |
| 核心计算 | 2 份 response scoring | K 份 response scoring + K 次生成 |

## 附录 C：风险与缓解措施

| 风险 | 概率 | 影响 | 缓解措施 |
|:-----|:-----|:-----|:---------|
| 生成速度过慢 | 高 | 训练耗时翻倍以上 | Phase 2 Rollout Buffer + 降低 gen_every_n_steps |
| 内存不足 | 低 | 无法运行 | 无独立 RM（双模型常驻 ~32GB + optimizer ~32GB ≈ 64-72GB）；逐分支 backward；ZeRO-3 |
| Reward hacking | 低 | 模型退化 | Math binary reward 天然抗 reward hacking；KL 约束 + ref anchor |
| 生成质量不稳定 | 中 | 训练信号噪声大 | entropy threshold 调优 + pos_penalty |
| Block score 偏差导致优化方向错误 | 低 | 训练不收敛 | num_mask_samples ≥ 2 + Phase 2 升级到 REINFORCE |
| 多 GPU 同步问题 | 低 | 训练 hang | 生成阶段 barrier + per-GPU 独立 buffer |
| LLMEngine 与训练环境的兼容性 | 中 | 生成阶段 crash | Phase 0 优先验证 LLMEngine 在训练 context 中正常工作 |
