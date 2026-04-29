# 重构实施清单

> 按 Phase 顺序执行，每步完成勾选。**原则：只迁移不做旧代码改动，始终保证旧代码可运行。**

---

## Phase 1: 基础骨架搭建

### 1.1 创建目录骨架
- [x] 创建 `posttraining/wedlm_train/` 及所有子包目录
- [x] 为每个子包创建 `__init__.py`（先空，逐步填充）

### 1.2 拆分 config
- [x] 新建 `wedlm_train/config/base.py`：抽取共用字段（model, data, training, attention, deepspeed, logging）
- [x] 新建 `wedlm_train/config/sft.py`：`SFTConfig(BaseTrainingConfig)`，SFT 独有字段（ar_loss, loss_weighting）
- [x] 新建 `wedlm_train/config/dpo.py`：`DPOConfig(SFTConfig)`，DPO 独有字段（dpo_*）
- [x] 新建 `wedlm_train/config/gspo.py`：`GSPOConfig(SFTConfig)`，GSPO 独有字段（gspo_*）
- [x] 新建 `wedlm_train/config/registry.py`：`from_yaml(path) -> BaseTrainingConfig`，按 `training_mode` 分发
- [ ] 保留原 `WeDLMTrainingConfig` 为别名（兼容迁移期）

### 1.3 迁移核心无业务模块
- [x] `dpo/src/masking.py` → `wedlm_train/batch/masking.py`（纯函数，直接复制）
- [x] `dpo/src/batch.py` → `wedlm_train/batch/wedlm_batch.py`（`WeDLMBatch` + `build_wedlm_batch`）
- [x] `dpo/src/model.py` → `wedlm_train/model/forward.py` + `wedlm_train/model/rotary.py`
- [x] `dpo/src/attention.py` → `wedlm_train/attention/dense.py` + `magi.py` + `registry.py`
- [x] `dpo/src/loss.py` 中 `compute_mlm_loss` → `wedlm_train/loss/mlm.py`
- [x] `dpo/src/loss.py` 中 `compute_ar_loss` → `wedlm_train/loss/ar.py`
- [x] `dpo/src/loss.py` 中 `compute_block_scores` → `wedlm_train/loss/block_scores.py`
- [x] `dpo/src/loss.py` 中 `compute_dpo_loss` → `wedlm_train/loss/dpo.py`
- [x] `dpo/src/loss.py` 中 `compute_gspo_loss` → `wedlm_train/loss/gspo.py`
- [x] `dpo/src/loss.py` 中 `compute_masked_token_logps` → `wedlm_train/loss/token_logps.py`
- [x] `dpo/src/reward.py` 拆分到 `wedlm_train/reward/` 各文件
- [x] 每个模块迁移后更新对应 `__init__.py` 的 `__all__`

### 1.4 合并且统一 data 模块
- [x] `dpo/src/data.py` 中 `get_im_end_token_id` + 常量 → `wedlm_train/data/utils.py`
- [x] `dpo/src/data.py` 中 `WeDLMPackedDataset` + `PackedBatch` → `wedlm_train/data/packed_dataset.py`
- [x] `finetune/src/data.py` 中 `WeDLMShuffledPackedDataset` → 合入 `wedlm_train/data/packed_dataset.py`
- [x] `dpo/src/data.py` 中 `WeDLMPairwiseDataset` → `wedlm_train/data/pairwise_dataset.py`
- [x] `dpo/src/data.py` 中 `WeDLMPromptDataset` → `wedlm_train/data/prompt_dataset.py`
- [x] 各 collate_fn → `wedlm_train/data/collate.py`
- [x] 对比 `finetune/src/data.py` 与 `dpo/src/data.py` 差异，确保无遗漏

---

## Phase 2: 业务逻辑拆分

### 2.1 提取 BaseTrainer
- [x] 从 `dpo/src/trainer.py` 提取 `_prepare_training()` → `BaseTrainer.__init__`
- [x] 提取 `train()` 主循环（epoch/batch/backward/step/log/save） → `BaseTrainer.train()`
- [x] 定义抽象方法 `train_step(batch) -> (loss, logs)`
- [x] 提取 `_init_wandb` → `wedlm_train/trainer/callbacks.py`
- [x] 提取 checkpoint save/logic → `wedlm_train/trainer/callbacks.py`

### 2.2 提取 SFTTrainer
- [x] `SFTTrainer(BaseTrainer)`：实现 `train_step` 逻辑 (SFT)
- [x] 管理 SFT 特有的 dataloader 创建
- [x] `_compute_ar_loss` 保留为 SFT/DPO 共用（从 base.py 继承）

### 2.3 提取 DPOTrainer
- [x] `DPOTrainer(SFTTrainer)`：管理 reference model 加载/冻结
- [x] 实现 `train_step` 逻辑（低内存 chosen/rejected 分步 backward）
- [x] DPO 特有 dataset 初始化

### 2.4 提取 GSPOTrainer
- [x] `GSPOTrainer(SFTTrainer)`：管理 online rollout + reference model
- [x] `_sample_gspo_online_candidates`、`_pack_gspo_candidates`、`_compute_gspo_rewards` 作为 private method
- [x] 实现 `train_step` 逻辑 (GSPO)

### 2.5 验证
- [ ] 运行原有 smoke tests 对各 trainer 进行单步测试（需 torch + GPU 环境）
- [ ] 确认 loss 数值与原实现一致

---

## Phase 3: CLI 与配置

### 3.1 重写入口
- [x] 新建 `scripts/train.py`：精简 CLI（`--config`, `--override key=value`）
- [x] 实现 YAML 配置 + CLI override 合并逻辑
- [x] 根据 `training_mode` 自动选择 Trainer 子类

### 3.2 配置覆盖机制
- [x] 实现 `--override` 参数解析（支持 key=value 格式）
- [ ] 或使用 Hydra/OmegaConf（按需引入）— 暂不引入额外依赖

### 3.3 预设配置文件
- [x] 迁移 `dpo/configs/example_sft.yaml` → `configs/sft_example.yaml`
- [x] 迁移 `dpo/configs/example_dpo.yaml` → `configs/dpo_example.yaml`
- [x] 迁移 `dpo/configs/deepmath_gspo_robust.yaml` → `configs/gspo_deepmath_robust.yaml`

---

## Phase 4: 测试与文档

### 4.1 迁移 smoke tests
- [x] 复制 `dpo/scripts/smoke_test_*.py` → `scripts/smoke_tests/`
- [x] 更新 import 路径为新模块
- [ ] 全部运行通过（需 torch 环境）

### 4.2 单元测试
- [x] `tests/test_config.py`：配置加载、字段校验
- [x] `tests/test_masking.py`：`sample_block_mask_ratios`, `reorder_block`, `build_2d_attention_mask`
- [x] `tests/test_loss.py`：`compute_dpo_loss`, `compute_gspo_loss` 数值正确性（合并 dpo/gspo/mlm）
- [ ] `tests/test_reward.py`：各 reward 函数输出 shape 和范围 — 待实现

### 4.3 文档
- [x] `posttraining/README.md`：快速开始、配置说明、自定义 reward 指南
- [ ] 原 `dpo/README.md` 添加迁移指引（可选）

---

## Phase 5: 收尾

### 5.2 包配置
- [ ] 更新根 `pyproject.toml`，`find_packages` 包含 `wedlm_train` — 保持原有不变
- [ ] 验证 `pip install -e .` 正常

### 5.3 最终验证
- [ ] 运行全量 smoke tests（需 torch 环境）
- [ ] 运行单元测试 `pytest tests/`（需 torch 环境）
- [ ] SFT/DPO/GSPO 各模式单步训练不报错

---

## 快速命令参考

```bash
# 创建目录骨架
mkdir -p posttraining/wedlm_train/{config,data,batch,model,attention,loss,reward,trainer,utils}
mkdir -p posttraining/{scripts/{download,analysis,smoke_tests},configs,data,tests}

# 运行 smoke tests
cd posttraining
python scripts/smoke_tests/test_dpo_train_step.py --config configs/dpo_example.yaml
python scripts/smoke_tests/test_gspo_train_step.py --config configs/gspo_deepmath_robust.yaml

# 运行单元测试
pytest tests/ -v
```
