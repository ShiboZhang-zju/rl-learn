# 第二次 SFT：K&K v2 实验报告

日期：2026-08-28

## 1. 实验目标

本轮从 Base Model 重新训练 Answer-only LoRA SFT，主要验证：

1. statement-first generator 是否减少旧 generator 的结构 shortcut；
2. 训练规模从 1k 增加到 5k 是否改善真实泛化；
3. 按 validation exact accuracy 选择 checkpoint 是否比直接使用最终 checkpoint 更合理。

本轮没有加入 CoT、GRPO/RL、DPO、answer-weighted loss、curriculum、hard negative 或大规模超参搜索。

## 2. Generator 与数据

### Generator

新增：`src/kk_sft/logic.py` 中的 `generate_puzzle_v2()`。

生成流程：

```text
随机生成 statements
→ exact solver 求解
→ 保留 solution_count == 1
→ 使用 solver 返回的 assignment 作为 ground truth
```

版本信息：

- dataset version：`kk-v2`
- generator version：`kk-v2-statement-first`
- seed：`20260829`
- people：Alice、Bob、Carol
- operator：same、different、and、or、not

### 数据规模

| Split | 数量 |
|---|---:|
| train | 5000 |
| val | 500 |
| test | 1000 |
| train-1k | 从 v2 train 固定取前 1000 条 |

完整性：

- 总 puzzle：6500；
- unique puzzle：6500；
- duplicate：0；
- `solution_count == 1`：6500/6500；
- raw/processed 行数一致；
- Legacy Test 原始 500 条未覆盖。

### 标签分布

| Pattern | Train 5k | Val 500 | Test 1k |
|---|---:|---:|---:|
| KKK | 591 | 60 | 126 |
| KKN | 621 | 73 | 120 |
| KNK | 674 | 52 | 128 |
| KNN | 603 | 63 | 140 |
| NKK | 597 | 60 | 98 |
| NKN | 633 | 63 | 133 |
| NNK | 628 | 65 | 129 |
| NNN | 653 | 64 | 126 |

标签分布不完全均匀，但没有旧实验中那种极端单一类别现象。

### 结构 shortcut audit

审计文件：`outputs/dataset_v2_feature_audit.json`

v2 train 中答案与单个 operator count 的互信息：

| Feature | v1 train MI | v2 train MI |
|---|---:|---:|
| same count | 0.1828 | **0.0927** |
| different count | 0.1215 | **0.0977** |
| and count | 0.0338 | **0.0225** |
| or count | 0.0532 | **0.0218** |
| not count | 0.0275 | **0.0032** |

v2 的核心 operator-count shortcut 相比 v1 明显减弱。v2 test 的对应 MI 也总体低于 v1 test：

- same：`0.1236` vs `0.1407`；
- different：`0.1184` vs `0.1466`；
- and：`0.0345` vs `0.0753`；
- or：`0.0512` vs `0.0476`；
- not：`0.0195` vs `0.0412`。

但是 top-level operator sequence 与 label 仍然存在关联，v2 train MI 为 `0.7807 bits`。因此本轮只能得出“shortcut 明显减弱”，不能声称完全消除结构相关性。

人工抽查：固定随机抽取 30 条 train 样本，保存了 raw puzzle、solver answer、SFT prompt 和 completion；solver answer 与记录一致。

## 3. 训练配置

两组实验均从 Base Model 重新开始，不加载第一次 SFT Adapter。

- Base：`Qwen/Qwen2.5-0.5B-Instruct`
- Answer-only completion
- LoRA rank：8
- LoRA alpha：16
- LoRA dropout：0.05
- target modules：Q/K/V/O、gate/up/down projections
- learning rate：`1e-4`
- batch size：16
- gradient accumulation：1
- BF16
- max length：512
- epochs：5
- P800 / CUDA

配置：

- `configs/sft_v2_1k.yaml`
- `configs/sft_v2_5k.yaml`

## 4. 每 epoch 结果与 checkpoint 选择

### V2-1K

| Epoch | Train loss | Val loss | Train exact | Val exact | Checkpoint |
|---:|---:|---:|---:|---:|---|
| 1 | 0.1552 | 0.0950 | 18.3% | 17.4% | `checkpoint-63` |
| 2 | 0.0930 | 0.0910 | 27.1% | 24.4% | `checkpoint-126` |
| 3 | 0.0862 | 0.0852 | 33.3% | 26.6% | `checkpoint-189` |
| 4 | 0.0786 | 0.0812 | 36.8% | 33.2% | `checkpoint-252` |
| 5 | 0.0736 | 0.0772 | 44.4% | **33.4%** | `checkpoint-315` |

最佳 checkpoint：

```text
outputs/sft_v2_1k_p800/checkpoint-315
```

### V2-5K

| Epoch | Train loss | Val loss | Train exact | Val exact | Checkpoint |
|---:|---:|---:|---:|---:|---|
| 1 | 0.1124 | 0.0691 | 41.5% | 41.8% | `checkpoint-313` |
| 2 | 0.0582 | 0.0425 | 65.9% | 63.0% | `checkpoint-626` |
| 3 | 0.0357 | 0.0306 | 78.3% | 71.2% | `checkpoint-939` |
| 4 | 0.0226 | 0.0286 | 85.0% | 72.2% | `checkpoint-1252` |
| 5 | 0.0154 | 0.0287 | 89.8% | **74.8%** | `checkpoint-1565` |

最佳 checkpoint：

```text
outputs/sft_v2_5k_p800/checkpoint-1565
```

两组最佳 val exact 都出现在第 5 epoch，因此本轮“best checkpoint”恰好等于 final checkpoint；选择依据仍然是 validation exact，而不是 train loss。

## 5. 最终统一评估

评估设置：

- greedy decoding；
- BF16；
- batch size：32；
- max new tokens：64；
- 同一 parser 和答案格式；
- Legacy Test：第一次实验固定的 500 条；
- V2 Test：本轮新生成的 1000 条。

### 总结果

| Model | Generator | Train size | Best epoch | Train exact | Val exact | Legacy Test | V2 Test |
|---|---|---:|---:|---:|---:|---:|---:|
| V1 | old | 1k | 10 | 91.5% | 46.0% | **47.6%** | 27.2% |
| V2-1K | statement-first | 1k | 5 | 44.4% | 33.4% | 20.6% | 35.9% |
| V2-5K | statement-first | 5k | 5 | 89.8% | **74.8%** | 43.4% | **77.9%** |

所有完整评估结果、预测分布和 8×8 confusion matrix：

```text
outputs/v2_eval/metrics.json
```

### Format accuracy

| Model | Legacy Test | V2 Test |
|---|---:|---:|
| V1 | 99.6% | 100.0% |
| V2-1K | 90.6% | 100.0% |
| V2-5K | 67.2% | 100.0% |

V2 模型在 Legacy Test 上出现较多格式失败，说明 Legacy 与新 v2 prompt/statement 分布存在较强 domain shift。V2 Test 上三者均能稳定解析，V1/V2 的答案能力比较应优先看各自对应测试集，并同时报告 Legacy 迁移结果。

## 6. 关键分析

### 6.1 Generator 修改是否有效？

在同样 1k 训练规模下：

- V1 在 Legacy Test：`47.6%`；V2-1K：`20.6%`；
- V1 在 V2 Test：`27.2%`；V2-1K：`35.9%`。

V2-1K 在新 generator 的 V2 Test 上高于 V1，且 prediction distribution 更接近 8 类分布；但 V2-1K 的 train/val 绝对能力低于 V1，说明 generator 改动并非单纯提升所有指标。

更合理的结论是：

> statement-first 减少了旧 generator 的 operator-count shortcut，但 1k 数据下训练仍不足，且新旧测试集之间存在明显分布差异。

### 6.2 增加数据是否有效？

V2-1K → V2-5K：

- Train：`44.4% → 89.8%`；
- Val：`33.4% → 74.8%`；
- V2 Test：`35.9% → 77.9%`；
- V2 Test train-test gap：`8.5` 个百分点；
- V2-1K train-test gap：`8.5` 个百分点。

5k 数据带来非常明显的提升，且 V2-5K prediction distribution 已接近均衡，说明增加有效、多样化的 statement-first 数据是本轮最主要的收益来源。

### 6.3 泛化还是记忆？

V2-5K 的 train exact 为 `89.8%`，V2 test 为 `77.9%`，同时 val 为 `74.8%`。虽然存在一定 gap，但 val/test 均显著高于随机的 `12.5%`，且预测分布健康，因此不能解释为单纯记忆。

V2-1K 仍有较明显欠拟合：train `44.4%`、val `33.4%`、V2 test `35.9%`。

### 6.4 Shortcut 是否仍存在？

核心 operator count 与 label 的关联明显减弱，但 top-level operator sequence 仍有一定相关性。因此 v2 不是完全去 shortcut 的数据集。V2-5K 在 V2 Test 上达到 `77.9%`，且预测分布接近 ground truth，说明模型已经获得了较强的 puzzle-dependent discrimination；不过后续若追求严格因果结论，仍应构造更强的 controlled test 或做结构反事实测试。

## 7. 输出产物

### Generator 与数据

- `src/kk_sft/logic.py`
- `scripts/generate_dataset_v2.py`
- `scripts/audit_dataset_v2.py`
- `data/raw_v2/train.jsonl`
- `data/raw_v2/val.jsonl`
- `data/raw_v2/test.jsonl`
- `data/raw_v2/train_1k.jsonl`
- `data/processed/v2_answer_only_train.jsonl`
- `data/processed/v2_answer_only_train_1k.jsonl`
- `data/processed/v2_answer_only_val.jsonl`
- `data/processed/v2_answer_only_test.jsonl`

### Audit

- `outputs/dataset_v2_feature_audit.json`
- `outputs/dataset_v2_audit_report.md`

### Adapter

- `outputs/sft_v2_1k_p800/`
- `outputs/sft_v2_5k_p800/`

### Epoch metrics

- `outputs/sft_v2_1k_epoch_metrics.json`
- `outputs/sft_v2_5k_epoch_metrics.json`
- `outputs/sft_v2_metrics.json`

### Final predictions

- `outputs/v2_eval/v1_legacy_test.jsonl`
- `outputs/v2_eval/v1_v2_test.jsonl`
- `outputs/v2_eval/v2_1k_legacy_test.jsonl`
- `outputs/v2_eval/v2_1k_v2_test.jsonl`
- `outputs/v2_eval/v2_5k_legacy_test.jsonl`
- `outputs/v2_eval/v2_5k_v2_test.jsonl`

### Final evaluation metrics

- `outputs/v2_eval/metrics.json`

## 8. 最终结论

本轮实验支持以下结论：

1. v2 statement-first generator 成功生成了 6500 条唯一解 puzzle，solver verification 全部通过；
2. v2 的核心 operator-count shortcut 相比 v1 明显减弱，但 top-level operator sequence 仍有相关性；
3. V2-1K 仍明显受训练规模限制；
4. V2-5K 在 V2 Test 上达到 `77.9%`，显著高于 V2-1K 的 `35.9%` 和 V1 的 `27.2%`；
5. 这轮最大的有效因素是增加 statement-first 数据规模，而不是简单延长 1k 数据训练；
6. V2-5K 已经具备较强的真实任务泛化能力，但在正式进入 GRPO 前，仍建议至少对 V2-5K 做 8-sample reward variance audit，并优先使用 V2 Test/controlled test 验证 reward 是否被结构 shortcut 污染。

本轮没有生成 Controlled Test，因为在已有 v2 审计和统一 V2 Test 结果已经足以完成当前阶段比较后，优先保留了实验变量的简洁性。
