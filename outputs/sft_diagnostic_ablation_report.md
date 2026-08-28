# SFT 诊断、训练步数与 Rollout Audit 报告

日期：2026-08-28

## 1. 当前结论

本轮实验把“已经证实”和“仍需谨慎解释”的部分分开：

### 已证实

1. Train/val/test 的答案标签没有严重不均衡。
2. CoT 1 epoch 的任务级学习接近随机：train/val/test exact 分别为 `12.6%/9.5%/16.2%`。
3. Answer-only 3 epoch 明显恢复了 puzzle-dependent discrimination。
4. Answer-only 5/10 epoch 继续提升，说明此前存在明显的训练不足。
5. Answer-only 3 epoch 的随机 rollout 已有较强 RL 信号：500 个 prompt 中 `78.0%` 至少有一个正确 rollout，`77.4%` 是 mixed-reward group。

### 仍不能完全确认

“CoT reasoning token dilution 是唯一主要原因”尚未被严格因果验证。CoT 3 epoch 的结果很差，支持长 CoT supervision 不利；但 Answer-only 从 1 epoch 到 3/5/10 epoch 的持续提升也证明训练步数是重要变量。还没有完成 reasoning 保留、answer-only loss masking 的直接对照。

## 2. 训练集标签分布

| Pattern | Train | Val | Test |
|---|---:|---:|---:|
| KKK | 122 | 13 | 71 |
| KKN | 130 | 30 | 70 |
| KNK | 107 | 28 | 56 |
| KNN | 129 | 30 | 61 |
| NKK | 123 | 19 | 56 |
| NKN | 121 | 21 | 58 |
| NNK | 134 | 34 | 80 |
| NNN | 134 | 25 | 48 |

三份数据合计 `1700` 条，8 类均有覆盖，没有发现标签分布导致的 KKN/KKK 偏置。

## 3. CoT 与 Answer-only 训练结果

| 实验 | Train exact | Val exact | Test exact | Test format |
|---|---:|---:|---:|---:|
| CoT + Answer，1 epoch | 12.6% | 9.5% | 16.2% | 100.0% |
| CoT + Answer，3 epochs | 13.0% | 15.0% | 14.2% | 100.0% |
| Answer-only，1 epoch | 17.8% | 15.0% | 14.8% | 100.0% |
| Answer-only，3 epochs | 36.2% | 26.0% | 30.8% | 100.0% |
| Answer-only，5 epochs | 50.3% | 31.5% | 36.6% | 100.0% |
| Answer-only，10 epochs | **91.5%** | **46.0%** | **46.8%** | 99.8% |

### 解释

- CoT 3 epoch 的 test `14.2%` 没有超过 CoT 1 epoch 的 `16.2%`，因此“单纯增加 CoT 训练步数”没有解决任务映射问题。
- Answer-only 从 1→3→5→10 epoch 持续提升，说明早期的主要问题至少包含明显的 underfitting。
- Answer-only 10 epoch 的 train/test gap 为 `91.5% → 46.8%`，这时已经出现明显泛化差距；但 test 仍继续提升，说明 0.5B 并非完全没有能力上限。
- 这些结果支持“长 CoT 监督可能降低有效 answer supervision 权重”，但不能把它单独视为已经完全证实的唯一原因。

## 4. CoT completion 的 token 结构

- CoT completion 平均约 `279.3` tokens；
- answer 部分平均约 `19.5` tokens；
- answer 占 completion 约 `7.0%`。

这解释了为什么 CoT 的 token-level accuracy 很高但 answer exact 很低，但要完成因果确认，还需要 reasoning 保留、answer-only loss masking 的实验。

## 5. 数据结构与潜在 shortcut 审计

审计文件：`outputs/dataset_feature_audit.json`

### 完整性

- 三个 split 共 `1700` 条；
- puzzle key 唯一数：`1700`；
- duplicate：`0`；
- 每条 puzzle 的 `solution_count=1`；
- speaker 顺序在所有 split 均为 `Alice, Bob, Carol`。

### 复杂度代理的 split 均值

| Feature | Train | Val | Test |
|---|---:|---:|---:|
| statement chars | 157.2 | 161.2 | 154.8 |
| expression nodes | 6.42 | 6.62 | 6.38 |
| expression depth | 2.31 | 2.30 | 2.36 |
| person_is count | 1.32 | 1.39 | 1.39 |
| same count | 1.52 | 1.39 | 1.52 |
| different count | 1.53 | 1.70 | 1.45 |
| not count | 0.67 | 0.67 | 0.65 |
| and count | 0.67 | 0.75 | 0.67 |
| or count | 0.71 | 0.73 | 0.70 |

粗粒度的 split difficulty proxy 接近，没有发现明显的 train/test 复杂度断层。

### 需要保留的警惕

答案模式与结构特征并非完全独立。训练集按答案模式统计时，`same/different/and/or` 的均值存在明显差异，例如：

- KKK：`same=2.39`、`different=0.78`；
- NNN：`same=0.65`、`different=2.21`；
- NNK：`and=0.84`；
- KKK：`or=1.04`；
- NNN：`or=0.36`。

这很可能来自生成器：它先采样 target，再筛选与 target 一致的表达式。因此虽然标签均衡，但模型可能利用 statement operator/template 作为 shortcut。当前只能说“没有 label imbalance”，不能扩大成“数据完全没有问题”。

## 6. Parser / verifier 抽查

从 `outputs/answer_only_3ep_test_eval.jsonl` 固定随机抽查 50 条：

- format valid：`50/50`；
- parser 正确解析：`50/50`；
- 与记录中的 `parsed_answer/correct` 保持一致：`50/50`；
- 抽查样本中 exact correct：`13/50`，与随机抽样对应的模型能力一致。

抽查明细保存在：`outputs/dataset_feature_audit.json`。

## 7. Answer-only 3 epoch Rollout Audit

Policy：`outputs/sft_answer_only_3ep_p800/`

设置：

- 测试 prompt：500；
- 每个 prompt 采样：8 次；
- temperature：`0.8`；
- top-p：`0.95`；
- max new tokens：`64`；
- dtype：BF16；
- seed：`20260828`。

结果：

| 指标 | 结果 |
|---|---:|
| zero-variance group ratio | 22.6% |
| mixed-reward group ratio | **77.4%** |
| all-wrong group ratio | 22.0% |
| all-correct group ratio | 0.6% |
| groups with at least one correct | **78.0%** |
| pass@8 | **78.0%** |
| mean unique answers/group | 4.378 |
| mean correct rollouts/group | 1.808 |
| mean reward | 0.326 |
| mean reward std | 0.314 |
| format-valid rollout ratio | 100.0% |

这里 reward 为 `exact answer=1.0` 加 `format=0.1`。因此 mixed-reward group 指同组同时出现 exact correct 和 incorrect rollout。

这个结果说明 Answer-only 3 epoch 已经是可进行 GRPO 可行性研究的 policy：绝大多数 group 具有相对优势信号，且不是所有采样都坍缩到同一个答案。但正式进入 GRPO 前仍应考虑 reward 设计、采样温度和 group size 的敏感性。

机器可读结果：

- `outputs/answer_only_3ep_rollout_audit.jsonl`
- `outputs/answer_only_3ep_rollout_audit.summary.json`

## 8. 实验产物

### Adapter

- `outputs/sft_cot_3ep_p800/`
- `outputs/sft_answer_only_5ep_p800/`
- `outputs/sft_answer_only_10ep_p800/`

### 评估文件

- `outputs/sft_cot_3ep_train_eval.jsonl`
- `outputs/sft_cot_3ep_val_eval.jsonl`
- `outputs/sft_cot_3ep_test_eval.jsonl`
- `outputs/answer_only_5ep_train_eval.jsonl`
- `outputs/answer_only_5ep_val_eval.jsonl`
- `outputs/answer_only_5ep_test_eval.jsonl`
- `outputs/answer_only_10ep_train_eval.jsonl`
- `outputs/answer_only_10ep_val_eval.jsonl`
- `outputs/answer_only_10ep_test_eval.jsonl`

### 脚本和配置

- `scripts/rollout_audit.py`
- `scripts/audit_dataset_features.py`
- `configs/sft_cot_3ep.yaml`
- `configs/sft_answer_only_5ep.yaml`
- `configs/sft_answer_only_10ep.yaml`

## 9. 阶段性决策建议

当前最合理的候选 policy 是 `Answer-only 3 epoch`，因为它已具备较好的 mixed-reward rollout 结构；但 Answer-only 10 epoch 的 greedy test 更高，是否用 10 epoch 作为 RL 起点需要重新跑 rollout audit，不能只按 greedy accuracy 选择。

在正式 GRPO 前建议：

1. 对 Answer-only 5/10 epoch 也跑同样的 8-sample rollout audit；
2. 固定 verifier reward，比较不同 policy 的 mixed group 比例；
3. 如果要确认 reasoning dilution，补做“保留 reasoning 输入、只对 answer 计算 loss”的 masked-loss 实验；
4. 检查 operator/template shortcut，必要时按结构和答案做分层均衡或反事实验证。

当前还不应使用“mode collapse”作为严格术语，更准确的描述是 CoT 1 epoch 的 `prediction collapse` 或 `majority-pattern shortcut`。
