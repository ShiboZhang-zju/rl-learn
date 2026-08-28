# V2-5K 多 checkpoint Rollout Variance Audit 对比报告

## 1. 审计设置

三个 checkpoint 使用完全相同的 V2 Test 和 sampling：

- prompts：1000；每 prompt 8 rollouts；总 rollouts：8000/checkpoint；
- temperature：0.8；top-p：0.95；max_new_tokens：64；dtype：BF16；
- seed：20260828；batch size：8；
- reward：exact parsed answer 正确为 1，否则为 0；
- 不进行模型更新。

## 2. 统一比较表

| Metric | Epoch 3 / ckpt-939 | Epoch 4 / ckpt-1252 | Epoch 5 / ckpt-1565 |
|---|---:|---:|---:|
| Val exact | 71.20% | 72.20% | 74.80% |
| Greedy accuracy | 70.80% | 75.10% | 77.90% |
| Mean rollout reward | 66.24% | 70.33% | 74.34% |
| Pass@8 | 92.80% | 92.80% | 92.10% |
| Mixed-reward | 54.60% | 49.60% | 39.30% |
| Zero-variance | 45.40% | 50.40% | 60.70% |
| All-wrong | 7.20% | 7.20% | 7.90% |
| All-correct | 38.20% | 43.20% | 52.80% |
| Avg correct/group | 5.299 | 5.626 | 5.947 |
| Avg unique answers | 1.847 | 1.725 | 1.520 |
| Avg reward std | 0.233 | 0.210 | 0.167 |
| Format valid | 100.00% | 99.96% | 99.89% |
| Parse success | 100.00% | 99.96% | 99.89% |

## 3. correct_count 分布

| Correct rollouts / 8 | Epoch 3 | Epoch 4 | Epoch 5 |
|---:|---:|---:|---:|
| 0 | 72 | 72 | 79 |
| 1 | 72 | 62 | 44 |
| 2 | 75 | 59 | 48 |
| 3 | 77 | 51 | 41 |
| 4 | 77 | 60 | 59 |
| 5 | 75 | 72 | 64 |
| 6 | 83 | 107 | 55 |
| 7 | 87 | 85 | 82 |
| 8 | 382 | 432 | 528 |

## 4. 详细结果

### epoch3_checkpoint-939

- checkpoint：`outputs/sft_v2_5k_p800/checkpoint-939`
- validation exact：71.2%
- greedy accuracy：70.8%
- mean rollout reward：0.6624
- pass@8：92.8%
- mixed-reward：54.6%
- zero-variance：45.4%
- all-wrong：7.2%
- all-correct：38.2%
- average correct rollouts/group：5.299
- average unique answers/group：1.847
- average reward std/group：0.2331
- format valid：100.0000%
- parse success：100.0000%

### epoch4_checkpoint-1252

- checkpoint：`outputs/sft_v2_5k_p800/checkpoint-1252`
- validation exact：72.2%
- greedy accuracy：75.1%
- mean rollout reward：0.7033
- pass@8：92.8%
- mixed-reward：49.6%
- zero-variance：50.4%
- all-wrong：7.2%
- all-correct：43.2%
- average correct rollouts/group：5.626
- average unique answers/group：1.725
- average reward std/group：0.2100
- format valid：99.9625%
- parse success：99.9625%

### epoch5_checkpoint-1565

- checkpoint：`outputs/sft_v2_5k_p800/checkpoint-1565`
- validation exact：74.8%
- greedy accuracy：77.9%
- mean rollout reward：0.7434
- pass@8：92.1%
- mixed-reward：39.3%
- zero-variance：60.7%
- all-wrong：7.9%
- all-correct：52.8%
- average correct rollouts/group：5.947
- average unique answers/group：1.520
- average reward std/group：0.1666
- format valid：99.8875%
- parse success：99.8875%

## 5. 推荐

**推荐 `checkpoint-1252`（epoch 4）作为第一版 GRPO initialization。**

原因：

- 相比 epoch 3，epoch 4 的 greedy accuracy 从 70.8% 提升到 75.1%，pass@8 保持 92.8%，all-wrong 保持低位 7.2%；
- 相比 epoch 5，epoch 4 的 mixed-reward 从 39.3% 提升到 49.6%，average reward std 从 0.1666 提升到 0.2100，zero-variance 从 60.7% 降到 50.4%；
- 平均 unique answers/group 为 1.725，高于 epoch 5 的 1.520，仍保留更多 exploration diversity；
- epoch 4 的 all-correct 为 43.2%，低于 epoch 5 的 52.8%，不会过度牺牲 competence。

### 为什么不选 epoch 3

epoch 3 的 exploration 和 variance 最好：mixed-reward 54.6%、平均 reward std 0.2331、unique answers/group 1.847。但 greedy accuracy 只有 70.8%，mean reward 66.24%，相对 epoch 4 的能力损失没有换来足够大的额外 group variance。因此 epoch 3 更适合作为备选的高探索 policy，而不是首选。

### 为什么不选 epoch 5

epoch 5 的 greedy accuracy 最高（77.9%），但 52.8% 的 groups 为 all-correct，zero-variance 达到 60.7%，mixed-reward 降到 39.3%，平均 reward std 只有 0.1666。它更像高能力、低组内方差的成熟 policy，作为 GRPO 初始化时有效 relative advantage 的比例偏低。

## 6. 背景对照：旧 Answer-only 3 epoch

旧 policy 的 greedy accuracy 为 30.8%，mixed-reward 77.4%，zero-variance 22.6%，all-wrong 22.0%，平均 unique answers/group 4.378。它是高探索但能力较弱的 policy，不作为本轮 V2 checkpoint 的候选。

## 7. 结论

V2-5K 三个 checkpoint 都有较低的 all-wrong ratio 和较高 pass@8；epoch 4 在 competence、exploration 和 reward variance 之间最平衡。建议下一步仅对 `checkpoint-1252` 做 64~128 prompts 的 GRPO smoke test，本轮没有启动 GRPO。

完整逐 rollout 文件：

- `outputs/v2_5k_ckpt939_rollout_audit.jsonl`
- `outputs/v2_5k_ckpt1252_rollout_audit.jsonl`
- `outputs/v2_5k_rollout_audit.jsonl`
