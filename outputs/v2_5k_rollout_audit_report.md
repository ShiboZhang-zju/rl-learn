# V2-5K Rollout Variance Audit

日期：2026-08-28

## 1. 审计目标

验证最佳 V2-5K SFT policy 是否具有进入 GRPO 的有效 group-relative reward variance。本次只做 inference + verifier reward audit，不进行任何模型更新。

Policy：

```text
outputs/sft_v2_5k_p800/checkpoint-1565
```

数据：

```text
data/processed/v2_answer_only_test.jsonl
```

共 `1000` 个 prompts，每个 prompt 独立采样 `8` 个 rollout。

## 2. Sampling 与 reward 配置

- `num_generations = 8`
- `temperature = 0.8`
- `top_p = 0.95`
- `max_new_tokens = 64`
- `dtype = bfloat16`
- `batch_size = 8`
- `seed = 20260828`
- device：P800 / CUDA

Reward 严格使用 exact-answer verifier：

```text
reward = 1  if parsed answer == ground truth
reward = 0  otherwise
```

格式无效或无法解析的 rollout reward 也为 `0`，但单独统计 format-valid 和 parse-success ratio。

## 3. V2-5K 结果

| Metric | V2-5K |
|---|---:|
| Total prompts | 1000 |
| Rollouts | 8000 |
| Greedy accuracy | 77.9% |
| Mean rollout reward | **74.3375%** |
| Pass@8 | **92.1%** |
| Mixed-reward group ratio | **39.3%** |
| Zero-variance group ratio | **60.7%** |
| All-wrong group ratio | 7.9% |
| All-correct group ratio | **52.8%** |
| Average correct rollouts/group | **5.947** |
| Average unique answers/group | **1.520** |
| Format-valid rollout ratio | 99.8875% |
| Parse-success rollout ratio | 99.8875% |

其中：

```text
zero-variance = all-wrong + all-correct = 7.9% + 52.8% = 60.7%
```

## 4. correct_count 分布

| Correct rollouts in group | Prompt count | Ratio |
|---:|---:|---:|
| 0 | 79 | 7.9% |
| 1 | 44 | 4.4% |
| 2 | 48 | 4.8% |
| 3 | 41 | 4.1% |
| 4 | 59 | 5.9% |
| 5 | 64 | 6.4% |
| 6 | 55 | 5.5% |
| 7 | 82 | 8.2% |
| 8 | 528 | 52.8% |
| **Total** | **1000** | **100%** |

非 zero-variance 的 mixed groups 数量为：

```text
44 + 48 + 41 + 59 + 64 + 55 + 82 = 393
```

即 `39.3%` 的 prompts 能为 GRPO 提供同组内 0/1 相对优势信号。

## 5. 与 Answer-only 3 epoch 对照

旧 policy：`outputs/sft_answer_only_3ep_p800/`

| Metric | Answer-only 3ep | V2-5K |
|---|---:|---:|
| Greedy accuracy | 30.8% | **77.9%** |
| Mean rollout reward | 32.6% | **74.3375%** |
| Pass@8 | 78.0% | **92.1%** |
| Mixed-reward | **77.4%** | 39.3% |
| Zero-variance | 22.6% | **60.7%** |
| All-wrong | 22.0% | **7.9%** |
| All-correct | 0.6% | **52.8%** |
| Avg correct / group | 1.808 | **5.947** |
| Avg unique answers | 4.378 | **1.520** |
| Format-valid | 100.0% | 99.8875% |

## 6. 结果解释

### 6.1 Greedy accuracy 显著提升

V2-5K 的 greedy accuracy 为 `77.9%`，远高于 Answer-only 3 epoch 的 `30.8%`。采样平均 reward 也从 `32.6%` 提升到 `74.3375%`，说明新 policy 的基础任务能力明显更强。

### 6.2 Mixed-reward group 明显减少

mixed-reward 从 `77.4%` 降至 `39.3%`。这不是模型退化，而是 policy 变得更确定：大量 prompts 的 8 个 rollout 全部正确。

### 6.3 All-correct 与 zero-variance 显著增加

V2-5K 有 `52.8%` 的 group 是 8/8 全部正确，导致相对 advantage 为零；zero-variance 总计 `60.7%`。

因此，当前 policy 的问题不是 all-wrong 太多，而是高置信度正确 group 太多，随机采样没有产生足够组内差异。

### 6.4 Exploration diversity 降低

平均 unique answers/group 从 `4.378` 降至 `1.520`。这与高 greedy accuracy 一致：模型大多数情况下稳定生成同一个正确答案，只有少数 prompt 仍然存在错误/探索分支。

### 6.5 Format/verifier 状态良好

8000 个 rollout 中：

- format-valid：`7991/8000`；
- parse-success：`7991/8000`；
- 无效/无法解析：`9/8000`。

因此当前主要限制是 reward variance，不是 parser 或输出格式问题。

## 7. GRPO readiness 判断

按照预设标准：

```text
mixed-reward >= 40%~50%
all-wrong 较低
仍有明显答案多样性
```

当前 V2-5K：

- mixed-reward：`39.3%`，略低于 40% 下限；
- all-wrong：`7.9%`，较低；
- average unique answers：`1.520`，多样性偏低；
- zero-variance：`60.7%`，偏高；
- all-correct：`52.8%`，很高。

### 结论

V2-5K 已经具备很强的任务能力和较低的 all-wrong 比例，但在当前 `temperature=0.8, top_p=0.95` 下，group 内随机性不足。它可以作为后续 GRPO 的候选 policy，但**不建议立即启动正式 GRPO**：约 60.7% 的 group 没有有效 relative advantage，首轮 GRPO 的有效 batch 比例可能偏低。

本次不启动 GRPO。

## 8. 与 Answer-only 3 epoch 的初始化选择

两者特点不同：

- Answer-only 3ep：mixed groups 多，探索多，但 greedy accuracy 低，all-wrong 高；
- V2-5K：greedy accuracy 高，all-wrong 低，但 all-correct/zero-variance 高，探索不足。

如果目标是**获得更多 GRPO group-level 学习信号**，3ep policy 的 mixed ratio 更好，但它会带来更多全错 groups。

如果目标是**作为高质量最终 policy 再做微调**，V2-5K 更好，但需要先确认如何恢复有效 group variance。

在不改变本次 audit sampling 参数的前提下，建议下一步优先：

1. 审计 V2-5K 较早 checkpoint，例如 checkpoint-1252、checkpoint-939；
2. 用相同固定采样设置比较其 mixed/all-correct/zero-variance；
3. 选择 mixed ratio 和 all-wrong 之间更平衡的 checkpoint；
4. 之后再做极小规模 GRPO smoke test。

这不是本次 audit 中修改 sampling 参数，也不应把当前 V2-5K 直接判定为不适合 RL；更准确地说，它是“能力很强，但当前 group variance 偏低”。

## 9. 输出文件

逐 prompt、逐 rollout 的完整结果：

```text
outputs/v2_5k_rollout_audit.jsonl
```

汇总结果：

```text
outputs/v2_5k_rollout_audit.summary.json
```

审计脚本：

```text
scripts/rollout_audit.py
```
