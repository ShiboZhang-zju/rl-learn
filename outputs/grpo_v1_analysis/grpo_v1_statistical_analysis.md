# GRPO-V1 统计与行为归因分析

日期：2026-08-28
模式：零训练成本（仅复用已有预测 + 固定 200-prompt 采样，未重训/未改 checkpoint/reward/数据）
状态：**GRPO_V1_ANALYSIS_COMPLETE**

## A. Executive conclusion

> **GRPO 相对 SFT Epoch4 在 Final Holdout 上的 +1.05pp 提升不具有统计显著性**（McNemar exact p=0.238；paired bootstrap 95% CI 含 0）。GRPO 相对 SFT Epoch5 的 −1.54pp 差距同样不显著（p=0.076，CI 接近但不含跨越 0 的边缘情形）。行为变化真实存在且为 **policy sharpening + 双向极化**（mixed→all-correct 38 vs mixed→all-wrong 17），但尚未转化为可证明的性能提升。

## B. Paired statistics（N=2000，Final Holdout，greedy）

### B.1 Correctness transition：SFT Epoch4 → GRPO

| 类别 | 含义 | 数量 | 比例 |
|---|---:|---:|---:|
| CC | E4 对 & GRPO 对 | 1365 | 68.25% |
| CW | E4 对 & GRPO 错（**破坏**） | 133 | 6.65% |
| WC | E4 错 & GRPO 对（**修复**） | 154 | 7.70% |
| WW | 两者都错 | 348 | 17.40% |

校验：`GRPO acc − E4 acc = 0.7595 − 0.7490 = +0.0105 = (WC−CW)/N = (154−133)/2000 = 0.0105` ✓（数值完全一致）

### B.2 McNemar（exact binomial 为主，chi-square 连续校正为辅）

| Comparison | A-only correct (b) | B-only correct (c) | Δ Accuracy | Exact p | chi2 p | 结论 |
|---|---|---:|---:|---:|---:|---|
| SFT Epoch4 vs GRPO | 133 | 154 | +1.05pp | **0.2377** | 0.2378 | 不显著 |
| SFT Epoch5 vs GRPO | 159 | 128 | −1.54pp | **0.0764** | 0.0766 | 不显著（边缘趋势） |
| SFT Epoch4 vs SFT Epoch5 | 76 | 128 | +2.59pp | **0.0003** | 0.0004 | 显著 |

效应量解读：
- E4→GRPO：GRPO 修复 154 题、破坏 133 题，discordant 比 1.16:1，偏移幅度 +21 净修复，量级不足以在 2000 样本上达到显著。
- E5→GRPO：GRPO 相对 E5 破坏 159、修复 128，净 −31，discordant 比 0.81:1；p=0.076 说明存在弱于显著性的负向趋势。
- E4→E5：SFT 本身多训 1 epoch 的效果显著（净 +52），是当前唯一统计可信的模型间差异。

### B.3 Paired Bootstrap 95% CI（10000 次，seed 20260828，paired resample）

| 比较 | mean Δ | 95% CI | 是否跨 0 |
|---|---:|---|---|
| GRPO − Epoch4 | +1.05pp | [−0.60pp, +2.75pp] | 跨 0 |
| GRPO − Epoch5 | −1.54pp | [−3.25pp, +0.10pp] | 跨 0（上界贴 0） |
| Epoch5 − Epoch4 | +2.59pp | [+1.20pp, +4.00pp] | **不跨 0** |

结论：三个比较中只有 E5−E4 的 CI 不跨 0；GRPO 的两个比较均无法排除 0 效应。

## C. Correctness transition：WC vs CW

- WC（GRPO 修复）154：主要集中在 **NKK（+42 net）**、NKN（+15）、KKK（+10）、KKN（+8）。
- CW（GRPO 破坏）133：主要集中在 **KNN（−20 net）**、NNN（−17）、KNK（−14）、NNK（−3）。
- GRPO 的净提升几乎全部来自 **NKK 单类**（+42 净修复），而 KNN/NNN/KNK 三类净受损 −51，被其他类的小幅修复抵消。

## D. Which puzzles improved？（GT pattern × operator features）

### D.1 按 GT pattern（A→C 完整表见 `class_transition.csv`）

| GT Pattern | N | Epoch4 Acc | GRPO Acc | Δ | Fixed WC | Broken CW | Net |
|---|---:|---:|---:|---:|---:|---:|---:|
| NKK | 238 | 63.9% | 81.5% | **+17.6pp** | 47 | 5 | **+42** |
| NKN | 239 | 73.6% | 79.9% | +6.3pp | 27 | 12 | +15 |
| KKK | 274 | 90.5% | 94.2% | +3.6pp | 11 | 1 | +10 |
| KKN | 238 | 65.1% | 68.5% | +3.4pp | 31 | 23 | +8 |
| NNK | 251 | 78.1% | 76.9% | −1.2pp | 13 | 16 | −3 |
| KNK | 275 | 70.5% | 65.5% | −5.1pp | 17 | 31 | −14 |
| NNN | 252 | 87.7% | 81.0% | −6.7pp | 2 | 19 | −17 |
| KNN | 233 | 67.0% | 58.4% | **−8.6pp** | 6 | 26 | −20 |

→ **提升不是广泛分布，而是高度集中**：NKK 一类贡献了全部净增益；KNN/NNN/KNK 显著倒退。

### D.2 Operator features（WC vs CW vs WW vs CC，均值）

| Feature | WC（修复） | CW（破坏） | WW（都错） | CC（都对） |
|---|---:|---:|---:|---:|
| same_count | 1.82 | 1.56 | 1.47 | 1.63 |
| different_count | 1.79 | 1.74 | 1.61 | 1.50 |
| and_count | 1.01 | 0.94 | 1.01 | 0.75 |
| or_count | 1.17 | 0.88 | 0.99 | 0.80 |
| not_count | 0.90 | 0.85 | 0.80 | 0.79 |
| expression_nodes | 8.25 | 7.49 | 7.80 | 6.89 |
| expression_depth | 2.87 | 2.68 | 2.70 | 2.57 |
| statement_chars | 194.8 | 178.5 | 179.5 | 165.9 |

描述性观察（不宣称因果）：被修复的题结构上更复杂（更多节点/字符/or），被破坏的题略简单；两者之间结构区分不强，主要差异体现在 **GT 类别**（NKK 修复最多）而非 operator 特征。

## E. Which puzzles regressed？

- 类别上：KNN（−20）、NNN（−17）、KNK（−14）。
- 代表性破坏样本（完整 20 条见 `broken_samples.json`）：
  - `kk_grpo_v1_final_holdout_001457`（GT=NNN）：A/B 均对，GRPO 改成 NKN（Bob 由 knave 错成 knight）；heavy `and`/`different`。
  - `kk_grpo_v1_final_holdout_000466`（GT=NKN）：A/B 均对，GRPO 改成 NNK；含 2 个 `not`。
  - `kk_grpo_v1_final_holdout_001657`（GT=KKN）：A/B 均对，GRPO 改成 KKK；全 `or`/重言式结构。

## F. Policy polarization（200-prompt rollout，8 rollouts/prompt，temp=0.8）

采样：Final Holdout 前 200 prompts，Epoch4 与 GRPO 统一参数，seed 20260828。

| Rollout state transition | Count |
|---|---:|
| all-correct → all-correct | 72 |
| all-correct → mixed | 3 |
| mixed → all-correct | **38** |
| mixed → all-wrong | **17** |
| mixed → mixed | 57 |
| all-wrong → all-wrong | 10 |
| all-wrong → mixed | 3 |

| Aggregate | Epoch4 | GRPO |
|---|---:|---:|
| all-correct | 37.5% | 55.0% |
| mixed | 56.0% | 31.5% |
| all-wrong | 6.5% | **13.5%** |
| mean reward | 0.667 | 0.730 |
| avg unique answers | 1.83 | 1.39 |
| pass@8 | 93.5% | **86.5%** |

→ **明确验证**：GRPO 同时发生 mixed→all-correct（38）与 mixed→all-wrong（17），正确方向约是错误方向的 2.2 倍；总体分布从 mixed 移向两个极端（sharpening），all-wrong 翻倍，pass@8 反而下降（all-wrong 组增多导致“至少一个正确”的概率降低）。

## G. Prompt bias（KKN example 检查，Final Holdout）

GT KKN 频率 = 238/2000 = **11.9%**。

| Model | KKN pred freq | pred−GT diff | NKK pred freq | NKK pred−GT diff |
|---|---:|---:|---:|---:|
| SFT Epoch4 | 9.85% | −2.05pp | 10.35% | −1.55pp |
| SFT Epoch5 | 11.60% | −0.30pp | 12.35% | +0.45pp |
| GRPO | 10.50% | −1.40pp | **14.70%** | **+2.80pp** |

→ prompt 中的 `KKN` 格式示例**未形成 KKN prediction prior**（三个模型 KKN 均未系统性高估）。GRPO 实际把预测**偏向 NKK**（+2.8pp over GT），这是 RL 的 sharpening 行为而非格式示例造成的先验。仅作潜在 prior 描述，不宣称 leakage。

## H. Final conclusion

明确区分四件事：

1. **Pipeline success**：✅ GRPO 训练/评测/选择流程完整跑通（625 步、40,000 rollouts、best=ckpt-200）。
2. **Behavior change**：✅ 真实存在 —— reward 0.730→0.755（train）、all-correct 37.5%→55%（200 rollout）、unique answers 1.83→1.39，prediction 向 NKK 集中；同时 mixed→all-correct（38）与 mixed→all-wrong（17）双向发生。
3. **Statistical performance improvement**：❌ **不支持** —— GRPO vs E4 +1.05pp（McNemar p=0.238，bootstrap CI [−0.60,+2.75]pp 跨 0）；GRPO vs E5 −1.54pp（p=0.076，CI [−3.25,+0.10]pp 跨 0）。**在 2000 样本上无法证明 GRPO 带来显著提升**。
4. **Comparison against stronger SFT baseline**：GRPO 未超过 E5（holdout 76.0% vs 77.5%），且该差距同样未达显著，但方向不利；E5 相对 E4 的提升（+2.59pp，p=0.0003）是唯一统计可信的显著差异。

### Claim 判定

| Claim | 判定 | 依据 |
|---|---|---|
| A：GRPO 显著优于 SFT Epoch4 | **NOT_SUPPORTED** | +1.05pp；McNemar p=0.238；bootstrap CI 跨 0 |
| B：GRPO 显著弱于 SFT Epoch5 | **NOT_SUPPORTED** | −1.54pp；McNemar p=0.076（边缘趋势）；bootstrap CI 跨 0 |
| C：主要行为变化是 sharpening，且双向极化 | **SUPPORTED** | mixed→all-correct 38 vs mixed→all-wrong 17；all-correct 上升、unique 下降、pass@8 下降；prediction 向 NKK 集中 |

## 输出文件

```text
outputs/grpo_v1_analysis/paired_correctness.json
outputs/grpo_v1_analysis/mcnemar_results.json
outputs/grpo_v1_analysis/bootstrap_results.json
outputs/grpo_v1_analysis/class_transition.csv
outputs/grpo_v1_analysis/prediction_transition.csv
outputs/grpo_v1_analysis/feature_transition.json
outputs/grpo_v1_analysis/prompt_bias.json
outputs/grpo_v1_analysis/rollout_transition_200.json
outputs/grpo_v1_analysis/rollout_A_final200.jsonl
outputs/grpo_v1_analysis/rollout_C_final200.jsonl
outputs/grpo_v1_analysis/fixed_samples.json
outputs/grpo_v1_analysis/broken_samples.json
outputs/grpo_v1_analysis/grpo_v1_statistical_analysis.md
```

本轮未重新训练、未修改 checkpoint/reward/数据，未启动 GRPO-V2。
