# H5 Causal Intervention: K=8 vs K=16

日期：2026-08-29
设计预注册：`outputs/grpo_k16_intervention/preregistered_design.md`（训练前创建）
判定：**H5_CAUSAL_NOT_SUPPORTED**

---

## 核心结论（一句话）

**把 K 从 8 加倍到 16 确实显著提高了 low-support gold 的 sampling coverage，
但没有缓解 lower-tail collapse —— 在初始 gold_q < 0.20 的区域反而显著更差。**

```text
Q1 sampling coverage:  确实提高（训练期 all-wrong 0.1050 -> 0.0788，相对 -25%）
Q2 lower-tail collapse:未缓解，低支持区显著更差
Q3 p10 / q<0.05:        无改善（n.s.，点估计略差）
Q4 common K_eval=8:     Pass@8 +0.5pp / all-wrong -0.5pp，均在噪声内；
                        mixed→all-wrong 19 -> 19，完全没有变化
```

---

## 0. 前置 gate（全部通过）

| Gate | 结果 | 依据 |
|---|---|---|
| 历史 K8 可否作因果 control | **HISTORICAL_K8_CONTROL_VALID** | 见 §1 |
| Sampler alignment | **SAMPLER_ALIGNED** | 800 prompts 顺序完全一致 |
| K16 smoke @128 completions/step | **通过**（峰值 45.7GB，无 OOM） | 未改动 prompt_batch_size / grad accumulation / grad checkpointing / max_completion |

## 1. 历史 K8 control 有效性

`git diff 87762e2..HEAD -- scripts/train_grpo_trl.py src/kk_sft/reward.py` → 199 insertions / 10 deletions，全部属于：

```text
- 日志与指标字段新增（dual-track metrics、rescue 统计）
- 采集钩子 on_step_end -> on_log
- 输出文件名改为通用名
- 注释
- reward 重构为 kk_sft.reward.compute_reward，exact 模式行为等价
```

唯一触及训练信号的改动：

```python
- reward = float(reward_cfg.get("correct", 1.0)) if correct else 0.0
+ reward = compute_reward(parsed.parsed, answer[j], pz["people"], reward_mode)
```

`mode="exact"` 时 `compute_reward` 返回 `float(parsed == gt)`，与旧式在 `correct: 1.0` 下逐位相同，
并由单元测试锁定（`test_exact_reward_is_binary` / `test_default_mode_is_exact` / `test_parse_failure_is_zero_in_both_modes`）。

**数值验证**：用 V2 时期的脚本版本与当前脚本各跑 2 步（同 config、同 seed），
两者 step1/step2 的 `loss / grad_norm / kl / entropy / reward_mean / num_tokens` **逐位一致**
（0.0004 / 0.547752 / 3.4e-05 / 0.013108 / 0.75 / 8277），证明训练语义未变。

**必须记录的噪声下限**：V2 原始记录与这两次重跑在 token 数上差 1（8276 vs 8277），
进而导致 loss/grad/kl 在 1e-4 量级不同。即 **GRPO rollout 采样在本加速器上不是 bit-reproducible**，
本轮所有 K8 vs K16 差异都必须放在这个噪声下限之上解读。

---

## Q1. K16 是否真的增加 low-support gold 的 sampling opportunity？**是**

理论（K=8 → K=16）：

| p | P(miss) K=8 | P(miss) K=16 | 覆盖率提升 |
|---:|---:|---:|---:|
| 0.05 | 66.34% | 44.01% | +22.3pp |
| 0.10 | 43.05% | 18.53% | **+24.5pp** |
| 0.15 | 27.25% | 7.43% | +19.8pp |
| 0.20 | 16.78% | 2.81% | +13.9pp |
| 0.30 | 5.76% | 0.33% | +5.4pp |

实测（训练期 group 指标，前 100 步）：

| 指标 | K8 | K16 | 变化 |
|---|---:|---:|---|
| exact all-wrong | 0.1050 | **0.0788** | **−25% 相对** |
| 零方差组占比 | 0.6200 | **0.5563** | −6.4pp |
| mixed 组占比 | 0.3800 | **0.4437** | +6.4pp |
| exact all-correct | 0.5150 | 0.4775 | −3.8pp |

**干预在机制层面确实生效了**：更多 low-support group 至少采到一次正确答案。

> 按预注册，训练期 group 指标只作 process diagnostics：
> K=16 时 `P(all-wrong)` 机械下降、`Pass@16` 与 `Pass@8` 不可比，
> 它们**不作为** H5 证据。

---

## Q2 & Q3. Lower-tail collapse 是否被缓解？**没有，且低支持区显著更差**

Fresh K-intervention Holdout（N=2000，seed 20260903，与全部历史数据零重叠）：

| Model | Top1 | mean gold_q | **p10 gold_q** | **frac q<0.05** | Top3 | norm. entropy |
|---|---:|---:|---:|---:|---:|---:|
| Epoch4 | 0.7525 | 0.6720 | 0.1808 | 0.0330 | 0.9810 | 0.2622 |
| Epoch5 | 0.7795 | 0.7199 | 0.1824 | 0.0430 | 0.9780 | 0.2077 |
| **K8 (V2, ckpt-600)** | 0.7710 | 0.7492 | **0.0534** | **0.0960** | 0.9740 | 0.1235 |
| **K16 (ckpt-625)** | 0.7745 | 0.7539 | **0.0530** | **0.0995** | 0.9740 | 0.1149 |
| K16 (ckpt-600，同 step 对照) | 0.7570 | 0.7447 | **0.0473** | **0.1050** | 0.9735 | 0.1202 |

### Primary bootstrap（10000 次配对重抽样，seed 20260903）

| Endpoint | K16 − K8 | 95% CI | 判定 |
|---|---:|---|---|
| **p10 gold_q** | −0.00495 | [−0.02279, +0.01292] | **n.s.** |
| **frac gold_q < 0.05** | +0.00336 | [−0.00750, +0.01400] | **n.s.**（点估计更差） |
| mean gold_q | +0.00470 | [−0.00390, +0.01351] | n.s. |
| normalized entropy | **−0.00864** | [−0.01331, −0.00388] | **EXCL0（K16 更尖）** |
| Top1 accuracy | +0.00356 | [−0.01000, +0.01750] | n.s. |

### 低支持区（初始 gold_q < 0.20，n=217）—— 最关键的反向结果

| 指标 | K8 | K16 | K16 − K8 | 95% CI |
|---|---:|---:|---:|---|
| median Δgold_q | −0.02208 | **−0.03634** | **−0.01355** | [−0.02760, −0.00000] **EXCL0** |
| **frac 最终 gold_q < 0.05** | **0.58986** | **0.67281** | **+0.08295** | [+0.02304, +0.14747] **EXCL0** |

（bootstrap 均值 +0.08336；逐样本精确差 +0.08295。）

**K16 在低支持区不是"没帮上忙"，而是显著更差。**

按初始 gold_q 分层（fresh holdout）：

| 初始 bin | N | median Δ K8 | median Δ K16 | q<0.05 K8 | K16 | Top1 K8 | K16 |
|---|---:|---:|---:|---:|---:|---:|---:|
| [0.00,0.05) | 66 | −0.0100 | −0.0090 | 0.864 | **0.803** | 0.061 | 0.091 |
| [0.05,0.10) | 56 | −0.0532 | **−0.0582** | 0.625 | **0.732** | 0.161 | **0.107** |
| [0.10,0.20) | 95 | −0.0662 | **−0.0983** | 0.379 | **0.547** | 0.221 | **0.179** |
| [0.20,0.40) | 229 | **+0.0975** | +0.0233 | 0.153 | 0.153 | 0.454 | 0.415 |
| [0.40,0.60) | 307 | +0.1901 | **+0.2552** | 0.062 | **0.033** | 0.668 | 0.691 |
| [0.60,0.80) | 318 | +0.2110 | +0.2201 | 0.022 | 0.019 | 0.890 | **0.931** |
| [0.80,1.00] | 929 | +0.0145 | +0.0160 | 0.003 | 0.002 | 0.986 | 0.987 |

```text
K16 对 low-support region 的作用是非单调的：
极低支持 [0.00,0.05) 略有 rescue，
但 [0.05,0.20) 明显恶化，
使 <0.20 聚合结果整体显著更差。
```

具体地：

```text
[0.00,0.05)  N=66   medΔ −0.0100 -> −0.0090   q<.05 0.864 -> 0.803   K16 略好
[0.05,0.10)  N=56   medΔ −0.0532 -> −0.0582   q<.05 0.625 -> 0.732   K16 更差
[0.10,0.20)  N=95   medΔ −0.0662 -> −0.0983   q<.05 0.379 -> 0.547   K16 更差
```

中高桶（≥0.20）K16 略好。**整体而言 K16 没有缩小两极，反而使低支持区聚合显著恶化。**

---

## Q4. 统一 K_eval=8 下 Pass@8 / all-wrong 是否改善？**基本没有**

固定 200-prompt 诊断子集，`K_eval = 8, temperature 0.8, top_p 0.95, max_new_tokens 64`：

| Metric | Epoch4 | K8 (V2) | **K16** |
|---|---:|---:|---:|
| Pass@8 | 0.9350 | 0.8400 | **0.8450** |
| all-correct | 0.3750 | 0.5900 | **0.6100** |
| **all-wrong** | 0.0650 | 0.1600 | **0.1550** |
| mixed | 0.5600 | 0.2500 | **0.2350** |
| avg correct/group | 5.335 | 6.035 | 5.840 |
| avg unique answers | 1.825 | 1.335 | 1.300 |

差异规模：Pass@8 +1 题、all-wrong −1 题、all-correct +4 题（n=200）—— 全部在噪声内。

### 状态转移（Epoch4 → GRPO）

| 转移 | K8 | **K16** |
|---|---:|---:|
| mixed → all-correct | 43 | 50 |
| **mixed → all-wrong** | **19** | **19** |
| mixed → mixed | 50 | 43 |
| all-correct → all-correct | 75 | 71 |
| all-wrong → all-wrong | 13 | 12 |

**`mixed → all-wrong` 一位不差地保持 19。** 这正是 H5 声称应当减少的量，K16 完全没有改变它。

---

## Q5. Equal-step comparison（相同 optimizer steps / prompt 覆盖）

Sampler alignment 已通过：同 step 下两组看到相同顺序的 8 个 unique prompts，
唯一区别是每题 8 vs 16 rollout。200-prompt 子集：

| Step | K8 p10 | K16 p10 | K8 q<.05 | K16 q<.05 | K8 Top1 | K16 Top1 | K8 low-medΔ | K16 low-medΔ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 (E4) | 0.1316 | — | 0.0300 | — | 0.7400 | — | — | — |
| 100 | 0.0605 | 0.0601 | 0.0750 | 0.0850 | 0.7200 | **0.7500** | −0.0197 | −0.0334 |
| 200 | 0.0876 | **0.0993** | 0.0700 | 0.0700 | 0.7500 | 0.7500 | **+0.0004** | −0.0129 |
| 300 | 0.0331 | **0.0745** | 0.1100 | **0.0950** | 0.7650 | **0.7700** | −0.0372 | **−0.0215** |
| 400 | 0.0219 | **0.0606** | 0.1200 | **0.0950** | 0.7550 | **0.7800** | −0.0373 | −0.0372 |
| 500 | 0.0198 | 0.0199 | **0.1050** | 0.1200 | 0.7550 | 0.7550 | −0.0459 | **−0.0372** |
| 600 | **0.0341** | 0.0178 | **0.1150** | 0.1300 | **0.7750** | 0.7400 | −0.0373 | **−0.0215** |

**结论：中段（300/400）K16 的 p10 更好，但到 500/600 反转；n=200 噪声大，
且 N=2000 的 fresh holdout 上 K16 的低支持区显著更差。不能据此支持 K16。**

## Q6. Equal-rollout-budget comparison

| 对照 | K16 p10 | K8 p10 | K16 q<.05 | K8 q<.05 | K16 low-medΔ | K8 low-medΔ | 倾向 |
|---|---:|---:|---:|---:|---:|---:|---|
| K16-100 vs K8-200 | 0.0601 | **0.0876** | 0.0850 | **0.0700** | −0.0334 | **+0.0004** | **K8** |
| K16-200 vs K8-400 | **0.0993** | 0.0219 | **0.0700** | 0.1200 | **−0.0129** | −0.0373 | **K16** |
| K16-300 vs K8-600 | **0.0745** | 0.0341 | **0.0950** | 0.1150 | **−0.0215** | −0.0373 | **K16** |

3 组中 2 组倾向 K16，1 组倾向 K8 —— **同样不一致**，且同为 n=200。

由于 equal-step 的 N=2000 主终点已经明确为"无改善/更差"，
这里不构成 COMPUTE_CONFOUNDED（那要求 equal-step 下 K16 明显更好）。

## Q7. Exact accuracy

Fresh holdout（N=2000）：

```text
K8  (V2, ckpt-600)  0.7710
K16      (ckpt-625) 0.7745      +0.35pp
McNemar: b=92 (K8 only), c=99 (K16 only), exact p = 0.6643  -> n.s.
```

V2 Val 曲线（仅用于选 checkpoint，不作结论依据）：

```text
K8 : 0.7220 0.7540 0.7540 0.7660 0.7600 0.7420 0.7720 0.7620   (step 0..625)
K16: 0.7220 0.7540 0.7400 0.7580 0.7560 0.7260 0.7620 0.7680
```

---

## Q8. 判定：H5_CAUSAL_NOT_SUPPORTED

| # | 判据 | 实测 | 结果 |
|---|---|---|---|
| 1 | 低支持区 lower-tail collapse 显著缓解 | 显著**更差**（median Δ −0.0136 EXCL0；q<0.05 +0.0834 EXCL0） | ❌ 反向 |
| 2 | gold_q<0.05 比例显著下降 | +0.0034，CI 跨 0 | ❌ |
| 3 | p10 gold_q 明显提高 | −0.0049，CI 跨 0 | ❌ |
| 4 | common K_eval=8：all-wrong↓ / Pass@8↑ | all-wrong −0.5pp、Pass@8 +0.5pp（n=200，噪声内）；mixed→all-wrong 19→19 | ❌ |
| 5 | equal-step comparison 支持 | 中段有利、后段反转；N=2000 主终点反向 | ❌ |
| 6 | equal-budget comparison 不反转 | 2/3 组倾向 K16、1/3 倾向 K8，不一致 | ⚠️ |

```text
H5_CAUSAL_NOT_SUPPORTED
```

按规格原文的情形：

```text
K16 明显提高 sampling coverage
但 lower tail / p10 / common-eval all-wrong / Pass@8 均没有改善
→ 之前的 H5 相关性更多反映 difficulty，而不是 K 的因果作用。
```

---

## 一个重要的机制观察（不是结论，是线索）

K16 虽然采到了更多 low-support correct，但**整体 sharpening 更强**：

```text
normalized entropy:  K16 − K8 = −0.00864  CI [−0.01331, −0.00388]  EXCL0
KL mean:             K8 0.00908  vs  K16 0.01544
Top1 accuracy:       几乎不变（+0.35pp，n.s.）
```

### 对 coverage 结论的严格限定

```text
Increasing group size from K=8 to K=16 successfully increased
rollout coverage, but this intervention did not mitigate the
observed lower-tail collapse. Therefore finite-K miss is not a
sufficient explanation of polarization, and there is no evidence
from this intervention that it is the dominant bottleneck.
```

**同时必须注明：**

```text
Changing K also changes group-normalized advantage geometry,
so K16 is not a pure coverage-only intervention.
```

即 K16 不是一个"只改 coverage"的干净干预：在 `scale_rewards="group"` 下，
改变 K 会同时改变 group mean / sample std / 每个 rollout 的 normalized advantage
（见下一轮 H6）。因此本轮结果只能说明
**finite-K miss 不是充分解释**，不能断言 coverage 完全无关。

---

## 输出文件

```text
configs/grpo_k16_intervention.yaml
scripts/audit_k16_sampler_alignment.py
scripts/generate_k16_holdout.py
scripts/audit_k16_probability.py
scripts/analyze_k16_causal.py

outputs/grpo_k16_intervention/preregistered_design.md
outputs/grpo_k16_intervention/audit/sampler_alignment.json
outputs/grpo_k16_intervention/          (训练产物，best = checkpoint-625)
outputs/grpo_k16_intervention_smoke/
outputs/grpo_k16_analysis/
  diag200_probability.json
  fresh_holdout_probability.json
  rollout_F_final200.jsonl               (K16, K_eval=8)
  equal_step_comparison.csv
  equal_budget_comparison.csv
  stratified_low_support.csv
  common_k8_behavior.json
  bootstrap_results.json
  k16_causal_report.md

data/raw_grpo_k16/final_holdout.jsonl
data/processed/grpo_k16_final_holdout.jsonl
data/processed/grpo_k16_manifest.json     (与全部历史数据重叠 = 0)
```

## 本轮禁止事项执行情况

```text
未跑 K=32          未改 reward（mode 仍为 exact）
未改 beta (0.01)   未改 LR (1e-5)
未改 init          未改训练数据
未改 epoch (1)     未改 generator
未改 prompt_batch_size / grad accumulation /
    gradient checkpointing / max_completion_length
```
