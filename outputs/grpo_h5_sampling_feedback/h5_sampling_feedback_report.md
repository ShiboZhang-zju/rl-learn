# H5 报告：Finite-K On-Policy Sampling Feedback

日期：2026-08-29
类型：**零训练诊断**（静态 retrospective + 训练轨迹）
状态：`GRPO_H5_SAMPLING_FEEDBACK_DIAGNOSIS_COMPLETE`
判定：**H5 = SUPPORTED**（7 条判据中 6 条成立，1 条因数据不足无法检验）

---

## 核心结论（一句话）

**Finite-K on-policy sampling 与一条 path-dependent feedback 机制一致：
已获得支持的答案被优先强化，低支持的 gold answer 越来越难获得正强化。**

最有力的证据是 **bin 级 median delta 的 V2 vs Epoch5 对照**：
初始 gold_q ∈ [0.10,0.20) 的题，Epoch5 中位数 **+0.0266**、GRPO-V2 中位数 **−0.0896**
（差 −0.1187，CI [−0.1584, −0.0686]）；
而 [0.40,0.60) 的题 Epoch5 +0.0905、V2 **+0.2560**（差 +0.1650，CI 不跨 0）。

> **这不是因果结论。** 真正的因果实验需要 K 干预（K=8 vs 16/32），本轮禁止实施。
> 本轮提供的是 path-dependence 证据。

---

## 0. 概率定义（必须明确）

H4 的 `gold_q` 是 **8 个 canonical legal answer 上重新归一化**的概率，
不是完整生成空间中的真实 `P(generate exact correct completion)`。

因此 `1-(1-gold_q)^8` 只能叫 **8-way implied hit probability**。
本报告中：

```text
真实 hit / miss          -> 一律来自已保存 rollout 的 correct_count
理论 1-(1-q)^K           -> 仅用于解释与预测，不作为真值
```

实测两者关系：低 q 区真实 miss 比 implied 更严重
（[0.00,0.05) 桶：implied miss 87.4% vs 实际 miss **100%**），
因为自由生成还有格式/解析失败与非 canonical 输出。

---

## Q1. Initial gold_q 是否预测 K=8 hit/miss？**是**

200-prompt 诊断子集，Epoch4 policy，K=8，temperature 0.8，top_p 0.95：

```text
spearman(initial gold_q, correct_count)  rho = +0.9323
```

按初始 gold_q 分桶的真实 miss rate：

| bin | N | mean q0 | implied hit | **actual hit** | **actual miss** |
|---|---:|---:|---:|---:|---:|
| [0.00,0.05) | 6 | 0.0171 | 0.1258 | 0.0000 | **1.0000** |
| [0.05,0.10) | 9 | 0.0792 | 0.4794 | 0.3333 | **0.6667** |
| [0.10,0.20) | 8 | 0.1327 | 0.6704 | 1.0000 | 0.0000 |
| [0.20,0.40) | 23 | 0.3188 | 0.9412 | 0.9565 | 0.0435 |
| [0.40,0.60) | 34 | 0.5025 | 0.9950 | 1.0000 | 0.0000 |
| [0.60,0.80) | 37 | 0.7133 | 0.9999 | 1.0000 | 0.0000 |
| [0.80,1.00] | 83 | 0.9475 | 1.0000 | 1.0000 | 0.0000 |

整体：implied hit 0.9295 vs **actual hit 0.9350**（miss 6.50%）。

---

## Q2. Low-support prompts 是否更容易 miss？**是**

`gold_q < 0.10` 的 15 道题里 **10 道 miss（66.7%）**；
`gold_q > 0.20` 的 177 道题里只有 **1 道 miss（0.6%）**。

理论对照（K=8）：

```text
p=0.05 -> P(miss)=66.3%     p=0.10 -> 43.0%
p=0.20 -> 16.8%             p=0.30 ->  5.8%
```

实测与理论**同序**（低 q → 高 miss），且低 q 区实际 miss 比理论更高。

---

## Q3. Miss 是否只是 difficulty proxy？

### 未控制的 HIT / MISS 对比（N=200，13 miss）

| 指标 | HIT (n=187) | MISS (n=13) |
|---|---:|---:|
| mean initial gold_q | 0.6977 | **0.0657** |
| mean delta gold_q | **+0.0845** | **−0.0465** |
| P(final correct) | 0.8289 | **0.0000** |
| final all-wrong rate | 0.1016 | **1.0000** |
| final mean correct_count | 6.4545 | **0.0000** |

Bootstrap（HIT − MISS，10000 次）：
`+0.1304, CI [+0.0836, +0.1828]`，**不跨 0**。

### 但是：控制 initial gold_q 后**无法检验**

```text
[0.00,0.05)   n_hit=0,  n_miss=6    -> 无法比较
[0.05,0.10)   n_hit=3,  n_miss=6    -> 样本过少
其余 bin      n_miss=0
```

200 道题里只有 **13 道 MISS**，且几乎全部集中在最低两个桶。
**无法在相近初始概率下比较 HIT vs MISS 的未来轨迹。**

因此 Q3 的诚实回答是：

```text
MISS 强烈预测最终恶化（统计显著），
但 MISS 与 difficulty 在本数据中几乎完全共线，
无法分离「miss 是难度表现」与「miss 是独立反馈信号」。
```

这是本轮最大的证据缺口，也是不能把 H5 升级为因果结论的主要原因。

---

## Q4. 未来 all-correct / all-wrong 何时概率分叉？**step 100 就已明显**

200-prompt 轨迹（teacher-forced，7 个 checkpoint）：

| 未来状态 | step0 | 100 | 200 | 300 | 400 | 500 | 600 |
|---|---:|---:|---:|---:|---:|---:|---:|
| future all-correct (n=59) | 0.8305 | **0.8767** | 0.8796 | 0.9212 | 0.9336 | 0.9562 | **0.9741** |
| future all-wrong (n=32) | 0.2127 | **0.1826** | 0.2672 | 0.1423 | 0.0958 | 0.0971 | **0.0447** |

**step 100 时两组已经分离（0.8767 vs 0.1826）**，之后单调拉开。

轨迹分类（8-way top1 在 step0 / step600 的正确性）：

| 类别 | n | median gold_q by step（0→600） |
|---|---:|---|
| stable_correct | 135 | 0.8771 → … → **0.9924** |
| wrong→correct | 20 | 0.3452 → … → **0.8462** |
| correct→wrong | 13 | 0.5085 → … → **0.2128** |
| stable_wrong | 32 | 0.1139 → … → **0.0240** |

`spearman(early delta@100, final delta) = +0.3391`。

**注意**：`correct→wrong`（n=13）起始 median 0.5085，远高于风险区，仍崩塌了。
说明初始支持度不是唯一决定因素。

---

## Q5. rich-get-richer / poor-get-poorer？**是（中位数口径）**

### 一个必须先讲清楚的均值陷阱

**所有 bin 的 mean delta_gold_q 都是正的**，包括最低桶（+0.0646）。
这是因为每个低桶里都有一小部分题发生极大幅度跃升（0.02 → 0.9），
把均值拉正，而**多数题其实在下降**（最低桶 62.1% 下降）。

所以本轮以 **median / 下降比例 / 低尾占比** 为准，均值仅作对照。

### bin 级 V2 vs Epoch5（10000 次 bootstrap）

| bin | N | median Δ V2 | median Δ E5 | **median 差 (V2−E5)** | tail<0.05 V2 | tail<0.05 E5 | **tail 差** |
|---|---:|---:|---:|---:|---:|---:|---:|
| [0.00,0.05) | 66 | −0.0028 | −0.0001 | −0.0029 [−0.0082,−0.0002] | 0.773 | 0.727 | +0.046 n.s. |
| [0.05,0.10) | 60 | −0.0499 | −0.0109 | **−0.0397 [−0.0543,−0.0288]** | 0.667 | 0.383 | **+0.282 [+0.150,+0.417]** |
| [0.10,0.20) | 118 | −0.0896 | +0.0266 | **−0.1187 [−0.1584,−0.0686]** | 0.449 | 0.110 | **+0.339 [+0.254,+0.432]** |
| [0.20,0.40) | 201 | −0.0177 | +0.0480 | −0.0545 n.s. | 0.199 | 0.010 | **+0.189 [+0.134,+0.244]** |
| [0.40,0.60) | 270 | +0.2560 | +0.0905 | **+0.1650 [+0.0984,+0.2172]** | 0.074 | 0.004 | +0.071 |
| [0.60,0.80) | 332 | +0.2143 | +0.0939 | **+0.1181 [+0.0996,+0.1347]** | 0.012 | 0.000 | +0.012 |
| [0.80,1.00] | 953 | +0.0106 | +0.0032 | +0.0071 | 0.003 | 0.000 | +0.003 |

**读法**：低支持桶（0.05–0.20）V2 中位数显著为负、Epoch5 为正 → GRPO 特有的恶化；
中高支持桶（0.40–0.80）V2 中位数增益是 Epoch5 的 **2–3 倍** → GRPO 特有的强化。

### 四分位轨迹（200-prompt）

| step | low-quartile median | high-quartile median | frac q<0.05 | median entropy |
|---:|---:|---:|---:|---:|
| 0 | 0.2112 | 0.9980 | 3.0% | 0.2968 |
| 100 | 0.1576 | 0.9993 | 7.5% | 0.1680 |
| 200 | 0.2415 | 0.9995 | 7.0% | 0.1250 |
| 300 | 0.1188 | 0.9998 | 11.0% | 0.1258 |
| 400 | 0.0759 | 0.9999 | 12.0% | 0.1055 |
| 500 | 0.0973 | 0.9999 | 10.5% | 0.1214 |
| 600 | **0.0952** | **0.9999** | **11.5%** | 0.0829 |

低四分位中位数 **0.2112 → 0.0952**，低尾占比 **3.0% → 11.5%**，熵单调下降。

---

## Q6. 是否明显强于 Epoch5 SFT control？**是**

整体低尾（`gold_q < 0.05` 占比）：

```text
Epoch4  3.30%
Epoch5  4.35%      <- 多训一个 epoch 的 SFT，仅 +1.05pp
V2     10.55%      <- GRPO，+7.25pp，是 Epoch5 的 6.9 倍
```

p10 gold_q：

```text
Epoch4 0.1642   Epoch5 0.1816（改善）   V2 0.0450（崩塌）
```

每桶「gold_q 下降比例」：

| bin | V2 下降% | E5 下降% | 差 |
|---|---:|---:|---:|
| [0.00,0.05) | 62.1 | 51.5 | +10.6 |
| [0.05,0.10) | 70.0 | 61.7 | +8.3 |
| [0.10,0.20) | 66.1 | 44.1 | **+22.0** |
| [0.20,0.40) | 51.2 | 39.3 | +11.9 |
| [0.60,0.80) | 15.7 | 27.1 | −11.4 |
| [0.80,1.00] | 9.9 | 17.9 | −8.0 |

**低桶 GRPO 下降比例明显高于 Epoch5，高桶明显低于 Epoch5** ——
这正是「富者愈富 / 穷者愈穷」的方向性特征，且是 GRPO 特有的。

---

## Q7. H5 判定：SUPPORTED（6/7，1 条无法检验）

| # | 判据 | 实测 | 结果 |
|---|---|---|---|
| 1 | Initial gold_q 强烈预测真实 K=8 hit/miss | spearman = **+0.9323** | ✅ |
| 2 | Low-support prompts 明显更高 miss rate | q<0.10：10/15 miss；q>0.20：1/177 miss | ✅ |
| 3 | 初始 MISS 最终更容易 gold_q 下跌 / stable wrong | Δ −0.0465 vs +0.0845（差 +0.1304 EXCL0）；P(correct\|MISS)=0.000；all-wrong 100% vs 10.2% | ✅（但 n=13 且与难度共线） |
| 4 | 控制 initial gold_q 后 HIT/MISS 仍预测未来 | **每桶 MISS ≤6、HIT ≤3，无法检验** | ⚠️ 未检验 |
| 5 | step100 前后概率分叉 | 0.8767 vs 0.1826（step0 时 0.8305 vs 0.2127） | ✅ |
| 6 | 高 support 上升 / 低 support 恶化 | low-quartile median 0.2112→0.0952；低尾 3.0%→11.5%；[0.40,0.80) median +0.21~+0.26 | ✅ |
| 7 | 明显强于 Epoch5 control | 低尾 +7.25pp vs Epoch5 +1.05pp；[0.10,0.20) median 差 −0.1187 EXCL0；tail 差 +0.339 EXCL0 | ✅ |

**6/7 成立，1 条数据不足 → `H5 = SUPPORTED`**

结论表述（严格按规格）：

```text
Finite-K on-policy sampling is consistent with a
path-dependent feedback mechanism that preferentially
reinforces already-supported answers and leaves low-support
gold answers increasingly unlikely to receive positive
reinforcement.
```

**不要**写成：

```text
K=8 causes polarization
```

---

## Empirical low-support risk region（§9）

不主张数学相变，只描述经验风险区。

**初始 gold_q（Epoch4）→ 最终落入 q<0.05 的比例：**

| 初始 bin | N=2000 轨迹外的静态统计 | 200-prompt 轨迹统计 |
|---|---:|---:|
| [0.00,0.05) | 77.3% | 100% (6/6) |
| [0.05,0.10) | 66.7% | 66.7% (6/9) |
| [0.10,0.20) | 44.9% | 62.5% (5/8) |
| [0.20,0.40) | 19.9% | 17.4% (4/23) |
| [0.40,0.60) | 7.4% | 5.9% (2/34) |
| [0.60,0.80) | 1.2% | 0% |
| [0.80,1.00] | 0.3% | 0% |

**经验风险区 ≈ 初始 gold_q < 0.20**：

```text
在该区间：
  - 实际 miss rate 明显上升（[0,0.10) 达 66.7%）
  - 44.9%~77.3% 的题最终落入 q<0.05
  - V2 median delta 转为负值，而 Epoch5 为正

在 gold_q > 0.40 之后：
  - 落入低尾的比例 < 8%
  - V2 median delta 强正（+0.21~+0.26）
```

---

## 未完成的 B3（如实报告）

§14 要求的「gold_q(t) → correct_count(t) → gold_q(t+1)」逐步对齐**未能完成**：

```text
V2 probe rollouts 的 20 个 prompt 取自 v2_answer_only_val.jsonl，
而 200-prompt 诊断子集取自 grpo_v1_final_holdout，
两者 id 无交集 -> by_step 为空。
```

本轮不为此重新推理或重新训练。因此逐 step 的
「HIT/MISS → 下一步 gold_q 变化」这一最接近 feedback loop 的直接证据**缺失**。

---

## 输出文件

```text
scripts/audit_h5_sampling_feedback.py

outputs/grpo_h5_sampling_feedback/
  initial_goldq_bins.csv      A1 按初始 gold_q 分桶的初/终状态
  hit_miss_analysis.csv       A3 真实 K=8 hit/miss 按桶统计
  controlled_hit_miss.csv     A5 控制初始概率后的 HIT/MISS（多数桶样本不足）
  tipping_region.csv          A6 经验风险区
  trajectory_200.jsonl        B 逐样本 7-checkpoint 概率轨迹
  trajectory_summary.csv      B1 四类轨迹的逐步统计
  early_divergence.json       B2 早期分叉
  epoch5_control.json         Epoch5 静态对照
  bootstrap_results.json      10000 次 bootstrap（含 bin 级 V2 vs Epoch5）
  h5_analysis.json            A2/A3/A4/A5 + B1/B3/B4 汇总
  h5_sampling_feedback_report.md
```

## 本轮禁止事项执行情况

```text
未启动新的 GRPO        未改变 K (仍为 8)
未改变 reward          未改变 beta
未改变 LR              未做 temperature sweep
未重新生成训练数据     未训练新 SFT
```

---

## Causal experiment warranted? **YES**

H5 = SUPPORTED，且判据 7（GRPO-specific vs Epoch5）有统计显著支持，
因此可以设计 K 干预实验。但**本轮不实施**，且必须注意：

```text
K=8 vs K=16/32 对照必须固定：
  same SFT init (checkpoint-1252)
  same exact reward
  same beta = 0.01
  same LR
  same data
  same seed strategy
  same number of unique prompts
```

以及**最关键的设计约束**：

```text
增加 K 会增加 rollout compute。
必须同时讨论并预注册两种对齐方式：

  (a) equal optimizer steps
      -> K 大的组看到 2x/4x 的 rollout
  (b) equal total rollout budget
      -> K 大的组 optimizer steps 减少一半/四分之一

否则无法区分「K 更大」与「计算量更大」。
```

此外，考虑到本轮判据 4 无法检验（MISS 与难度共线），
K 干预实验还应考虑加入「按难度分层」的分析设计。
