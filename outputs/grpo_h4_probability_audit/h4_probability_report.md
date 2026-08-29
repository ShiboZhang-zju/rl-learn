# H4 报告：8-way answer-space probability landscape

日期：2026-08-29
类型：**零训练诊断**（teacher-forced scoring，无 sampling、无训练）
状态：`GRPO_H4_PROBABILITY_DIAGNOSIS_COMPLETE`
判定：**H4 = SUPPORTED**

---

## 核心结论（一句话）

**GRPO 的主要可观测作用更接近「在 8 个合法答案上重新分配并 sharpen 概率」，
而不是广泛扩大 gold answer 的 high-rank support。**

关键量化证据：Top1 gold coverage +2.85pp（CI 不跨 0），
而 **Top3 coverage +0.15pp（n.s.）**；
新增的 rank1 正确题中 **98.0% 原本就已排在 rank≤3**。
V2 的「greedy↑ 但 Pass@8↓」可由 **median gold_q↑ 而 p10 gold_q↓** 的尾部分化直接解释。

> 这个实验**不能**证明「GRPO 没有学到推理」。它只能说明：
> 当前观测到的 RL 行为，主要可由 answer-space 概率重分配解释。

---

## 0. 方法与校验

- 数据：**GRPO-V3 Fresh Holdout**，N=2000（seed 20260902）；
  另对固定 **200-prompt 诊断子集**单独打分，用于对接已有 rollout 分类。
- 模型：Epoch4 / Epoch5 / V1 / V2 / V3（**Epoch5 保留为「更多 SFT」对照**）。
- 不使用 temperature sampling。每题枚举 **8 个 canonical answer completion**，
  teacher-force 计算 completion-token 的 sequence log-prob，
  再用 logsumexp 在 8 个候选上归一化得到 `q`。
- Candidate 构造与 `kk_sft.data.format_answer_only_completion` **逐字节一致**
  （对全部 2000 题做了断言校验）。

### Scoring correctness gate（§8）

8-way argmax 与已有 greedy prediction 的一致率：

| Model | agreement | disagreements |
|---|---:|---:|
| Epoch4 | 0.9540 | 92 / 2000 |
| Epoch5 | 0.9685 | 63 / 2000 |
| V1 | 0.9775 | 45 / 2000 |
| V2 | **0.9830** | 34 / 2000 |
| V3 | 0.9790 | 42 / 2000 |

全部 ≥ 0.95 → **`H4_SCORING_VALID`**

### q 的含义（必须明确）

```text
q 是模型在 8 个 canonical legal answers 上重新归一化后的 answer-space distribution。
它不是模型整个语言生成空间中的绝对 probability。
```

### 已知 tokenization 偏差（不隐藏）

```text
"knight" -> 1 token，"knave" -> 2 tokens
KKK = 18 completion tokens，NNN = 21 tokens
```

sequence log-prob 因此带有**每个候选固定的长度偏移**，会把绝对 q 推向 knight 多的 pattern。
但该偏移对**所有模型完全相同**（同一 tokenizer、同样 8 个字符串），
在同一批样本上的**跨模型比较中会抵消**。

经验检验：`spearman(gold token 长度, gold_q 的 V2−E4 变化) rho = −0.0574`
（几乎无相关），且尾部崩塌样本在 8 个 pattern 上分布大致均匀
（KKK 1 / KKN 16 / KNK 9 / KNN 15 / NKK 22 / NKN 19 / NNK 19 / NNN 19，n=120）。
**长度偏差无法解释本报告的任何跨模型结论。**

敏感性检查（mean-token-logprob）见 §8 caveat：该归一化把分数压缩约 20 倍，
q 近乎均匀（normalized entropy 0.961–0.974），对尾部指标失效，
因此**不能用来否定**主结果；主结果仍按规格使用真实 sequence log-probability。

---

## Q1. GRPO 是否让 8-way distribution 更尖？**是**

| Metric | Epoch4 | Epoch5 | V1 | V2 | V3 |
|---|---:|---:|---:|---:|---:|
| normalized entropy | 0.2560 | 0.2013 | 0.1454 | **0.1208** | 0.1360 |
| effective support (e^H) | 1.8446 | 1.6288 | 1.4298 | **1.3527** | 1.4000 |
| mean top1 margin | 2.7796 | 3.5521 | 4.3461 | **5.0221** | 4.7697 |
| mean gold margin | 2.1645 | 2.8758 | 3.1908 | **3.7675** | 3.4599 |
| mean gold_q | 0.6780 | 0.7199 | 0.7283 | **0.7491** | 0.7276 |

Bootstrap（V2 − Epoch4，10000 次配对重抽样）：

```text
normalized_entropy  -0.13517  CI [-0.14135, -0.12885]   EXCL0
effective_support   -0.49193  CI [-0.51780, -0.46589]   EXCL0
top1_margin         +2.24329  CI [+2.15273, +2.33508]   EXCL0
gold_margin         (见 bootstrap_results.json)
```

有效支撑从 1.84 降到 1.35 —— 8 个候选里平均只有约 1.35 个仍有实质概率。

---

## Q2. gold answer 的 Top1 / Top2 / Top3 rank 如何变化？

| Coverage | Epoch4 | Epoch5 | V1 | V2 | V3 |
|---|---:|---:|---:|---:|---:|
| **Top1** | 0.7490 | 0.7740 | 0.7585 | **0.7775** | 0.7515 |
| **Top2** | 0.9240 | 0.9300 | 0.9175 | **0.9185** | 0.9035 |
| **Top3** | 0.9770 | 0.9780 | 0.9715 | **0.9785** | 0.9675 |

Bootstrap（V2 − Epoch4）：

```text
top1_coverage  +0.02841  CI [+0.01350, +0.04400]   EXCL0   <- 改善
top3_coverage  +0.00147  CI [-0.00500, +0.00800]   n.s.    <- 基本不动
```

**这是 H4 最核心的一条证据：Top1 明显改善，Top3 纹丝不动（Top2 甚至略降 0.55pp）。**
说明 gold 本来几乎就已经在候选集合前部（Epoch4 Top3 = 97.7%），
RL 做的是把 rank2/3 推上 rank1，**没有**把新的 gold 拉进高概率区。

---

## Q3. 新增正确题来自 rank2/3→rank1 还是 rank4+→rank1？

| Epoch4 → | promoted to rank1 | existing-support (rank≤3→1) | new-support (rank>3→1) | 占比 | demoted from rank1 |
|---|---:|---:|---:|---:|---:|
| Epoch5 | 123 | 121 | 2 | **98.4%** | 73 |
| V1 | 140 | 135 | 5 | **96.4%** | 121 |
| V2 | 152 | 149 | 3 | **98.0%** | 95 |
| V3 | 136 | 132 | 4 | **97.1%** | 131 |

**答：绝大多数（96–98%）来自 rank2/3→rank1。**
V2 从 rank>3 直接拉到 rank1 的只有 **3 道题**（占全部 promotion 的 2.0%）。

完整 rank 转移矩阵见 `rank_transition.json`。

---

## Q4. mixed→all-correct 与 mixed→all-wrong 是否都是 sharpening，只是方向不同？

在固定 200-prompt 诊断子集上，把 Epoch4 的 rollout 状态与 GRPO 状态配对，
看 teacher-forced 概率景观如何变化：

### Epoch4 → V2

| 转移 | n | gold_q | gold_rank | gold_margin | normalized entropy |
|---|---:|---:|---:|---:|---:|
| **mixed→all-correct** | 43 | 0.6173 → **0.9547** (+0.3375) | 1.233 → 1.000 | +0.760 → +3.908 | 0.3621 → **0.0865** (−0.2756) |
| **mixed→all-wrong** | 19 | 0.3133 → **0.0620** (−0.2513) | 2.000 → 2.737 | −0.679 → **−3.703** | 0.4388 → **0.2475** (−0.1912) |
| mixed→mixed | 50 | 0.5305 → 0.6031 (+0.0726) | 1.340 → 1.260 | +0.394 → +0.601 | 0.4014 → 0.3064 (−0.0951) |
| all-correct→all-correct | 75 | 0.9527 → 0.9852 | 1.000 → 1.000 | +5.414 → +8.218 | 0.0772 → 0.0238 |

**答：是。这是 H4 最直接的证据。**

两个方向的 **entropy 都下降**（−0.2756 与 −0.1912），
但概率质量朝**相反方向**集中：

- `mixed→all-correct`：gold_q ↑ +0.34，gold margin 转正并大幅拉开
- `mixed→all-wrong`：gold_q ↓ −0.25，**错误答案的 margin 从 −0.68 拉到 −3.70**（错误侧被 sharpen）

V1（mixed→all-correct +0.3034 / entropy −0.2464；mixed→all-wrong −0.2755 / −0.2023）
与 V3（+0.3205 / −0.2433；−0.2958 / −0.2214）呈现**完全相同的结构**。

---

## Q5. 为什么 V2 可以 greedy accuracy ↑ 同时 Pass@8 ↓？

### gold_q 的全部分位数

| Model | p10 | p25 | p50 | p75 | p90 |
|---|---:|---:|---:|---:|---:|
| Epoch4 | 0.1642 | 0.4376 | 0.7735 | 0.9721 | 0.9991 |
| Epoch5 | **0.1816** | 0.5000 | 0.8670 | 0.9928 | 0.9999 |
| V1 | 0.0474 | 0.4968 | 0.9422 | 0.9975 | 0.9999 |
| **V2** | **0.0450** | 0.5792 | **0.9698** | 0.9992 | 1.0000 |
| V3 | 0.0373 | 0.4999 | 0.9526 | 0.9987 | 1.0000 |

### 头部 / 尾部占比

| Model | gold_q < 0.01 | < 0.05 | < 0.10 | > 0.90 | > 0.99 |
|---|---:|---:|---:|---:|---:|
| Epoch4 | 1.30% | 3.30% | 6.30% | 35.60% | 19.85% |
| Epoch5 | 1.80% | 4.35% | 7.30% | 46.20% | 26.75% |
| V1 | 4.40% | 10.20% | 13.10% | 55.85% | 33.65% |
| **V2** | **5.45%** | **10.55%** | 13.40% | **60.35%** | **42.60%** |
| V3 | 5.95% | 10.85% | 13.85% | 56.95% | 36.90% |

**答：正是 §14 预测的「median↑ 而 p10↓」。**

```text
median gold_q  0.7735 -> 0.9698   中位数大幅上升（多数题更确定、更正确）
p10     gold_q  0.1642 -> 0.0450   下尾崩塌（少数困难题 gold 概率被压得更低）
gold_q < 0.05   3.30%  -> 10.55%   崩塌题占比翻了 3.2 倍
gold_q > 0.99  19.85%  -> 42.60%   头部更集中
```

多数题被推到「几乎确定正确」，少数题被推到「几乎确定错误」，
于是同时出现 `greedy↑ / all-correct↑ / all-wrong↑ / Pass@8↓ / avg unique↓`。

### 8-way implied Pass@8 与实际 rollout Pass@8

| Model | 8-way implied Pass@8 | actual rollout Pass@8 | spearman(gold_q, correct_fraction) |
|---|---:|---:|---:|
| Epoch4 | 0.9295 | 0.9350 | rho = 0.9323 (p≈2e-89) |
| V1 | 0.8825 | 0.8650 | rho = 0.8834 (p≈4e-67) |
| V2 | 0.8684 | 0.8400 | rho = 0.8757 (p≈2e-64) |
| V3 | 0.8525 | 0.8200 | rho = 0.9084 (p≈6e-77) |

```text
1 - (1 - q_gold)^8
```

**这不是严格的真实 Pass@8**，因为 q 只在 8 个 canonical answer 上归一化；
但 implied 值与实际值高度吻合（0.9295/0.9350 → 0.8684/0.8400 → 0.8525/0.8200），
且 gold_q 与 rollout 正确比例的相关系数达 **0.88–0.93**。
**answer-space 的 q 对 sampling 行为有很强解释力。**

---

## Q6. 这些现象是 GRPO-specific，还是 Epoch5 也会出现？

**分层回答：sharpening 本身是通用的，但造成 V2 悖论的那部分是 GRPO 特有的。**

### 通用部分（Epoch5 同样出现，方向一致但幅度更小）

| Metric | Epoch4→Epoch5 | Epoch4→V2 | V2−Epoch5 |
|---|---:|---:|---:|
| normalized entropy | −0.0547 EXCL0 | −0.1352 EXCL0 | **−0.0805 EXCL0** |
| effective support | −0.2159 EXCL0 | −0.4919 EXCL0 | **−0.2760 EXCL0** |
| top1 margin | +0.7726 EXCL0 | +2.2433 EXCL0 | **+1.4707 EXCL0** |
| top1 coverage | +0.0249 EXCL0 | +0.0284 EXCL0 | +0.0036 **n.s.** |
| existing-support promotion | 98.4% | 98.0% | — |

「更多训练 → 更尖、margin 更大、rank2/3→rank1」是**通用效应**。
但 GRPO 的 sharpening 幅度是 Epoch5 的 **2.5 倍**，且 V2−Epoch5 仍显著。

### GRPO 特有的部分（Epoch5 不出现甚至反向）

| Metric | Epoch4→Epoch5 | Epoch4→V1 | Epoch4→V2 | Epoch4→V3 |
|---|---:|---:|---:|---:|
| **p10 gold_q** | **+0.0174（改善）** | −0.1168 | **−0.1192** | **−0.1269** |
| **gold_q<0.05 占比** | +1.05pp | +6.90pp | **+7.25pp** | +7.55pp |
| **Top2 coverage** | **+0.60pp** | −0.65pp | **−0.55pp** | −2.05pp |
| Top3 coverage | +0.10pp n.s. | −0.55pp n.s. | +0.15pp n.s. | −0.95pp EXCL0 |
| demoted from rank1 | 73 | 121 | 95 | 131 |

**Epoch5 的下尾是改善的（p10 0.1642→0.1816），Top2 coverage 也是上升的。**
只有 GRPO 模型出现「下尾崩塌 + Top2 收窄」。

结论：

```text
sharpening（entropy↓ / margin↑ / rank2→1）      = 通用训练效应，GRPO 幅度更大
下尾崩塌 + Top2 coverage 收窄                    = GRPO-specific
后者正是 all-wrong↑ / Pass@8↓ 的直接来源
```

---

## Q7. H4 判定：SUPPORTED

| # | 判据 | 实测 | 结果 |
|---|---|---|---|
| 1 | entropy / effective support 明显下降 | 0.2560→0.1208；1.8446→1.3527（均 EXCL0） | ✅ |
| 2 | top1 margin 明显增大 | 2.7796→5.0221（+2.2433 EXCL0） | ✅ |
| 3 | Top1 改善但 Top2/Top3 提升很小 | Top1 +2.85pp EXCL0；Top2 −0.55pp；Top3 +0.15pp n.s. | ✅ |
| 4 | 新 correct 主要来自 rank2/3→rank1 | V2 149/152 = **98.0%**；rank>3→1 仅 3 题 | ✅ |
| 5 | mixed→all-correct：gold mass↑ + entropy↓ | gold_q +0.3375；entropy −0.2756 | ✅ |
| 6 | mixed→all-wrong：gold mass↓ + wrong margin↑ + entropy↓ | gold_q −0.2513；margin −0.679→−3.703；entropy −0.1912 | ✅ |
| 7 | V2 greedy↑/Pass@8↓ 可由尾部分化解释 | median↑0.7735→0.9698；p10↓0.1642→0.0450；implied Pass@8 0.9295→0.8684 贴合实际 0.9350→0.8400 | ✅ |

**七条判据全部成立 → `H4 = SUPPORTED`**

### 判定含义（严格限定）

```text
GRPO 的主要可观测作用更接近 answer-space probability
redistribution / sharpening，
而不是广泛扩大 gold answer 的 high-rank support。
```

**不要**写成：

```text
GRPO 没有学到推理。
```

本实验无法证明这一点——它只说明「当前观测到的 RL 行为主要可由概率重分配解释」。
Top1 accuracy 确实上升了（V2 +2.85pp 相对于 Epoch4 显著），
只是这个上升几乎全部来自把已有 rank2/3 候选推到 rank1。

---

## §16 概率质量流动（GT × candidate 的 8×8 矩阵）

平均 `q(candidate)`（全样本）：

| cand | E4 | E5 | V1 | V2 | V3 | V1−E4 | V2−E4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| KKK | 0.1293 | 0.1375 | 0.1446 | 0.1367 | 0.1492 | +0.0153 | +0.0074 |
| KKN | 0.1078 | 0.1269 | 0.1134 | 0.1286 | 0.1318 | +0.0056 | **+0.0208** |
| KNK | 0.1362 | 0.1394 | 0.1197 | 0.1439 | 0.1272 | −0.0165 | +0.0077 |
| KNN | 0.1306 | 0.1206 | 0.0975 | 0.1318 | 0.1058 | **−0.0330** | +0.0012 |
| NKK | 0.1000 | 0.1237 | 0.1382 | 0.1080 | 0.1062 | **+0.0382** | +0.0080 |
| NKN | 0.1369 | 0.1213 | 0.1343 | 0.1116 | 0.1100 | −0.0026 | **−0.0253** |
| NNK | 0.1315 | 0.1225 | 0.1361 | 0.1281 | 0.1409 | +0.0046 | −0.0034 |
| NNN | 0.1278 | 0.1081 | 0.1161 | 0.1114 | 0.1289 | −0.0116 | −0.0164 |

按 GT 条件化后（节选，完整 8×8 见 `probability_mass_by_class.json`）：

```text
V1:  GT=NKK 时 q(NKK) 0.526 -> 0.713 (+0.187)   <- NKK 偏移是连续概率质量移动
     GT=KNK 时 q(KNK) 0.681 -> 0.673 (-0.008)
     GT=KNN 时 q(KNN) 0.645 -> 0.597 (-0.048)

V2:  GT=KKN 时 q(KKN) 0.581 -> 0.721 (+0.141)
     GT=KNK 时 q(KNK) 0.681 -> 0.815 (+0.134)
     GT=KNN 时 q(KNN) 0.645 -> 0.724 (+0.079)
     GT=NKK 时 q(NKK) 0.526 -> 0.633 (+0.107)   <- 低于 V1 的 0.713，对应 V2 的 NKK 回退
```

**此前看到的 NKK / KKN / KNK / KNN 类别漂移，不是离散预测计数的抖动，
而是可以直接观测到的连续概率质量移动。**

---

## §8 Caveats

1. **mean-token-logprob 敏感性检查失效**：除以 token 数会把分数压缩约 20 倍，
   使 8-way q 近乎均匀（normalized entropy 0.961–0.974），
   `gold_q < 0.05` 这类尾部指标在所有模型上恒为 0。
   因此 mean-token 版本**无法证实也无法否证**尾部崩塌；
   主结果按规格使用真实 sequence log-probability。
   跨模型方向一致性：top1 / mean_gold_q / median / top3 / entropy 五个指标两种口径同向，
   仅 p10 与 tail<0.05 因上述退化而不可比。
2. **长度偏差已量化为可忽略**：`spearman(gold token 长度, gold_q 变化) rho = −0.0574`，
   且长度偏差对所有模型是常数，在同一批样本上的跨模型比较中抵消。
3. **8-way implied Pass@8 不是真实 Pass@8**（q 只在 8 个 canonical answer 上归一化）。
4. **Top1/Top2/Top3 coverage 按 rank 定义**，与 §10 的 `gold_rank_1 / ≤2 / ≤3` 是同一组量。
5. 200-prompt 诊断子集来自 `grpo_v1_final_holdout`（已非 untouched），
   仅用于对接已有 rollout 分类；总体结论以 N=2000 的 fresh holdout 为准。

---

## 输出文件

```text
scripts/audit_h4_probability_landscape.py

outputs/grpo_h4_probability_audit/
  candidate_tokenization.json        8 候选 pattern / completion / token_count
  sample_probability_landscape.jsonl 逐样本 5 模型完整 8-way scores 与 q
  sample_probability_landscape.csv   逐样本景观表
  aggregate_metrics.json             五模型聚合指标 + gate + sanity
  rank_transition.json               rank 转移矩阵 + existing/new support 拆分
  topk_coverage.json                 Top1/Top2/Top3 gold coverage
  rollout_probability_alignment.json implied Pass@8 / 相关性 / 状态条件景观
  probability_mass_by_class.json     平均 q 与 GT×candidate 8×8 矩阵
  bootstrap_results.json             10000 次配对 bootstrap（seed 20260830）
  smoke_check.json                   N=100 smoke 的 gate 与 sanity
  h4_probability_report.md           本报告
```

## 本轮禁止事项执行情况

```text
未启动 GRPO-V4/V5     未重新 SFT
未修改 reward         未修改 beta
未修改 generator      未生成新训练数据
未做 temperature sweep 未做超参数搜索
```
