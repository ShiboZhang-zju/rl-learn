# H3 诊断报告：GRPO 是否放大 structure/operator → answer-pattern shortcut

日期：2026-08-29
类型：**纯诊断，零训练**
状态：`GRPO_H3_DIAGNOSIS_COMPLETE`
判定：**H3 = NOT_SUPPORTED**

---

## 核心结论（一句话）

**没有证据表明 GRPO 在强化 structure/operator shortcut。**

数据里确实存在残余的 structure→GT 相关性，但很弱；
换成模型预测后，这种相关性**没有随 GRPO 增强**，
控制 GT 后 GRPO-V2 反而**显著下降**，且 GRPO 与「只是把 SFT 多训一个 epoch」(Epoch5) 无法区分。

---

## 数据与一致性

- 分析集：**GRPO-V3 Fresh Holdout**，N=2000，seed 20260902，与全部历史数据零重叠
- 模型（全部复用已有逐样本预测，**未重新推理**）：
  `Epoch4 / Epoch5 / GRPO-V1 / GRPO-V2 / GRPO-V3`
- 一致性校验：五个模型 N 均为 2000，**id 完全一致**、**GT 完全一致**、无重复 → 可配对比较

## 结构特征（复用 `scripts/audit_dataset_features.row_features`，未新增特征工程）

| 特征 | 说明 | 基数 |
|---|---|---:|
| `same_count` / `different_count` / `and_count` / `or_count` / `not_count` / `person_is_count` | 表达式树中各算子出现次数 | 3–8 |
| `expression_nodes` / `expression_depth` | 表达式规模 / 深度 | 15 / 3 |
| `top_ops` | 按 Alice/Bob/Carol 的 statement 顶层算子序列 | **211** |
| `top_ops_multiset` | `top_ops` 的算子多重集（去掉 speaker↔operator 配对） | 55 |
| `op_signature` | 整棵表达式树的算子多重集 | **582** |

### 关于 `top_ops` 基数的一个硬限制

规格要求按 Alice/Bob/Carol 的顶层算子序列建 signature。但 N=2000 下它实现了 **211 个取值**
（平均 9.5 样本/类）：

```text
top_ops         : N>=20 的 signature 只有 5 个，覆盖 5.6% 数据
top_ops_multiset: N>=20 的 signature 有 38 个，覆盖 91.8% 数据
```

因此 signature 级表格（§7/§8/§9）改用更粗的 `top_ops_multiset`（仍为纯结构特征，
只丢弃 speaker↔operator 的配对）。`top_ops` 结果一并保留并标注为低覆盖。

同理，§7 要求的「每个 (GT, signature) 单元格 N>=20」在 `top_ops_multiset` 下
只有 **5 个**单元格达标（8 类 × 38 signature，平均每格 ~6.5 样本），
因此 GT 控制结论**以 conditional MI（按 GT 分层加权合并，用满 2000 样本）为准**，
而非逐格准确率表。

---

## Q1. 数据中还有多少 structure→GT shortcut？

MI 单位为 nats。`null` 列为 permutation 打乱 prediction 1000 次得到的零分布均值，
由于 `MI(特征; GT)` 与 `MI(特征; prediction)` 的 (Kx, Ky, N) 相同，
该 null 可作为 plug-in 估计量偏差的下限（一阶近似）。

| 特征 | 基数 | MI(特征;GT) | null（偏差下限） | 超出 null | MI / H(GT) |
|---|---:|---:|---:|---:|---:|
| `same_count` | 8 | 0.0697 | 0.0133 | **+0.0564** | 3.35% |
| `different_count` | 7 | 0.0608 | 0.0113 | **+0.0494** | 2.92% |
| `or_count` | 7 | 0.0257 | 0.0105 | +0.0152 | 1.24% |
| `and_count` | 7 | 0.0243 | 0.0103 | +0.0140 | 1.17% |
| `not_count` | 5 | 0.0096 | 0.0079 | +0.0017 | 0.46% |
| `person_is_count` | 7 | 0.0088 | 0.0117 | −0.0029 | 0.42% |
| `expression_nodes` | 15 | 0.0269 | 0.0266 | +0.0003 | 1.29% |
| `expression_depth` | 3 | 0.0021 | 0.0040 | −0.0019 | 0.10% |
| `top_ops` | 211 | 0.7122 | 0.4183 | **+0.2940** | 34.3% |
| `top_ops_multiset` | 55 | 0.3157 | 0.1030 | **+0.2127** | 15.2% |
| `op_signature` | 582 | 0.9166 | 0.8323 | +0.0843 | 44.1% |

**答：存在，但很弱。**

- 最好的单特征 `same_count` 只解释 GT 熵的 **3.35%**；`different_count` 2.92%；
  `and/or/not/person_is/expression_*` 基本在噪声水平（见下方 permutation p 值）。
- 完整算子序列 `top_ops` 名义上达 34%，但**其中 59% 是估计偏差**（0.712 vs null 0.418）；
  扣掉偏差后真实信号约 0.294，即 GT 熵的 14%。
- `op_signature`（582 类）名义 44%，偏差占 **91%**，基本不可用。

**这一步只说明数据里有残余信号，不构成 H3 成立的证据。**

---

## Q2. GRPO 是否让 structure→prediction 相关性增强？

### 2a. 原始 MI

`H(prediction)` 在五个模型上几乎相同（2.074–2.078 ≈ log 8），
因此**不存在「GRPO 压低预测熵导致 MI 机械下降」的混淆**，跨模型可直接比较。

| 特征 | Epoch4 | Epoch5 | V1 | V2 | V3 |
|---|---:|---:|---:|---:|---:|
| `same_count` | 0.0619 | 0.0606 | **0.0695** | 0.0567 | 0.0616 |
| `different_count` | 0.0581 | 0.0561 | 0.0568 | 0.0538 | 0.0536 |
| `and_count` | 0.0302 | 0.0221 | 0.0245 | 0.0262 | 0.0244 |
| `or_count` | 0.0285 | 0.0232 | 0.0364 | 0.0239 | 0.0311 |
| `not_count` | 0.0088 | 0.0071 | 0.0083 | 0.0103 | 0.0092 |
| `person_is_count` | 0.0123 | 0.0086 | 0.0121 | 0.0085 | 0.0100 |
| `expression_nodes` | 0.0339 | 0.0265 | 0.0345 | 0.0318 | 0.0310 |
| `expression_depth` | 0.0053 | 0.0034 | 0.0029 | 0.0032 | 0.0030 |
| `top_ops` | 0.7360 | 0.7325 | **0.7637** | 0.7349 | 0.7228 |
| `top_ops_multiset` | **0.3400** | 0.3244 | 0.3355 | 0.3262 | **0.3151** |
| `op_signature` | 0.9046 | 0.9100 | 0.9057 | 0.9104 | 0.8971 |
| `H(prediction)` | 2.0740 | 2.0760 | 2.0735 | 2.0773 | 2.0779 |

`top_ops_multiset` 从 Epoch4 到 GRPO 是**单调下降**的（0.3400 → 0.3151）。

### 2b. Bootstrap 95% CI（1000 次，配对重抽样）

以 **Epoch4** 为基线：

| 对比 | `same_count` | `top_ops` | `top_ops_multiset` | `op_signature` |
|---|---|---|---|---|
| Epoch5 − Epoch4 | −0.0015 n.s. | −0.0023 n.s. | −0.0157 n.s. | +0.0041 n.s. |
| V1 − Epoch4 | +0.0077 n.s. | **+0.0238 \*** | −0.0032 n.s. | +0.0007 n.s. |
| V2 − Epoch4 | −0.0051 n.s. | +0.0025 n.s. | −0.0132 n.s. | +0.0061 n.s. |
| V3 − Epoch4 | −0.0005 n.s. | −0.0074 n.s. | −0.0209 n.s. | −0.0044 n.s. |

以 **Epoch5** 为基线（分离「GRPO」与「只是多训一会 SFT」）：

| 对比 | `same_count` | `top_ops` | `top_ops_multiset` | `op_signature` |
|---|---|---|---|---|
| V1 − Epoch5 | **+0.0090 \*** | **+0.0255 \*** | +0.0126 n.s. | −0.0031 n.s. |
| V2 − Epoch5 | −0.0039 n.s. | +0.0046 n.s. | +0.0031 n.s. | +0.0015 n.s. |
| V3 − Epoch5 | +0.0013 n.s. | −0.0051 n.s. | −0.0054 n.s. | −0.0078 n.s. |

（`*` = 95% CI 不跨 0）

**唯一支持 H3 的观察**：V1 的 `top_ops` 原始 MI 相对 Epoch4 与 Epoch5 都显著上升
（+0.024 / +0.026）。V2、V3 无上升，`top_ops_multiset` 全部无上升。

**但原始 MI 有一个已知混杂**：`MI(结构; 预测)` 包含两条路径
`结构 → 预测`（真 shortcut）与 `结构 → GT → 预测`（间接）。
模型越准，预测越接近 GT，第二条路径贡献越大，**原始 MI 会随准确率上升而机械上升**。
V1 准确率 0.7550 高于 Epoch4 的 0.7335，因此这个上升需要用 Q3 的 GT 控制来判定。

---

## Q3. 控制 GT 后，这种关联是否仍然存在？

核心指标：**conditional MI**（按 GT 分层加权合并，用满 2000 样本）与
**conditional NMI = MI(·\|GT) / H(预测\|GT)**（消除各模型残差不确定性不同的影响）。

### conditional NMI

| 特征 | Epoch4 | Epoch5 | V1 | V2 | V3 |
|---|---:|---:|---:|---:|---:|
| `same_count` | 0.07197 | 0.08017 | 0.08196 | 0.07281 | 0.07948 |
| `different_count` | 0.07250 | 0.07598 | 0.07268 | 0.06824 | 0.06177 |
| `and_count` | 0.07174 | 0.06700 | 0.06272 | 0.06869 | 0.07096 |
| `or_count` | 0.06896 | 0.07520 | 0.07199 | 0.07301 | 0.07334 |
| `not_count` | 0.04561 | 0.04979 | 0.05058 | 0.05203 | 0.05361 |
| `person_is_count` | 0.09223 | 0.09494 | 0.09083 | 0.10727 | 0.09072 |
| `expression_nodes` | **0.15948** | 0.17778 | 0.17766 | 0.18187 | 0.17511 |
| `expression_depth` | 0.03232 | 0.03975 | 0.03095 | 0.03366 | 0.03278 |
| `top_ops` | 0.72480 | 0.72754 | **0.74546** | 0.72750 | 0.72787 |
| `top_ops_multiset` | 0.42112 | 0.43703 | 0.43753 | 0.43192 | 0.43123 |
| `op_signature` | 0.81405 | 0.83811 | 0.81853 | 0.83502 | 0.81678 |
| `H(预测\|GT)` | 0.9967 | 0.9427 | 0.9598 | 0.9165 | 0.9782 |

**关键点：Epoch5（纯 SFT 多训一个 epoch）与三个 GRPO 模型基本重合。**
例如 `expression_nodes`：Epoch4 0.1595 → **Epoch5 0.1778**，
而 V1 0.1777 / V2 0.1819 / V3 0.1751 —— GRPO 没有超过 Epoch5。

### Bootstrap 95% CI（conditional MI delta）

| 对比 | `same_count` | `top_ops` | `top_ops_multiset` | `op_signature` |
|---|---|---|---|---|
| Epoch5 − Epoch4 | +0.0055 n.s. | **−0.0421 \*** | −0.0148 n.s. | −0.0324 n.s. |
| V1 − Epoch4 | +0.0080 n.s. | −0.0146 n.s. | −0.0057 n.s. | −0.0314 n.s. |
| V2 − Epoch4 | −0.0046 n.s. | **−0.0643 \*** | **−0.0341 \*** | **−0.0554 \*** |
| V3 − Epoch4 | +0.0057 n.s. | −0.0136 n.s. | −0.0004 n.s. | −0.0150 n.s. |
| V1 − Epoch5 | +0.0018 n.s. | +0.0257 n.s. | +0.0091 n.s. | +0.0032 n.s. |
| V2 − Epoch5 | −0.0102 n.s. | −0.0220 n.s. | −0.0195 n.s. | −0.0241 n.s. |
| V3 − Epoch5 | +0.0008 n.s. | +0.0275 n.s. | +0.0136 n.s. | +0.0172 n.s. |

**答：控制 GT 后，关联没有增强。**

- 所有 count 特征在所有模型上均为 n.s.；
- V2 在 `top_ops` / `top_ops_multiset` / `op_signature` 上**显著下降**
  （−0.064 / −0.034 / −0.055，CI 均不跨 0）；
- 以 Epoch5 为基线时，**V1 / V2 / V3 全部 n.s.** ——
  GRPO 相对「多训一会 SFT」没有带来任何可测的 structure 依赖变化；
- 唯一在 Q2 中显著的 V1 `top_ops` 原始 MI 上升，在控制 GT 后变为 n.s.
  （+0.0257，CI [−0.0147, +0.0670]），说明它主要由「结构→GT→预测」的间接路径贡献。

---

## Q4. NKK / KKN / KNK / KNN 的异常能否被具体 structure 解释？

`P(预测=类别 | signature)` 变化最大的组合（`top_ops_multiset`，n≥20）：

**V1 vs Epoch4（NKK 异常）**

| 类别 | signature | n | Epoch4 | V1 | Δ | 绝对样本数 |
|---|---|---:|---:|---:|---:|---:|
| NKK | `and\|and\|not` | 22 | 0.000 | 0.136 | **+0.136** | 3 |
| NKK | `different\|not\|not` | 27 | 0.111 | 0.222 | +0.111 | 3 |
| NKK | `different\|not\|or` | 48 | 0.021 | 0.125 | +0.104 | 5 |
| NKK | `different\|or\|or` | 29 | 0.035 | 0.138 | +0.103 | 3 |
| NKK | `and\|not\|or` | 39 | 0.077 | 0.179 | +0.103 | 4 |

**V2 vs Epoch4（KKN/KNN 上升、KNK/NKK 下降）**

| 类别 | signature | n | Epoch4 | V2 | Δ | 绝对样本数 |
|---|---|---:|---:|---:|---:|---:|
| KKN | `different\|or\|or` | 29 | 0.414 | 0.586 | +0.172 | 5 |
| KNN | `and\|and\|or` | 21 | 0.143 | 0.286 | +0.143 | 3 |
| KKN | `not\|or\|person_is` | 49 | 0.143 | 0.245 | +0.102 | 5 |
| KNK | `different\|not\|or` | 48 | 0.292 | 0.188 | −0.104 | 5 |
| KNK | `and\|not\|or` | 39 | 0.103 | 0.000 | −0.103 | 4 |

**答：不能归因到具体 structure。**

类别异常**分散在至少 5 个不同 signature** 上，每个组合只贡献 **3–5 个样本**；
没有任何一个结构能解释 NKK 或 KKN 的整体位移。这与 Q2/Q3 的 MI 结果一致：
结构并没有被 GRPO 以某种集中方式放大。

---

## Q4b. 被 RL 修好的题 vs 被 RL 破坏的题，是否集中在不同的 structure？

| 模型 | WC（SFT 错→GRPO 对） | CW（SFT 对→GRPO 错） | 净变化 |
|---|---:|---:|---:|
| V1 | 138 | 101 | +37 |
| V2 | 159 | 98 | +61 |
| V3 | 138 | 112 | +26 |

对 `signature × {WC, CW}` 列联表做卡方独立性检验：

| 模型 | signature 数 | chi2 | dof | p | 结论 |
|---|---:|---:|---:|---:|---|
| V1 | 37 | 24.9 | 36 | **0.9193** | 无显著差异 |
| V2 | 38 | 36.0 | 37 | **0.5156** | 无显著差异 |
| V3 | 38 | 33.1 | 37 | **0.6509** | 无显著差异 |

**答：WC 与 CW 在结构上没有分离。** 同一个 signature 常常同时出现在两个列表里，
例如 V2 的 `not|or|person_is`（WC=8, CW=7）、`different|or|same`（WC=10, CW=10）。

若 GRPO 在放大 shortcut，我们应当看到「退化」集中在 shortcut 会误导的结构、
「修复」集中在别处。实测两者混合在同一批结构里，更像**整体难度**而非**结构偏置**。

---

## Q5. 统计校准

### Permutation 检验（1000 次，打乱 prediction）

| 特征 | Epoch4 | Epoch5 | V1 | V2 | V3 | 判读 |
|---|---|---|---|---|---|---|
| `same_count` | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 | 真实信号 |
| `different_count` | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 | 真实信号 |
| `and_count` | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 | 真实信号 |
| `or_count` | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 | 真实信号 |
| `top_ops` | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 | 真实信号（但含大量偏差，见 Q1） |
| `top_ops_multiset` | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 | 真实信号 |
| `op_signature` | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 | 真实但极弱（偏差占 91%） |
| `not_count` | 0.267 | 0.512 | 0.353 | 0.095 | 0.235 | **无信号** |
| `person_is_count` | 0.373 | 0.864 | 0.416 | 0.929 | 0.798 | **无信号** |
| `expression_nodes` | 0.030 | 0.372 | 0.017 | 0.080 | 0.144 | 边缘 |
| `expression_depth` | 0.161 | 0.503 | 0.765 | 0.670 | 0.751 | **无信号** |

即：`same / different / and / or` 四个算子计数与预测确有真实关联，
但 **`not` / `person_is` / `expression_depth` 完全没有**。

### Bootstrap

`MI(GRPO) − MI(Epoch4)` 与 `MI(GRPO) − MI(Epoch5)` 的 1000 次配对重抽样 CI 见 Q2/Q3。
跨模型比较用同一特征、同一样本、同一 GT，估计偏差在各模型间相同，差值可解释。

---

## H3 判定：NOT_SUPPORTED

| # | 判据 | 实测 | 结果 |
|---|---|---|---|
| 1 | 数据 structure→GT 存在残余 signal | 存在但弱（最好单特征 3.35% GT 熵；算子序列扣偏差后 14%） | ✅（弱） |
| 2 | GRPO 后 `MI(结构; 预测)` 明显高于 SFT | 仅 V1 的 `top_ops` 原始 MI 上升；`top_ops_multiset` 单调下降；V2/V3 无上升 | ❌ |
| 3 | 控制 GT 后关联仍增强 | V2 显著**下降**（三特征 CI 均不跨 0）；V1/V3 n.s.；相对 Epoch5 全部 n.s. | ❌ |
| 4 | class anomaly 可追溯到具体 structure | 分散在 ≥5 个 signature，每个仅 3–5 样本 | ❌ |
| 5 | CW 退化集中在 shortcut structures | 卡方 p = 0.52 / 0.65 / 0.92，WC 与 CW 在结构上无分离 | ❌ |

```text
H3 = NOT_SUPPORTED

数据 shortcut 存在，但没有证据表明 GRPO 在放大它。
```

### 需要如实记录的一点反向观察

V1 的 `top_ops` **原始** MI 相对 Epoch4（+0.0238）与 Epoch5（+0.0255）都显著上升，
是本轮唯一支持 H3 的统计量。但它有两个解释：

1. 原始 MI 混杂了「结构→GT→预测」的间接路径，模型越准该项越大；
2. 控制 GT 后同一对比变为 n.s.（+0.0257，CI [−0.0147, +0.0670]）。

且 V2/V3 均未复现该上升。因此它不足以支持 H3，但也不应被略去：
**V1 是唯一一个在原始指标上出现 structure 依赖上升的模型**，
这与 V1 表现出最强的 NKK 类别异常在时间上是吻合的。

### 为什么没有判 INCONCLUSIVE

规格中 INCONCLUSIVE 的两个条件都不成立：

- 「常见 signature 样本数太少」：`top_ops` 确实只有 5 个 signature 达到 N≥20（5.6%），
  但改用同属纯结构特征的 `top_ops_multiset` 后有 38 个 signature、覆盖 **91.8%** 数据，
  且 MI 分析使用全部 2000 样本，统计功效充足。
- 「不同指标方向互相冲突」：除上述 V1 原始 MI 一项外，
  其余指标（conditional MI / NMI / 类别归因 / WC-CW 分离）方向一致地指向「无增强」。

### 一个方法论收获

本轮最重要的控制是 **Epoch5**。若只比较 Epoch4 vs GRPO，
`expression_nodes` 的 conditional NMI 从 0.1595 升到 0.1777–0.1819，容易被读成「RL 放大结构依赖」；
但 Epoch5（纯 SFT 多训一个 epoch）是 0.1778，与 GRPO 完全重合。
**这些变化是「训练更多」的通用效应，不是 RL 特有。**

---

## 输出文件

```text
outputs/grpo_h3_shortcut_audit/
  dataset_structure_mi.json      Q1 structure→GT MI
  model_prediction_mi.json       Q2/Q3 五模型 MI / NMI / conditional MI + 预测熵
  permutation_results.json       1000 次置换零分布与经验 p
  bootstrap_results.json         vs Epoch4 与 vs Epoch5 两组基线的 1000 次配对 CI
  wc_cw_separation.json          WC/CW 结构分离卡方检验
  conditional_gt_analysis.csv    (GT, signature) 准确率（仅 5 格达 N≥20）
  structure_accuracy.csv/json    按 signature 的五模型准确率 + GT/预测分布
  class_bias_by_structure.csv    P(预测=类别 | signature, 模型)
  transition_by_structure.csv    按 signature 的 WC / CW
  sample_features.csv            逐样本表（id / GT / 结构特征 / 五模型预测与对错）
  h3_shortcut_report.md          本报告
```

生成脚本：`scripts/audit_h3_shortcut.py`（零训练、零推理，复用已有预测与
`scripts/audit_dataset_features.row_features`）。

## 未执行事项（本轮禁止）

```text
未训练 GRPO-V4
未重新训练任何模型
未修改 reward / beta / LR
未修改 generator
未生成 structure-balanced 数据集
未引入神经网络特征或 LLM judge 难度标注
```

## 下一步建议（不自动执行）

H1（KL）、H2（reward sparsity）、H3（structure shortcut）**均不支持**。
三条干预都真实生效，但都没能解释或缓解 polarization。
建议下一轮把问题从「优化过程缺什么」转向「目标与任务的错位」：

```text
1. 复核 polarization 是否本身就是「用 exact-match 目标做 RL」的必然结果
   —— H2 中 partial reward 的 Goodhart 证据支持这一方向
2. 若要继续追 structure，需要的是更大的诊断集（top_ops 需 N >> 2000 才能支撑
   per-signature 分析），而不是新特征
3. 在提出独立假设之前，不建议再对 reward / beta / LR 做扫描
```
