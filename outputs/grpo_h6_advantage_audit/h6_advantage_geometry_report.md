# H6: Group-Normalized Advantage Geometry Audit

日期：2026-08-29
类型：**零训练诊断**（无 optimizer step、无新 checkpoint）
判定：**H6_GEOMETRY_SUPPORTED**（H6-B：**尚未因果验证**）

---

## H6 定义回顾

```text
H6:
With binary reward and group-wise sample-std normalization,
group size changes the effective weighting of successful and
failed rollouts even when the completions and rewards themselves
are held fixed.

This altered advantage geometry may contribute to the
stronger sharpening observed under K=16.
```

严格限制：本轮**只能**验证「K 是否改变 advantage / task-gradient geometry」。
**不能**据此断言 advantage geometry 造成了最终 polarization。

---

## Q1. K=8→16 在数学上如何改变 binary-reward advantage？

Phase A 解析枚举（gate：`H6_ANALYTIC_FORMULA_VALID`，
与 `torch.std(correction=1)` 的最大误差 **1.89e-08**）。

公式：

$$s=\sqrt{\frac{m(K-m)}{K(K-1)}},\qquad
A_+=\sqrt{\frac{(K-1)(K-m)}{Km}},\qquad
A_-=-\sqrt{\frac{(K-1)m}{K(K-m)}}$$

### rare-success 组（m=1,2,3）

| K | m | sample std | A+ | A− | sum A+ | sum \|A−\| | mean \|A\| |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 1 | 0.3536 | **+2.4742** | −0.3535 | 2.4742 | 2.4742 | 0.6185 |
| 8 | 2 | 0.4629 | +1.6198 | −0.5399 | 3.2397 | 3.2397 | 0.8099 |
| 8 | 3 | 0.5175 | +1.2074 | −0.7244 | 3.6221 | 3.6221 | 0.9055 |
| 16 | 1 | 0.2500 | **+3.7485** | −0.2499 | 3.7485 | 3.7485 | 0.4686 |
| 16 | 2 | 0.3416 | +2.5610 | −0.3659 | 5.1220 | 5.1220 | 0.6402 |
| 16 | 3 | 0.4031 | +2.0151 | −0.4650 | 6.0452 | 6.0452 | 0.7556 |

与规格预期一致（K8 m=1: A+≈+2.475 / A−≈−0.354；K16 m=1: A+=+3.750 / A−=−0.250）。

### 三个必须同时看的事实

1. **单个 rare success 的 A+ 变大**：m=1 时 2.4742 → 3.7485（**+51.5%**）。
2. **sum A+（= sum\|A−\|）也变大**：2.4742 → 3.7485（+51.5%），
   因为 $\sum A_+ = \sqrt{m(K-m)(K-1)/K}$。
3. **但 mean\|A\| 变小**：0.6185 → 0.4686（**−24.2%**）。
   更多 rollout 摊薄了单位样本的 advantage 尺度。

所以「K16 让每个 rare success 权重更大」与「K16 的整体 advantage 尺度」
是**两件不同的事**；真正决定 batch 梯度的是
**多少 completion 携带非零 advantage** 以及**它们的权重总和**。

注：TRL 在分母额外加 `1e-4`，对 A+ 的最大修正量为 **1.499e-03**（相对 ~2.8e-4），
本报告所有数值均使用 TRL 的实际实现（`s + 1e-4`）。

---

## Q2. 真实 rollout composition 中这种变化有多常见？

固定 200 prompts × 16 completions（SFT Epoch4，seed 20260904，一次性生成，未重新采样）：

```text
m_correct 分布 (n=200):
  m=0 : 13      m=6 :  6      m=12:  7
  m=1 :  1      m=7 :  7      m=13: 11
  m=2 :  6      m=8 :  6      m=14: 13
  m=3 :  5      m=9 :  6      m=15:  9
  m=4 :  6     m=10 :  6      m=16: 86
  m=5 :  5     m=11 :  7
```

分层：

```text
零方差组（m=0 或 m=16）     99 / 200 = 49.5%   -> 两种 K 下 advantage 全 0，无 geometry 差异
mixed 组（1 ≤ m ≤ 15）     101 / 200 = 50.5%  -> 受 grouping 影响
rare-success（m=1..3）      12 / 200 =  6.0%  -> 受影响最大的一类
```

**约一半的 prompt 会经历 geometry 变化；其中 rare-success 只占 6%。**

整体（20 random partitions × 200 prompts = 4000 次划分）：

```text
mean |ΔA|  = 0.1025   (median 0.0217)
cosine     = 0.9626
```

只统计 mixed 组时（见下表）差异明显放大。

---

## Q3. K16 是否让一批原本 synthetic-K8 下 zero-advantage 的 wrong completions 获得 negative advantage？

**是，且高度集中在 rare-success 组。**（§9 的关键机制）

| bucket (m/16) | n_prompts | mean \|ΔA\| | cosine | **newly_nonzero_wrong** / prompt | newly_nonzero_correct / prompt |
|---|---:|---:|---:|---:|---:|
| m=0 | 13 | 0.0000 | nan | 0.000 | 0.000 |
| **m=1** | 1 | 0.2499 | 0.9661 | **8.000** | 0.000 |
| **m=2** | 6 | 0.1880 | 0.9641 | **3.867** | 0.000 |
| **m=3** | 5 | 0.2294 | 0.9541 | **1.920** | 0.000 |
| m=4-7 | 24 | 0.1964 | 0.9632 | 0.117 | 0.000 |
| m=8 | 6 | 0.1926 | 0.9656 | 0.000 | 0.000 |
| m=9-12 | 26 | 0.2036 | 0.9600 | 0.000 | 0.215 |
| m=13-15 | 33 | 0.2067 | 0.9646 | 0.000 | **4.048** |
| m=16 | 86 | 0.0000 | nan | 0.000 | 0.000 |

**机制解释（与 §9 的预测完全一致）**：

```text
16 rollout 中只有 1 个 correct：
  K16:  1 correct + 15 wrong 全在一个 mixed group
        -> correct 有 positive advantage
        -> 15 个 wrong 全有 negative advantage

  拆成 8+8:
        含 correct 的那组 -> mixed（有信号）
        另一组 8 个 wrong  -> all-wrong -> advantage = 0
```

即 K16 **不只是"让 correct 被看到"**，它还把一批本来零梯度的 wrong samples
变成负梯度 samples。m=1 时平均每个 partition 有 **8.0 个** such samples。

**对称现象**：高成功率组（m=13-15）中，K16 让平均 **4.048 个** correct completions
从 zero-advantage 变为 positive-advantage（拆分可能造出 all-correct 子组）。

总量（20 partitions × 200 prompts）：

```text
wrong   completions 0 -> negative : 872
correct completions 0 -> positive : 2784
```

净效果：K16 **动员了更多 completion 参与梯度**：

```text
nonzero advantages / group:  K16 8.080  vs  synthetic-K8 7.166   (+0.914)
positive total weight:       K16 3.206  vs  2.840   (ratio 1.129)
negative total weight:       K16 3.206  vs  2.840   (ratio 1.129)
```

---

## Q4. 固定完全相同 completion/reward 后，advantage vector 差异多大？

Counterfactual 设计（严格隔离 geometry）：

```text
同一组 16 个 completion、同一组 16 个 reward、
同一个 policy (SFT Epoch4)、同一批 token。

A16     : 16 个 reward 一起 normalize
A8-split: 同一批 16 个 completion 随机分成 8+8，各自 normalize 后拼回

不同：only grouping / normalization。
这不是在模拟真实 K8 training trajectory，只是 controlled counterfactual。
```

结果（20 random partitions；另存 `first8/last8` 作为 deterministic sanity check，
结果一致：mean\|ΔA\| 0.0971 vs 0.1025，cosine 0.9626 vs 0.9626）：

```text
mixed 组 mean |ΔA| ≈ 0.19 - 0.25
mixed 组 cosine   ≈ 0.95 - 0.97
```

即 per-element 平均差异约 0.2（advantage 的典型尺度是 0.25–3.75），
但**向量方向高度一致**。

---

## Q5 & Q6. LoRA task-gradient：norm 变了吗？direction 变了吗？

Phase C：64 prompts（优先覆盖 m=1~3、mixed medium、high-success）× 20 partitions，
在 **ratio=1 的 frozen on-policy 设定**下计算梯度。

Loss 与 TRL 0.23 对齐（`loss_type="dapo"` 默认）：

```text
per_token_loss = -min(coef_1, coef_2) * A  ->  -A        (coef_1 = coef_2 = 1)
loss           = sum(per_token_loss * completion_mask) / num_items_in_batch
```

**KL 被排除**：H6 只隔离 reward-relative gradient 分量。
实际训练中 `beta=0.01` 存在，但在 SFT 初始化处 reference≈policy，KL 本来就接近 0。

**零 optimizer step**：只做 `torch.autograd.grad`，不调用 `optimizer.step`，
不产生任何新 checkpoint。两次 backward 复用同一批 log-probability（同一计算图）。

### 结果（64 prompts × 20 partitions，10000 次 prompt 配对 bootstrap）

| 指标 | mean | median | p10 | p90 | 95% CI |
|---|---:|---:|---:|---:|---|
| **norm ratio ‖g16‖/‖g8‖** | **1.2180** | 1.0644 | 1.0340 | 1.5807 | **[1.1813, 1.2564]** |
| **cosine(g16, g8)** | **0.9965** | 0.9999 | 0.9973 | 0.9999 | **[0.9932, 0.9987]** |
| ‖g16−g8‖/‖g8‖ | 0.2345 | 0.0728 | 0.0390 | 0.5846 | [0.1978, 0.2722] |

**Q5 答：是。梯度范数平均增大约 21.8%，CI 不跨 1。**
**Q6 答：基本不变。cosine ≈ 0.9965，方向几乎完全一致。**

即：**K 的改变主要是梯度"强度"的放大，而不是"方向"的改变。**

---

## Q7. rare-success groups (m=1~3) 是否差异最大？**是**

| 指标 | 全体 (n=64) | **rare-success m=1..3 (n=12)** |
|---|---:|---:|
| norm ratio mean | 1.2180 CI [1.1813,1.2564] | **1.2688 CI [1.2096,1.3341]** |
| cosine mean | 0.9965 CI [0.9932,0.9987] | 0.9957 CI [0.9918,0.9988] |
| ‖g16−g8‖/‖g8‖ | 0.2345 CI [0.1978,0.2722] | **0.2825 CI [0.2204,0.3494]** |
| newly_nonzero_wrong / prompt | 0.218 | **m=1: 8.000 / m=2: 3.867 / m=3: 1.920** |

rare-success 组在**所有三项**差异指标上都是最大的，
且 `newly_nonzero_wrong` 几乎完全集中在这一区（m≥8 时恒为 0）。
这与 H5/K16 最关心的 low-support 区域直接对应。

---

## Q8. H6_GEOMETRY = **SUPPORTED**

| 判据 | 实测 | 结果 |
|---|---|---|
| advantage vectors 系统性变化 | mixed 组 mean\|ΔA\| 0.19–0.25；nonzero 数 +0.914；正/负总权重 ×1.129 | ✅ |
| gradient norm 或 direction 有明确变化 | **norm ratio 1.2180，CI [1.1813,1.2564] 不跨 1** | ✅ |
| rare-success 组尤其明显 | norm ratio 1.2688；相对差 0.2825；newly_nonzero_wrong 8.0/3.87/1.92 | ✅ |

```text
H6_GEOMETRY_SUPPORTED
```

### 必须写清的限定条件

1. **变化主要是"强度"而非"方向"**：cosine ≈ 0.9965（CI [0.9932,0.9987]），
   K16 基本是沿着同一个方向把梯度放大 ~22%。
2. **约一半 prompt 完全不受影响**：m=0 与 m=16 组（99/200）在两种 K 下 advantage 全为 0，
   geometry 差异为 0。
3. **rare-success 只占 6%（12/200 prompts）**，其梯度子组 n=12，CI 较宽。
4. 只使用 **SFT Epoch4** 一个 policy 状态（规格 §16 指定 Epoch4 为 primary；
   K8/K16 checkpoint 为 secondary，本轮未做以避免延迟主结论）。

---

## Q9. Does this establish that geometry caused polarization? **NO — not yet causally tested.**

```text
NO — not yet causally tested.
```

本轮只能写：

```text
Group-normalized advantage geometry is a plausible mechanism
that changes the effective task-gradient when K changes.
```

**不能**写：

```text
It caused the K16 sharpening.
```

要证明后者，需要在**固定 K** 的前提下操纵 normalization rule 的训练干预。
本轮禁止训练，因此不做。

---

## 与 H5/K16 结果的关系

```text
K16 实测：entropy diff −0.00864 EXCL0，即 K16 sharpening 更强（V2 报告）。
本轮给出的一个自洽解释：
  K16 -> 更少零方差子组 -> 更多 completion 携带非零 advantage
       -> 每 prompt 的 task-gradient 范数 ×1.22（rare-success ×1.27）
       -> 有效更新更强 -> sharpening 更强。
```

这与 H5 的「coverage」解释是**并存且不可分离**的：改变 K 同时改变
(a) sampling coverage 与 (b) advantage geometry。因此 K16 **不是一个纯净的 coverage 干预**，
H5 因果结论已按此严格限定（见 K16 报告修正 3）。

---

## 输出文件

```text
scripts/audit_h6_advantage_geometry.py     Phase A + Phase B
scripts/audit_h6_gradient_geometry.py      Phase C

outputs/grpo_h6_advantage_audit/
  analytic_advantage_table.csv         K=8/16 × m=0..K 全枚举
  analytic_gate.json                   H6_ANALYTIC_FORMULA_VALID
  fixed_rollouts_200x16.jsonl          一次性生成的 200×16 completions
  group_composition_summary.csv        每个 prompt 的 m_correct 与 K16 权重
  partition_advantage_comparison.csv   21 种划分 × 200 prompts 的逐项对比
  partition_summary.json               按 m 分层的 geometry 差异
  newly_nonzero_mechanism.json         §9 机制的计数
  gradient_geometry.json               64 prompts × 20 partitions 的梯度几何
  rare_success_gradient_geometry.json  m=1..3 子组
  bootstrap_results.json               10000 次 prompt 配对 bootstrap
  h6_advantage_geometry_report.md      本报告
```

## 本轮禁止事项执行情况

```text
未跑 K=32          未启动新 GRPO
未做 optimizer.step   未产生新 checkpoint
未做 reward shaping   未做 beta / LR sweep
未生成新训练数据      未把 H6-A 支持写成 polarization 因果证明
```
