# GRPO-V3 报告：Binary Reward Sparsity / All-Wrong Zero-Advantage 验证（H2）

日期：2026-08-29
状态：**GRPO_V3_H2_COMPLETE**
H2 判定：**NOT_SUPPORTED**

---

## 0. 一句话结论

Partial reward **确实为大量 all-wrong group 恢复了训练信号**
（训练期 673 个 all-wrong group 中 162 个被 rescue，24.1%）；
但 **All-wrong 不降反升（16.0% → 18.0%），Pass@8 继续下降（84.0% → 82.0%），
mixed→all-wrong 从 19 增到 25**，且 fresh holdout 上 exact greedy accuracy
**显著变差**（−1.75pp，p=0.0276）。

因此：**all-wrong zero advantage 虽然真实存在，但不是当前 failure mode 的主要成因。**

---

## 1. Phase A：零训练 reward-rescue audit

未做任何新 rollout，直接复用 `outputs/grpo_v2_analysis/rollout_D_final200.jsonl`
（GRPO-V2，200 prompts × 8 rollouts，真实模型输出），对同一批 rollout 同时计算两种 reward。

### 主样本：GRPO-V2，200-prompt 子集

```text
exact all-wrong groups                32
  ├─ partial 仍然 zero-variance        22
  └─ partial 恢复 non-zero variance    10

reward rescue rate = 10 / 32 = 31.25%     [Wilson 95% CI 0.180, 0.486]
```

整体零方差组（8 次 rollout reward 全同）：`exact 150/200 → partial 140/200`。

### 上下文样本（同一 200-prompt 子集，不同 policy）

| Source | groups | exact all-wrong | partial gains variance | still zero-var | rescue rate | 95% CI |
|---|---:|---:|---:|---:|---:|---|
| **GRPO-V2** | 200 | **32** | **10** | **22** | **31.2%** | [0.180, 0.486] |
| SFT Epoch4 | 200 | 13 | 7 | 6 | 53.8% | [0.291, 0.768] |
| GRPO-V1 | 200 | 27 | 7 | 20 | 25.9% | [0.132, 0.447] |
| GRPO-V2 probe（20 prompts × 5 steps） | 100 | 14 | 5 | 9 | 35.7% | [0.163, 0.612] |

### Rescue 与 non-rescue 的结构差异

- **被 rescue 的组**：8 次 rollout 出现 **2 种不同的错误 pattern**（例：gt=NNN，
  预测 `[NNK×7, NKK×1]` → partial `[0.667×7, 0.333×1]` → std>0 → advantage≠0）。
- **未被 rescue 的 22 组**：8 次 rollout 全部预测**同一个**错误 pattern
  （例：gt=NKK，预测全为 KNN → partial 全为 0.0），partial reward 恒定，仍然零方差。

**Gate 判定：31.2% > 25% → CONTINUE_TO_PHASE_B。**
须注意：CI 下界 0.180 低于 0.25，该 gate 是工程阈值而非显著性标准。

> **Phase A 实现纠正**：初版用 `partial_std > 1e-12` 判定，把 float32 噪声
> （std≈1e-8 → advantage≈1e-4）误判为 rescued，得到虚高的 96.9%。
> 改为 **distinct partial value > 1 且 std>1e-6 且 max|advantage|>1e-3** 后为 10/32。

---

## 2. Phase C 训练期机制验证（真实 rollout）

`outputs/grpo_v3_partial/audit.json`：

```text
reward_mode = partial
exact-all-wrong groups seen : 673
rescued (regained variance) : 162      -> 24.1%
rescued examples archived   : 5
```

实例（step 2，真实 rollout）：

```text
gt = NNN
predictions = [NNK, NNK, NNK, NNK, NNK, NNK, NKK, NNK]
exact       = [0, 0, 0, 0, 0, 0, 0, 0]                -> std = 0, advantage = 0
partial     = [.667, .667, .667, .667, .667, .667, .333, .667]  -> std > 0, advantage ≠ 0
```

**机制确实生效。** 按分桶的 pooled rescue rate 随训练下降：

```text
1-100  34.7%   301-400 19.6%
101-200 29.8%   401-500 22.8%
201-300 20.8%   501-600 22.8%  601-625 18.2%
```

即：policy 越 sharpen，all-wrong group 越同质化，可被 rescue 的比例越低
—— 这正是 sharpening 与 reward sparsity 的耦合。

---

## 3. 训练动态：V3（partial）vs V2（exact）

| bucket | exact_reward_mean | shaped_reward_mean | exact mixed | exact all-correct | exact all-wrong | pooled rescue | KL |
|---|---:|---:|---:|---:|---:|---:|---:|
| **V3** 1–100 | 0.7252 | 0.8429 | 0.3987 | 0.5112 | **0.0900** | 0.3472 | 0.0035 |
| V3 101–200 | 0.7438 | 0.8574 | 0.2888 | 0.5813 | 0.1300 | 0.2981 | 0.0100 |
| V3 201–300 | 0.7520 | 0.8638 | 0.2587 | 0.6088 | 0.1325 | 0.2075 | 0.0162 |
| V3 301–400 | 0.7491 | 0.8640 | 0.2437 | 0.6162 | 0.1400 | 0.1964 | 0.0172 |
| V3 401–500 | 0.7411 | 0.8696 | 0.2338 | 0.6125 | 0.1537 | 0.2276 | 0.0226 |
| V3 501–600 | 0.7478 | 0.8681 | 0.2263 | 0.6200 | 0.1537 | 0.2276 | 0.0234 |
| V3 601–625 | **0.7462** | 0.8735 | 0.2100 | 0.6250 | **0.1650** | 0.1818 | 0.0285 |
| **V2** 601–625 | **0.7575** | 0.7575 | 0.2200 | 0.6250 | **0.1550** | 0.0000 | 0.0085 |
| V2 1–100 | 0.7286 | 0.7286 | 0.3800 | 0.5150 | 0.1050 | 0.0000 | 0.0046 |

关键：**V3 末段 exact_all_wrong 0.1650 > V2 的 0.1550**，方向不利。
V3 早期 all-wrong 更低（0.09 vs 0.105），但训练后段反超。

**V2 Val（checkpoint 选择依据）**

| Step | 0 | 100 | 200 | 300 | 400 | 500 | 600 | 625 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V3 | 0.7220 | 0.7340 | 0.7580 | 0.7680 | **0.7740** | 0.7320 | 0.7540 | 0.7400 |
| V2 | 0.7220 | 0.7540 | 0.7540 | 0.7660 | 0.7600 | 0.7420 | **0.7720** | 0.7620 |

**best checkpoint = `outputs/grpo_v3_partial/checkpoint-400`（V2 Val 0.7740）**
规则：仅按 V2 Val exact accuracy，并列取更早；未使用 partial reward / Pass@8。
选出后才打开 fresh holdout。

**V3 KL trajectory**：first 3.92e-05 → last 5.65e-03，mean 0.0160，max 0.179
（V2：mean 0.0091，max 0.361）。V3 平均 KL 更大但尖峰更小。

---

## 4. 固定 200-prompt 行为对照

采样与前三轮完全一致：8 rollouts，temperature 0.8，top_p 0.95，max 64 tokens，seed 20260828。

| Metric | Epoch4 | GRPO-V1 | GRPO-V2 | **GRPO-V3** |
|---|---:|---:|---:|---:|
| Mean reward | 0.6669 | 0.7300 | 0.7544 | **0.6931** |
| **Pass@8** | 93.5% | 86.5% | 84.0% | **82.0%** |
| Mixed | 56.0% | 31.5% | 25.0% | 29.0% |
| All-correct | 37.5% | 55.0% | 59.0% | **53.0%** |
| **All-wrong** | 6.5% | 13.5% | 16.0% | **18.0%** |
| Avg unique | 1.825 | 1.390 | 1.335 | 1.370 |
| Avg correct/group | 5.335 | 5.840 | 6.035 | **5.545** |

### mixed → all-correct / mixed → all-wrong（相对 Epoch4）

| | mixed→all-correct | mixed→all-wrong |
|---|---:|---:|
| Epoch4 → GRPO-V1 | 38 | 17 |
| Epoch4 → GRPO-V2 | 43 | 19 |
| Epoch4 → **GRPO-V3** | **34** | **25** |

V2→V3 直接转移：`all-wrong → all-wrong 24, all-wrong → mixed 7, mixed → all-wrong 11`。

### exact-all-wrong 组中 partial reward 可 rescue 的比例

| | Epoch4 | V1 | V2 | V3 |
|---|---:|---:|---:|---:|
| all-wrong groups | 13 | 27 | 32 | 36 |
| rescued | 7 | 7 | 10 | 9 |
| rescue rate | 53.8% | 25.9% | 31.2% | 25.0% |

### 显著性（V2 → V3，同 200 prompts，McNemar）

| Metric | V2 | V3 | delta | V2-only / V3-only | p |
|---|---:|---:|---:|---:|---:|
| all-wrong | 0.1600 | 0.1800 | **+2.00pp** | 8 / 12 | 0.5034 |
| all-correct | 0.5900 | 0.5300 | −6.00pp | 23 / 11 | 0.0576 |
| mixed | 0.2500 | 0.2900 | +4.00pp | 21 / 29 | 0.3222 |

---

## 5. 五模型统一评测（greedy，指标恒为 exact accuracy）

| Model | V2 Val (500) | V2 Test (1000) | V2 Holdout (2000) | **V3 Fresh Holdout (2000)** |
|---|---:|---:|---:|---:|
| SFT Epoch4 (ckpt-1252) | 0.7220 | 0.7570 | 0.7385 | 0.7335 |
| SFT Epoch5 (ckpt-1565) | 0.7480 | 0.7790 | 0.7530 | 0.7625 |
| GRPO-V1 (ckpt-200) | 0.7660 | 0.7500 | 0.7350 | 0.7550 |
| **GRPO-V2 (ckpt-600)** | 0.7720 | **0.7840** | **0.7705** | **0.7695** |
| **GRPO-V3 (ckpt-400)** | **0.7740** | 0.7700 | 0.7430 | 0.7520 |

交叉一致性：本轮重跑的 Epoch4 / Epoch5 / V1 / V2 在 v2_val、v2_test、v2_holdout 上
与 V2 轮报告**逐位一致**（0.7220 / 0.7570 / 0.7385；0.7480 / 0.7790 / 0.7530；
0.7660 / 0.7500 / 0.7350；0.7720 / 0.7840 / 0.7705）。

**V3 在 V2 Val 上最高（0.7740），但那正是它被选中的集合；在三个 held-out 集合上均低于 V2。**

### Fresh Holdout 统计（N=2000，paired）

| 对比 | delta | McNemar exact p | bootstrap 95% CI |
|---|---:|---:|---:|
| **V3 − V2** | **−1.75pp** | **0.0276** | [−3.25, −0.25]pp（不跨 0） |
| V3 − Epoch5 | −1.05pp | 0.1919 | [−2.55, +0.40]pp（跨 0） |
| V3 − V1 | −0.30pp | 0.7314 | [−1.75, +1.15]pp（跨 0） |
| V3 − Epoch4 | +1.85pp | 0.0286 | [+0.25, +3.45]pp |

V2 Holdout 上：V3 − V2 = **−2.75pp, p=0.0004, CI [−4.20, −1.25]**。

**V3 显著劣于 V2**，且不再显著优于 Epoch5。

### 8-class accuracy（fresh V3 holdout）

| Model | KKK | KKN | KNK | KNN | NKK | NKN | NNK | NNN | macro |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Epoch4 | .937 | .596 | .754 | .668 | .598 | .723 | .754 | .826 | .7320 |
| Epoch5 | .944 | .732 | .790 | .711 | .687 | .720 | .746 | .752 | .7604 |
| GRPO-V1 | .970 | .668 | .710 | .609 | .756 | .780 | .753 | .785 | .7538 |
| **GRPO-V2** | .944 | **.736** | **.847** | **.735** | .646 | .708 | .727 | .802 | **.7680** |
| GRPO-V3 | .959 | .732 | .758 | .639 | .663 | .678 | .773 | .802 | .7504 |

V3 vs V2 per-class delta（pp）：`KNK −8.87, KNN −9.88, NKN −2.97, KKN −0.40, NNN 0.00,
KKK +1.49, NKK +1.63, NNK +4.69`。**class 极化依旧，只是再次换了受益类别。**

预测分布（fresh holdout，GT 为 KKK 269 / KKN 250 / KNK 248 / KNN 253 / NKK 246 / NKN 236 / NNK 256 / NNN 242）：

```text
GRPO-V2  KKK 268  KKN 255  KNK 291  KNN 264  NKK 204  NKN 229  NNK 260  NNN 228
GRPO-V3  KKK 291  KKN 256  KNK 255  KNN 205  NKK 219  NKN 223  NNK 293  NNN 256
```

---

## 6. 报告必须回答的 9 个问题

1. **Exact all-wrong 有多少？** 200-prompt 诊断集：Epoch4 13、V1 27、V2 32、**V3 36**；
   训练期累计 673 组（625 steps × 8 groups）。
2. **Partial reward 能 rescue 多少组的 variance？** Phase A 10/32 = **31.2%**；
   训练期 162/673 = **24.1%**；V3 最终 policy 9/36 = 25.0%。
3. **这些组是否真的产生 non-zero advantage？** 是。已归档 5 个真实 group 实例
   （`outputs/grpo_v3_partial/audit.json`），且单元测试锁定了该机制。
4. **V3 vs V2 all-wrong 如何变化？** 16.0% → **18.0%**（+2.0pp，p=0.5034）。**变差。**
5. **Pass@8 如何变化？** 84.0% → **82.0%**（−2.0pp）。**变差。**
6. **mixed→all-wrong 如何变化？** 19 → **25**。**变差。**（mixed→all-correct 43 → 34）
7. **Exact greedy accuracy 如何变化？** V2 Val 0.7720 → 0.7740（选择集）；
   但 fresh holdout **0.7695 → 0.7520（−1.75pp, p=0.0276）**，v2_holdout 0.7705 → 0.7430。**变差。**
8. **Fresh Holdout 是否显著改善？** **否，显著恶化**（p=0.0276，CI 不跨 0）。
9. **H2 = SUPPORTED / NOT_SUPPORTED / INCONCLUSIVE？** → **NOT_SUPPORTED**

---

## 7. H2 判定：NOT_SUPPORTED

| 判据 | 要求 | 实测 | 结果 |
|---|---|---|---|
| 机制：大量 all-wrong 组恢复 variance | 是 | 673 组中 162 组（24.1%），5 个实例已归档 | ✅ 机制生效（排除 INCONCLUSIVE） |
| 行为：V3 All-wrong < V2 All-wrong | 需下降 | 16.0% → 18.0%（+2.0pp） | ❌ 上升 |
| 行为：V3 Pass@8 > V2 Pass@8 | 需恢复 | 84.0% → 82.0%（−2.0pp） | ❌ 继续下降 |
| 行为：mixed→all-wrong 减少 | 需减少 | 19 → 25 | ❌ 增加 |
| 附带：exact greedy 不明显下降 | 需不降 | fresh holdout −1.75pp，p=0.0276 | ❌ 显著下降 |

**H2 的三条行为判据全部不成立，且唯一有利的判据（机制）成立。**

```text
all-wrong zero advantage 确实存在，也被 partial reward 部分修复，
但它不是当前 polarization failure mode 的主要成因。
```

### 必须单独记录的重要发现：partial reward 是一个**错位的代理目标**

这正是 §18 警告的解释陷阱，且在本轮**实际发生了**：

```text
V3 shaped (training) reward  = 0.8735        <- 训练信号持续上升
V3 exact reward (rollout)    = 0.7462   vs V2 0.7575   <- 真实目标下降
V3 avg correct/group         = 5.545    vs V2 6.035    <- 每组答对次数下降
```

即：模型学会了「每个人少错一点」（partial reward ↑），
但**没有**学会「三个人全对」（exact ↓）。

Hamming-style dense reward 与「全对才算对」这一任务目标**并非单调一致**：
从「三人全错」到「只错一人」会提高 partial reward，但任务得分仍然是 0。
优化前者并不必然优化后者。本轮结果为这一风险提供了直接证据。

---

## 8. 产物

```text
src/kk_sft/reward.py                        (compute_reward + TRL 语义的 group advantage)
tests/test_reward.py                        (8 tests，锁定 H2 核心机制)
scripts/train_grpo_trl.py                   (reward.mode + 双轨指标 + rescue 实例归档)
scripts/analyze_h2_reward_rescue.py         (Phase A)
scripts/generate_grpo_v3_holdout.py
scripts/evaluate_grpo_v3_final.py
scripts/analyze_grpo_v3.py / _behavior.py / _stats.py

configs/grpo_v3_partial_reward.yaml

data/raw_grpo_v3/final_holdout.jsonl        (fresh, seed 20260902)
data/processed/grpo_v3_final_holdout.jsonl
data/processed/grpo_v3_manifest.json        (与全部历史数据重叠 = 0)

outputs/grpo_v3_h2_audit/
outputs/grpo_v3_partial_smoke/              (H2_SMOKE_PASS)
outputs/grpo_v3_partial/                    (best = checkpoint-400)
outputs/grpo_v3_final/                      (20 份逐样本预测)
outputs/grpo_v3_final_metrics.json
outputs/grpo_v3_analysis/
outputs/grpo_v3_report.md
```

`outputs/grpo_v1/`、`outputs/grpo_v1_analysis/`、`outputs/grpo_v2_kl001/`
`outputs/grpo_v2_analysis/` 均未修改。

## 9. Fresh holdout 洁净性（seed 20260902）

```text
vs data/raw/{train,val,test}.jsonl               0 / 0 / 0
vs data/raw_v2/{train,val,test}.jsonl            0 / 0 / 0
vs data/raw_grpo_v1/{train,final_holdout}.jsonl  0 / 0
vs data/raw_grpo_v2/final_holdout.jsonl          0
internal_duplicates                              0
universe_key_count                               16999
```

## 10. 已知 caveat

1. Phase A 主样本仅 200 prompts（32 个 all-wrong 组），Wilson CI 下界 0.180；
   gate 判定依赖工程阈值而非显著性。
2. 2-step smoke 只覆盖 16 组，期望 rescued 数约 0.8，因此**无法**验证机制；
   机制证据来自 Phase A（真实 rollout）与正式训练（673 组）。
   脚本已区分「机制未触发」与「无触发机会」。
3. 200-prompt 诊断子集来自 `grpo_v1_final_holdout`（已非 untouched），
   只用于行为对照，泛化结论以 fresh V3 holdout 为准。
4. V2 的 train_metrics.jsonl 生成于双轨字段加入之前，本轮按
   「V2 为 exact 模式 → shaped ≡ exact」做了字段重建。
5. `frac_reward_zero_std`（TRL 原生）阈值 1e-8 偏敏感，本轮以自算的
   `*_zero_variance_ratio`（阈值 1e-6）为准。
