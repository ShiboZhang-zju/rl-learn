# GRPO-V2 报告：KL regularization 单变量验证（H1）

日期：2026-08-29
状态：**GRPO_V2_KL_COMPLETE**
H1 判定：**NOT_SUPPORTED**

---

## 0. 一句话结论

`beta=0.01` + 正确的 SFT-Epoch4 reference **确实生效**（KL 可测、policy 明显改变），并且
**显著提升了 greedy accuracy**（fresh holdout +3.55pp vs V1，p=0.0001）；
但它**没有缓解** H1 所针对的失败模式：`all-wrong` 不降反升，`Pass@8` 继续下降。

因此：**beta=0 不是「错误方向 polarization」的主要原因。**

---

## 1. reference 实际是谁

TRL 0.23 在 PEFT + `beta>0` 时会走 `grpo_trainer.py` 的
`elif is_peft_model(model): self.ref_model = None` 分支，并在算 KL 时用
`disable_adapter()`，即 **reference 会退化成 Qwen Base**，而不是 SFT Epoch4。
本轮按规格显式加载并挂载了 frozen reference。

| 项 | 值 |
|---|---|
| `REFERENCE_MODE` | `explicit_sft_epoch4` |
| policy init | `outputs/sft_v2_5k_p800/checkpoint-1252` (SFT Epoch4) |
| reference | `outputs/sft_v2_5k_p800/checkpoint-1252` (**frozen**) |
| reference_trainable_params | **0** |
| `trainer.beta` | **0.01** |
| `trainer.ref_model is not None` | True |

训练前 audit（`outputs/grpo_v2_kl001/audit/reference_audit.json`，2 个真实 prompt，
policy-at-init 与 reference 在**同一 token 序列**上比对）：

```text
mean_abs_logprob_diff = 2.341e-03
max_abs_logprob_diff  = 6.832e-02
initial_kl            = 5.725e-05      -> ≈ 0，REFERENCE_AUDIT_PASS
```

差异来自 bf16 下两次独立 forward 的数值噪声；k3 估计的 KL 为 `5.7e-05`，确认
`policy_init ≈ reference`。

训练后审计（`outputs/grpo_v2_kl001/audit.json`）：

```text
policy_param_delta_max    = 8.815e-04   (> 0，policy 确实更新)
reference_param_delta_max = 0.000e+00   (reference 完全未变)
kl_logged = True  kl_all_finite = True  loss_all_finite = True  grad_norm_all_finite = True
KL_SMOKE_PASS
```

---

## 2. beta 是否真的生效

**生效。** 证据链完整，不是「KL ≈ 0」的情形：

```text
KL 覆盖 625/625 steps
kl_first = 3.92e-05
kl_last  = 6.31e-03
kl_mean  = 9.08e-03
kl_max   = 3.61e-01      (个别 step 的尖峰)
```

`beta * kl_mean ≈ 9.1e-05`，与 per-token loss 量级（1e-4 ~ 1e-3）可比，
属于**弱到中等强度**约束，但足以产生可测量的 policy 差异（下节）。

---

## 3. KL trajectory（每 50 步抽样）

| Step | 1 | 50 | 100 | 150 | 200 | 250 | 300 | 350 | 400 | 450 | 500 | 550 | 600 | 625 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| KL | 4e-05 | 1.12e-03 | 4.70e-03 | 3.74e-03 | 1.06e-02 | 4.50e-03 | 3.51e-03 | 2.93e-03 | 2.99e-03 | 5.36e-03 | 6.83e-03 | 5.91e-03 | 6.72e-03 | 6.31e-03 |

形态：从 0 快速爬升到 ~5e-3，中段（300–400）回落到 ~3e-3，后段（500–625）稳定在 ~6e-3。
**没有发散，也从未回到 0。**

---

## 4. 训练动态：V2 vs V1（每 100 步分桶均值）

| bucket | reward_mean | mixed | all-correct | all-wrong | zero-var | avg correct/group | avg unique | entropy | **KL** | loss | grad_norm | peak VRAM (GB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1–100 | 0.7286 | 0.3800 | 0.5150 | 0.1050 | 0.6200 | 5.829 | 1.504 | 0.0176 | 0.0046 | 0.0001 | 0.811 | 26.08 |
| 101–200 | 0.7367 | 0.2712 | 0.5950 | 0.1338 | 0.7288 | 5.894 | 1.359 | 0.0129 | 0.0080 | 0.0006 | 0.981 | 27.71 |
| 201–300 | 0.7489 | 0.2412 | 0.6262 | 0.1325 | 0.7588 | 5.991 | 1.311 | 0.0116 | 0.0091 | 0.0002 | 0.984 | 28.17 |
| 301–400 | 0.7458 | 0.2487 | 0.6162 | 0.1350 | 0.7512 | 5.966 | 1.319 | 0.0117 | 0.0106 | 0.0001 | 1.205 | 28.17 |
| 401–500 | 0.7483 | 0.2338 | 0.6312 | 0.1350 | 0.7662 | 5.986 | 1.304 | 0.0107 | 0.0091 | 0.0004 | 1.083 | 28.17 |
| 501–600 | 0.7605 | 0.2200 | 0.6350 | 0.1450 | 0.7800 | 6.084 | 1.274 | 0.0111 | 0.0132 | 0.0003 | 1.266 | 28.17 |
| 601–625 | 0.7575 | 0.2200 | 0.6250 | 0.1550 | 0.7800 | 6.060 | 1.300 | 0.0106 | 0.0085 | 0.0010 | 1.152 | 28.17 |

V1 同表（reference）：

| bucket | reward_mean | mixed | all-correct | all-wrong | avg correct/group | avg unique | entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1–100 | 0.7297 | 0.3812 | 0.5200 | 0.0988 | 5.838 | 1.525 | 0.0174 |
| 101–200 | 0.7289 | 0.2800 | 0.5763 | 0.1437 | 5.831 | 1.371 | 0.0135 |
| 201–300 | 0.7558 | 0.2338 | 0.6300 | 0.1363 | 6.046 | 1.308 | 0.0107 |
| 301–400 | 0.7466 | 0.2250 | 0.6388 | 0.1363 | 5.973 | 1.286 | 0.0105 |
| 401–500 | 0.7567 | 0.2225 | 0.6400 | 0.1375 | 6.054 | 1.279 | 0.0102 |
| 501–600 | 0.7527 | 0.2150 | 0.6338 | 0.1512 | 6.021 | 1.276 | 0.0107 |
| 601–625 | 0.7544 | 0.2150 | 0.6250 | 0.1600 | 6.035 | 1.255 | 0.0102 |

**两条轨迹几乎重合。** V2 末段 `mixed` 0.2200 vs V1 0.2150，`all-correct` 0.6250 vs 0.6250，
`all-wrong` 0.1550 vs 0.1600。KL 约束**没有改变 rollout 分布的演化方向**。

> 注：V1 的 `zero_variance_group_ratio` 恒为 0，是其采集 bug（读了不存在的
> `train/frac_reward_zero_std` 键）导致该字段为 null，本轮已修正；V1 的
> `loss/grad_norm/entropy` 存在 off-by-one（写在 `on_step_end`，落后 TRL 日志一步），
> 本轮改为 `on_log` 采集。V1 的 reward 函数统计量（`reward_mean`/`mixed`/`all_correct`/
> `all_wrong`/`avg_unique`）是逐步对齐的，可直接比较。

**V2 Val（greedy，500 prompts）**

| Step | 0 | 100 | 200 | 300 | 400 | 500 | 600 | 625 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| exact | 0.7220 | 0.7540 | 0.7540 | 0.7660 | 0.7600 | 0.7420 | **0.7720** | 0.7620 |
| format | 1.0000 | 1.0000 | 1.0000 | 0.9980 | 1.0000 | 0.9940 | 1.0000 | 1.0000 |
| parse | 1.0000 | 1.0000 | 1.0000 | 0.9980 | 1.0000 | 0.9940 | 1.0000 | 1.0000 |

## 5. checkpoint selection

规则：V2 Val 最高；并列取更早。

```text
best = outputs/grpo_v2_kl001/checkpoint-600   (V2 Val exact = 0.7720)
```

仅在 best checkpoint 选出之后才打开 fresh holdout。

---

## 6. 四模型统一评测（greedy，同一 evaluator）

格式：`exact / format / parse`

| Model | V2 Val (500) | V2 Test (1000) | V1 Holdout (2000) | **V2 Fresh Holdout (2000)** |
|---|---:|---:|---:|---:|
| A. SFT Epoch4 (ckpt-1252) | 0.7220 / 1.000 / 1.000 | 0.7570 / 0.999 / 0.999 | 0.7490 / 1.000 / 1.000 | 0.7385 / 1.000 / 1.000 |
| B. SFT Epoch5 (ckpt-1565) | 0.7480 / 1.000 / 1.000 | 0.7790 / 1.000 / 1.000 | 0.7750 / 1.000 / 1.000 | 0.7530 / 1.000 / 1.000 |
| C. GRPO-V1 (ckpt-200) | 0.7660 / 1.000 / 1.000 | 0.7500 / 0.996 / 0.996 | 0.7595 / 0.998 / 0.998 | 0.7350 / 0.997 / 0.997 |
| D. **GRPO-V2 (ckpt-600)** | **0.7720** / 1.000 / 1.000 | **0.7840** / 0.996 / 0.996 | **0.7720** / 0.998 / 0.998 | **0.7705** / 0.998 / 0.998 |

一致性交叉验证：本轮重跑的 A/B/C 与 V1 报告完全吻合
（Epoch4 74.9%、Epoch5 77.5%、GRPO-V1 76.0% on V1 holdout）。

### 统计检验（paired，fresh holdout）

| 对比 | delta | McNemar exact p | bootstrap 95% CI |
|---|---:|---:|---:|
| V2 − V1 | **+3.55pp** | **0.0001** | [+1.80, +5.25]pp |
| V2 − Epoch5 | **+1.75pp** | **0.0251** | [+0.25, +3.25]pp |
| V2 − Epoch4 | **+3.20pp** | **0.0002** | [+1.55, +4.80]pp |
| V2 − V1（V1 holdout） | +1.25pp | 0.1817 | [−0.50, +3.00]pp（跨 0） |

**GRPO-V2 在 fresh holdout 上同时显著超过 V1 与最强的 SFT baseline（Epoch5）。**
这是 V1 没有做到的（V1 vs Epoch4 无统计显著性）。

---

## 7. V1 vs V2 行为对照（固定 200-prompt 诊断子集）

采样与 V1 完全一致：8 rollouts，temperature 0.8，top_p 0.95，max_new_tokens 64，
seed 20260828，bf16；子集 = `grpo_v1_final_holdout` 前 200 条。

| Metric | Epoch4 | GRPO-V1 | GRPO-V2 |
|---|---:|---:|---:|
| Mean reward | 0.6669 | 0.7300 | **0.7544** |
| **Pass@8** | 93.5% | 86.5% | **84.0%** |
| Mixed | 56.0% | 31.5% | 25.0% |
| All-correct | 37.5% | 55.0% | **59.0%** |
| **All-wrong** | 6.5% | 13.5% | **16.0%** |
| Avg unique | 1.825 | 1.390 | **1.335** |
| Avg correct/group | 5.335 | 5.840 | 6.035 |

显著性（V1 → V2，同 200 条 prompt，McNemar）：

| Metric | V1 | V2 | delta | V1-only / V2-only | p |
|---|---:|---:|---:|---:|---:|
| all-wrong | 0.1350 | 0.1600 | **+2.50pp** | 11 / 16 | 0.4421 |
| all-correct | 0.5500 | 0.5900 | +4.00pp | 19 / 27 | 0.3020 |
| mixed | 0.3150 | 0.2500 | −6.50pp | 38 / 25 | 0.1299 |

### mixed → all-correct / mixed → all-wrong

| | mixed→all-correct | mixed→all-wrong | 比值 |
|---|---:|---:|---:|
| Epoch4 → GRPO-V1 | 38 | 17 | 2.24 : 1 |
| Epoch4 → **GRPO-V2** | 43 | 19 | 2.26 : 1 |

V2 把更多 mixed 推到了两端（43+19=62 vs V1 的 38+17=55），
但 **分流比例完全没变**——错误方向的那一支没有被 KL 抑制。

---

## 8. class-specific regression 是否改善

V2 − V1 在 fresh holdout 上按 GT pattern 的 delta（pp）：

```text
KKK  -2.65    KKN +13.10    KNK +14.56    KNN +13.77
NKK -11.93    NKN  -5.00    NNK  -0.77    NNN  +6.90
```

V1 当时的模式是「净增益几乎全部来自 NKK，KNN/NNN/KNK 净受损」。
V2 的模式是「增益来自 KKN/KNK/KNN，而 **NKK 净受损 −11.9pp**」。

**结论：class 间此消彼长的极化结构依旧存在，只是换了一批受益/受损类别，并未被修复。**

fresh holdout 上的 8-class accuracy：

| Model | KKK | KKN | KNK | KNN | NKK | NKN | NNK | NNN | macro |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Epoch4 | 0.909 | 0.619 | 0.678 | 0.668 | 0.638 | 0.754 | 0.755 | 0.888 | 0.7386 |
| Epoch5 | 0.936 | 0.710 | 0.701 | 0.684 | 0.753 | 0.725 | 0.713 | 0.797 | 0.7524 |
| GRPO-V1 | 0.935 | 0.618 | 0.617 | 0.599 | **0.847** | 0.758 | 0.747 | 0.780 | 0.7377 |
| GRPO-V2 | 0.905 | 0.749 | **0.762** | 0.737 | 0.727 | 0.711 | 0.731 | 0.849 | **0.7715** |

V2 的预测分布更贴近 GT 分布（V1 明显过预测 NKK：325 vs GT 243；V2 为 229），
说明 KL 确实抑制了 V1 那种极端类别偏移——但**没有转化为 rollout 多样性的恢复**。

---

## 9. H1 判定

### H1 NOT_SUPPORTED

判定依据（对照 §15 规则）：

| 判据 | 要求 | 实测 | 结果 |
|---|---|---|---|
| KL 生效 | — | KL 全程非零、均值 9.08e-03、max 0.361 | ✅ 生效（排除 INCONCLUSIVE） |
| All-wrong | 明显下降 | 13.5% → 16.0%（**+2.5pp**，p=0.4421） | ❌ 未改善，方向相反 |
| Pass@8 | 明显恢复 | 86.5% → 84.0%（**−2.5pp**） | ❌ 未恢复，继续下降 |
| All-correct | 无明显损失 | 55.0% → 59.0%（+4.0pp） | ✅ 无损失 |
| Greedy accuracy | 无明显下降 | 0.7350 → 0.7705（**+3.55pp**，p=0.0001） | ✅ 显著上升 |

**H1 的两条决定性判据（all-wrong ↓、Pass@8 ↑）同时不成立**，且
`avg_unique` 继续从 1.390 降到 1.335，说明 **sharpening 没有被抑制，反而略强**。

因此：

```text
beta=0（缺少 reference KL）不是「all-correct 与 all-wrong 同时上升」
这一 polarization 现象的主要原因。
```

### 一个必须分开记录的重要副产物

KL 约束带来了 **V1 未能取得的、统计显著的 greedy accuracy 提升**：
V2 在 fresh holdout 上同时显著优于 V1（+3.55pp, p=0.0001）和 Epoch5（+1.75pp, p=0.0251）。
所以 **`beta>0` 值得保留**，但它改善的是「greedy 决策质量」，
**不是** H1 假设的「减少错误方向 sharpening」。这两件事应分开记账。

### 关于强度的诚实说明

`beta=0.01` 的惩罚项量级（`beta * kl_mean ≈ 9.1e-05`）只相当于 loss 的很小一部分。
本轮不判 INCONCLUSIVE，因为 policy 差异是可测量的（+3.55pp，CI 不跨 0）。
但**不能排除更大的 beta 会改变行为结论**。按 §16，**本轮不自动跑 beta sweep**；
若下一轮要提高强度，需在 EXPERIMENT_STATE.md 中显式登记。

---

## 10. 产物

```text
configs/grpo_v2_kl001.yaml
scripts/train_grpo_trl.py                 (beta 可配置 + 显式 reference + 采集修正)
scripts/generate_grpo_v2_holdout.py
scripts/evaluate_grpo_v2_final.py
scripts/analyze_grpo_v2.py
scripts/analyze_grpo_v2_behavior.py
scripts/analyze_grpo_v2_stats.py

data/raw_grpo_v2/final_holdout.jsonl              (fresh, seed 20260901)
data/processed/grpo_v2_final_holdout.jsonl
data/processed/grpo_v2_manifest.json              (重叠全为 0)

outputs/grpo_v2_kl001/                    (训练产物)
outputs/grpo_v2_kl001_smoke/              (2-step smoke + audit.json)
outputs/grpo_v2_final/                    (16 份逐样本预测)
outputs/grpo_v2_final_metrics.json
outputs/grpo_v2_analysis/                 (训练动态 / 行为对照 / 统计检验)
outputs/grpo_v2_report.md
```

`outputs/grpo_v1/` 与 `outputs/grpo_v1_analysis/` 未被修改。

## 11. Fresh holdout 的洁净性

`grpo_v2_manifest.json` 记录的去重校验：

```text
vs data/raw/{train,val,test}.jsonl              0 / 0 / 0
vs data/raw_v2/{train,val,test}.jsonl           0 / 0 / 0
vs data/raw_grpo_v1/{train,final_holdout}.jsonl 0 / 0
internal_duplicates                             0
```

## 12. 已知 caveat

1. 200-prompt 诊断子集来自 `grpo_v1_final_holdout`（已非 untouched），按 §14 仅用于
   **行为对照**，不作为泛化指标；泛化结论以 fresh holdout 为准。
2. V1 的 `zero_variance_group_ratio` 因采集 bug 不可用（全 null），本轮已修正。
3. V1 的 `loss` / `grad_norm` / `entropy` 存在 off-by-one，本轮已修正为 `on_log` 采集。
4. §14 的 D 组采样使用与 A/C 相同的脚本、种子与参数（batch-size 取默认 8）。
