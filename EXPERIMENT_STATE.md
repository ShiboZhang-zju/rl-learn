# EXPERIMENT STATE

本文件是所有 Agent 的**第一入口**。在开始任何工作之前请完整阅读。

最近的完整实验记录不要复制到别处，只需要维护这一个文件。

---

## Last known good commit

```text
dc0526a   # Complete GRPO-V3 H2 partial-reward experiment
```

上面的 commit 是**本文档所描述的实验状态被产生的那个 commit**（GRPO-V3 完成）。

```text
GRPO-V1 结果由 30976e5 产生。
GRPO-V2 结果由 87762e2 产生。
GRPO-V3 结果由 dc0526a 产生。
```

恢复时先比较（避免「文档描述状态 A，但 checkout 的代码是状态 B」）：

```bash
git rev-parse HEAD
git merge-base --is-ancestor 87762e202fcd1e5cdc34173e85723bd37424d1fe HEAD && echo IN_SYNC
```

判定：

```text
IN_SYNC                  -> 一致，继续执行 bootstrap
HEAD 落后/与该 commit 分叉 -> STOP，不要训练，先报告差异
```

每次完成正式实验并更新本文件时，一起把这个 commit 更新为**该实验结果所在的 commit**。

---

## 项目目标

```text
Qwen2.5-0.5B-Instruct
3-person Knights & Knaves
Answer-only
SFT → Rollout Audit → GRPO
```

---

## 当前进度

```text
SFT            COMPLETE
Rollout Audit  COMPLETE
GRPO-V1        VALID
GRPO-V1 Stats  COMPLETE
GRPO-V2        COMPLETE   (KL regularization 单变量验证，beta=0.01)
GRPO-V3        COMPLETE   (binary reward sparsity 单变量验证，reward.mode=partial)
GRPO-V4        NOT STARTED
```

```text
H1            NOT_SUPPORTED
best          outputs/grpo_v2_kl001/checkpoint-600   (V2 Val exact = 0.7720)

H2            NOT_SUPPORTED
best          outputs/grpo_v3_partial/checkpoint-400 (V2 Val exact = 0.7740)
```

**不要重新运行已经完成的实验。**

---

## 当前关键 checkpoint

```text
SFT Epoch4 / GRPO init:
outputs/sft_v2_5k_p800/checkpoint-1252

SFT Epoch5 / strongest SFT baseline:
outputs/sft_v2_5k_p800/checkpoint-1565

GRPO-V1 best:
outputs/grpo_v1/checkpoint-200

GRPO-V2 best (KL beta=0.01, V2 Val 0.7720):
outputs/grpo_v2_kl001/checkpoint-600
```

Base model：

```text
Qwen/Qwen2.5-0.5B-Instruct
```

---

## 恢复步骤（新 GPU 机器）

```bash
git clone <repo>          # 或 git pull
cd rl-learn
bash scripts/bootstrap_gpu.sh
```

Checkpoint 以**普通 Git 对象**保存（本仓库未使用 Git LFS），
`clone` / `pull` 之后三个关键 adapter 即可直接获得，无需 `git lfs pull`。

脚本输出 `READY_TO_RESUME` 后，再从下面的「下一步实验」继续。

---

## 环境（已验证，2026-08-29）

```text
python        3.10.19
torch         2.5.1+cu118
cuda          11.8 (available)
gpu           CUDA GPU
transformers  4.57.1
datasets      4.0.0
peft          0.20.0
trl           0.23.0
accelerate    1.14.0
huggingface-hub 0.34.3
safetensors   0.5.3
scipy         1.15.3
numpy         1.26.4
pyyaml        6.0.2
tqdm          4.67.1
```

依赖固定文件：`requirements-gpu.txt`

当前训练使用 **TRL `GRPOTrainer`**（`scripts/train_grpo_trl.py`）。

---

## GRPO-V1 最终结论

Final Holdout（N=2000，greedy）：

```text
Epoch4 Final Holdout: 74.9%
GRPO-V1:              76.0%
Epoch5:               77.5%
```

统计分析：

```text
GRPO - Epoch4 = +1.05pp
McNemar p = 0.2377
bootstrap 95% CI = [-0.60, +2.75]pp

结论：
没有统计证据证明 GRPO 显著优于 Epoch4。
```

详细报告：`outputs/grpo_v1_analysis/grpo_v1_statistical_analysis.md`

### Failure mode

Rollout 分布变化（train/probe 分布，SFT init → GRPO 末期）：

```text
Mixed       38.1% → 21.6%
All-correct 52.0% → 62.0%
All-wrong    9.8% → 16.3%
Pass@8      93.5% → 86.5%
```

结论：

```text
policy sharpening + 双向极化
部分 mixed → all-correct
部分 mixed → all-wrong
```

（200-prompt 复核：mixed→all-correct 38，mixed→all-wrong 17，比例约 2.2:1）

### 类别异常（GT pattern，Final Holdout）

```text
NKK   +17.6pp
KNN    -8.6pp
NNN    -6.7pp
KNK    -5.1pp
```

净增益几乎全部来自 NKK 一类；KNN / NNN / KNK 三类净受损。

---

## GRPO-V2 最终结论（KL regularization，beta=0.01）

配置：`configs/grpo_v2_kl001.yaml`。除 `beta` 0→0.01 外与 V1 完全一致
（同 init、同 seed、同训练数据、同采样、同 LR、同 1 epoch、同 reward）。

### reference 实现方式（重要）

TRL 0.23 在 PEFT + beta>0 时会用 `disable_adapter()` 当 reference，
即 **Qwen Base**，不是 SFT Epoch4。本轮已改为显式 frozen reference：

```text
REFERENCE_MODE = explicit_sft_epoch4
reference = outputs/sft_v2_5k_p800/checkpoint-1252 (frozen, trainable params = 0)
trainer.beta = 0.01, trainer.ref_model is not None = True

训练前 audit: initial_kl = 5.72e-05 (≈0)
训练后 audit: policy_param_delta = 8.82e-04 (>0)
              reference_param_delta = 0.000e+00
```

### 结果（greedy）

Fresh GRPO-V2 Holdout（N=2000，seed 20260901，与全部历史数据零重叠）：

```text
Epoch4   73.85%
Epoch5   75.30%
GRPO-V1  73.50%
GRPO-V2  77.05%      <- 主要泛化指标
```

统计检验（paired）：

```text
V2 - V1     = +3.55pp   McNemar p = 0.0001   95% CI [+1.80, +5.25]pp
V2 - Epoch5 = +1.75pp   McNemar p = 0.0251   95% CI [+0.25, +3.25]pp
V2 - Epoch4 = +3.20pp   McNemar p = 0.0002   95% CI [+1.55, +4.80]pp
```

**GRPO-V2 是第一个同时显著超过 Epoch4 与 Epoch5 的模型。**（V1 未做到）

### 行为对照（固定 200-prompt 诊断子集，8 rollouts，T=0.8）

| Metric | Epoch4 | GRPO-V1 | GRPO-V2 |
|---|---:|---:|---:|
| Pass@8 | 93.5% | 86.5% | 84.0% |
| All-correct | 37.5% | 55.0% | 59.0% |
| All-wrong | 6.5% | 13.5% | 16.0% |
| Mixed | 56.0% | 31.5% | 25.0% |
| Avg unique | 1.825 | 1.390 | 1.335 |

```text
mixed→all-correct / mixed→all-wrong
  Epoch4 → V1:  38 / 17    (2.24 : 1)
  Epoch4 → V2:  43 / 19    (2.26 : 1)     <- 比例未变
```

### H1 判定：NOT_SUPPORTED

```text
KL 确实生效：625/625 steps 有 KL，mean 9.08e-03，max 0.361，从未 ≈0
             -> 排除 INCONCLUSIVE

但两条决定性判据同时不成立：
  All-wrong  13.5% -> 16.0%   (+2.5pp, p=0.4421)   未下降，方向相反
  Pass@8     86.5% -> 84.0%   (-2.5pp)             未恢复，继续下降

且 avg_unique 1.390 -> 1.335，sharpening 未被抑制。
```

结论：

```text
beta=0（缺少 reference KL）不是
「all-correct 与 all-wrong 同时上升」这一 polarization 的主要原因。
```

**已由 dc0526a 提交并推送。** 该结论不再依赖工作区状态。

### Failure Mode

```text
Pass@8:
V1 86.5% → V2 84.0%

All-wrong:
V1 13.5% → V2 16.0%

因此：
KL 提升了 greedy generalization，
但没有解决错误方向 polarization。
```

副产物（必须分开记账）：

```text
KL 显著提升了 greedy accuracy（+3.55pp vs V1，p=0.0001）。
beta>0 值得保留，但它改善的是 greedy 决策质量，
不是 H1 假设的「减少错误方向 sharpening」。
```

关于强度的诚实说明：

```text
beta * kl_mean ≈ 9.1e-05，只占 loss 的很小一部分。
本轮不判 INCONCLUSIVE（policy 差异可测量且 CI 不跨 0），
但不能排除更大的 beta 会改变行为结论。
禁止自动跑 beta sweep；若要提高强度需在此显式登记。
```

详细报告：`outputs/grpo_v2_report.md`

### 训练动态备注（影响所有历史对比）

```text
V1 的 zero_variance_group_ratio 因采集 bug 全为 null（读了不存在的
  "train/frac_reward_zero_std" 键）——本轮已修正。
V1 的 loss / grad_norm / entropy 存在 off-by-one（写在 on_step_end，
  落后 TRL 日志一步）——本轮改为 on_log 采集。
V1 的 reward 函数统计量逐步对齐，可直接比较。
```

### 实现纠正（erratum）

```text
TRL 0.23 scale_rewards="group"
使用 sample std (ddof=1)，不是 population std。

V1 历史产物不重写；
该修正作为实验记录 erratum 保留。
```

依据（`trl/trainer/grpo_trainer.py`）：

```python
std_rewards = rewards.view(-1, self.num_generations).std(dim=1)   # torch.std 默认 correction=1
advantages  = advantages / (std_rewards + 1e-4)
```

即 group 内归一化用的是**无偏样本标准差**，且 TRL 不对 std==0 特判，
而是给分母加 `1e-4`，零方差组自然得到 advantage ≈ 0。

`scale_rewards="group"` 本身**未改动**，只修正了注释与报告表述。
V1 报告中的 "population std" 表述有误，已在
`outputs/grpo_v2_report.md` 中以 erratum 形式记录。

---

## 当前待验证假设

按顺序：

```text
H1:
beta=0 缺少 reference KL，
导致 policy 过度 sharpening / drift。
  -> 已验证：NOT_SUPPORTED（GRPO-V2，2026-08-29）

H2:
binary exact reward 太稀疏，
all-wrong group advantage=0，无法自我纠正。
  -> 下一个

H3:
generator 中残余 structure→label shortcut
可能被 RL 放大。
```

---

## GRPO-V3 最终结论（binary reward sparsity，reward.mode=partial）

配置：`configs/grpo_v3_partial_reward.yaml`。除 `reward.mode` exact→partial 外
与 GRPO-V2 **完全一致**（同 init、同 reference、同 beta=0.01、同 seed、
同训练数据、同采样、同 LR、同 1 epoch、同 625 steps）。

Partial reward = `correct_person_count / 3`：`1.0 / 0.6667 / 0.3333 / 0.0`，
invalid = 0.0。无 alpha、无 format bonus、无 length reward。

### Phase A：零训练 reward-rescue audit（未做任何新 rollout）

复用 `outputs/grpo_v2_analysis/rollout_D_final200.jsonl`（真实模型输出）：

```text
exact all-wrong groups                32
  ├─ partial 仍然 zero-variance        22
  └─ partial 恢复 non-zero variance    10

reward rescue rate = 10 / 32 = 31.25%   [Wilson 95% CI 0.180, 0.486]
Gate: 31.2% > 25% -> CONTINUE_TO_PHASE_B
```

未被 rescue 的 22 组：8 次 rollout 全部预测**同一**错误 pattern（partial 恒定）。
被 rescue 的 10 组：8 次中出现 2 种错误 pattern（例 gt=NNN → [NNK×7, NKK×1]）。

上下文：Epoch4 7/13 = 53.8%，V1 7/27 = 25.9%，V2 probe 5/14 = 35.7%。

### 训练期机制（真实 rollout，673 组）

```text
exact-all-wrong groups seen : 673
rescued (regained variance) : 162     -> 24.1%
rescued examples archived   : 5       (见 audit.json)
```

rescue rate 随训练下降：34.7%（1-100）→ 18.2%（601-625）。
policy 越 sharpen，all-wrong group 越同质化，越难被 rescue。

### 结果（greedy，指标恒为 exact accuracy）

Fresh GRPO-V3 Holdout（N=2000，seed 20260902，与全部历史数据零重叠）：

```text
Epoch4      73.35%
Epoch5      76.25%
GRPO-V1     75.50%
GRPO-V2     76.95%
GRPO-V3     75.20%
```

统计检验（paired）：

```text
V3 - V2     = -1.75pp   McNemar p = 0.0276   95% CI [-3.25, -0.25]pp   <- 显著变差
V3 - Epoch5 = -1.05pp   McNemar p = 0.1919   95% CI [-2.55, +0.40]pp   (跨 0)
V3 - Epoch4 = +1.85pp   McNemar p = 0.0286   95% CI [+0.25, +3.45]pp
```

V2 Holdout 上：V3 − V2 = −2.75pp，p = 0.0004。

### 行为对照（固定 200-prompt 子集，8 rollouts，T=0.8）

| Metric | Epoch4 | GRPO-V1 | GRPO-V2 | GRPO-V3 |
|---|---:|---:|---:|---:|
| Pass@8 | 93.5% | 86.5% | 84.0% | 82.0% |
| All-correct | 37.5% | 55.0% | 59.0% | 53.0% |
| All-wrong | 6.5% | 13.5% | 16.0% | 18.0% |
| Mixed | 56.0% | 31.5% | 25.0% | 29.0% |
| Avg unique | 1.825 | 1.390 | 1.335 | 1.370 |
| Avg correct/group | 5.335 | 5.840 | 6.035 | 5.545 |

```text
mixed→all-correct / mixed→all-wrong
  Epoch4 → V1:  38 / 17
  Epoch4 → V2:  43 / 19
  Epoch4 → V3:  34 / 25      <- 两个方向都更差
```

### H2 判定：NOT_SUPPORTED

```text
机制成立：673 个 all-wrong group 中 162 个（24.1%）恢复了 non-zero variance，
          5 个真实 group 实例已归档 -> 排除 INCONCLUSIVE

但三条行为判据全部不成立：
  All-wrong       16.0% -> 18.0%   (+2.0pp, p=0.5034)   未下降
  Pass@8          84.0% -> 82.0%   (-2.0pp)             未恢复
  mixed→all-wrong 19    -> 25                           反而增加

且 exact greedy accuracy 显著下降：
  fresh holdout 76.95% -> 75.20%   (-1.75pp, p=0.0276)
```

结论：

```text
H2 = NOT_SUPPORTED

Partial reward 确实恢复了部分 exact-all-wrong group 的
reward variance / non-zero advantage，
因此机制干预真实生效。

但 All-wrong、Pass@8、mixed→wrong 和最终 exact accuracy
均未改善，反而恶化。

因此 all-wrong zero-advantage 虽然真实存在，
但不是当前 polarization failure 的主要原因。
```

### 重要发现：partial reward 是错位的代理目标（Goodhart）

```text
Partial reward 出现 proxy misalignment / Goodhart：

shaped training reward ↑
但 exact task reward ↓

因此后续禁止继续做类似 partial/Hamming reward shaping，
除非提出新的独立假设。
```

数据：

```text
V3 shaped (training) reward = 0.8735          <- 训练信号持续上升
V3 exact reward (rollout)   = 0.7462  vs V2 0.7575   <- 真实目标下降
V3 avg correct/group        = 5.545   vs V2 6.035
```

Hamming-style dense reward 与「三人全对才算对」**非单调一致**：
从「三人全错」到「只错一人」提高 partial reward，但任务得分仍为 0。
模型学会了「每人少错一点」，没有学会「三人全对」。

### 训练动态备注

```text
V2 的 train_metrics.jsonl 生成于双轨字段加入之前；
本轮按「V2 为 exact 模式 -> shaped ≡ exact」重建了 exact_*/shaped_* 字段。
TRL 原生 frac_reward_zero_std 阈值 1e-8 偏敏感，
本轮以自算的 *_zero_variance_ratio（阈值 1e-6）为准。
```

详细报告：`outputs/grpo_v3_report.md`

---

## 当前待验证假设

按顺序：

```text
H1:
beta=0 缺少 reference KL，
导致 policy 过度 sharpening / drift。
  -> 已验证：NOT_SUPPORTED（GRPO-V2，2026-08-29）

H2:
binary exact reward 太稀疏，
all-wrong group advantage=0，无法自我纠正。
  -> 已验证：NOT_SUPPORTED（GRPO-V3，2026-08-29）

H3:
GRPO 是否放大了 generator 中残余的
structure/operator → answer-pattern shortcut。
  -> 下一个（H1/H2 均不支持后，这是唯一剩下的候选）
```

H1 与 H2 **均不支持**，两者的干预（KL、partial reward）都真实生效，
但都没能减少错误方向 polarization。这提示：

```text
polarization 可能不是由「缺少约束」或「缺少信号」造成的，
而可能来自被优化目标本身与任务目标的错位（partial reward 的证据支持这一点），
或来自 generator / 数据分布层面的 shortcut。
```

## 下一步实验

```text
GRPO-V4
```

**只验证 H3（generator / policy shortcut）。不要自动开始。**

起始点（建议，未执行）：

```text
init checkpoint    = checkpoint-1252
reference          = checkpoint-1252 (frozen, explicit_sft_epoch4)
beta               = 0.01
reward.mode        = exact           <- 回到 exact：partial 已证明有害
lr = 1e-5, num_generations = 8, temperature = 0.8, top_p = 0.95, 1 epoch
```

H3 的方向（需人工确认后再执行）：

```text
a: 审计 generator 是否存在 structure/operator -> answer-pattern 的可利用相关性
b: 检查 GRPO 是否放大了 a 中的 shortcut
c: 若成立，考虑在数据层面（而非 reward 层面）修正
```

**不要自动开始 GRPO-V4。**

本轮额外禁止：

```text
任何新的 reward shaping（partial 已证明有害）
reward alpha sweep
beta sweep
LR sweep
第二 epoch
修改训练集 / generator（在 H3 审计完成前）
```

新的评测集状态：

```text
data/processed/grpo_v2_final_holdout.jsonl  已被 GRPO-V2/V3 使用（非 untouched）
data/processed/grpo_v3_final_holdout.jsonl  已被 GRPO-V3 使用（非 untouched）
  下一轮需另生成新的 holdout（建议 seed 20260903）。
```

任意关键恢复项失败时，不要自动重新训练，先报告差异。
