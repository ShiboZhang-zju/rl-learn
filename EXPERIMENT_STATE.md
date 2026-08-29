# EXPERIMENT STATE

本文件是所有 Agent 的**第一入口**。在开始任何工作之前请完整阅读。

最近的完整实验记录不要复制到别处，只需要维护这一个文件。

---

## Last known good commit

```text
87762e2   # Complete GRPO-V2 KL regularization experiment
```

上面的 commit 是**本文档所描述的实验状态被产生的那个 commit**（GRPO-V2 完成）。

```text
GRPO-V1 结果由 30976e5 产生。
GRPO-V2 结果由 87762e2 产生。
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
GRPO-V3        NOT STARTED
```

```text
H1            NOT_SUPPORTED
best          outputs/grpo_v2_kl001/checkpoint-600   (V2 Val exact = 0.7720)
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

## 下一步实验

```text
GRPO-V3
```

**只验证 H2（binary reward sparsity / all-wrong zero advantage）。**

起始点（建议，未执行）：

```text
init checkpoint    = checkpoint-1252
reference          = checkpoint-1252 (frozen, explicit_sft_epoch4)
beta               = 0.01        <- 保留：已证明能显著提升 greedy accuracy
lr = 1e-5, num_generations = 8, temperature = 0.8, top_p = 0.95, 1 epoch
```

唯一计划修改（二选一，需人工确认后再执行）：

```text
方案 a: reward 由 binary exact 改为 partial / shaped
方案 b: 对 all-wrong group 做特殊处理（advantage=0 无法自我纠正）
```

**不要自动开始 GRPO-V3。**

除了上一节列出的禁止项，本轮额外禁止：

```text
beta sweep
把 GRPO-V2 的 greedy 提升当作 H1 被支持的证据
用 grpo_v1_final_holdout 作为最终泛化指标（已非 untouched）
```

新的 untouched 评测集已就绪：

```text
data/processed/grpo_v2_final_holdout.jsonl    (2000, seed 20260901)
  已被 GRPO-V2 使用过；下一轮需另生成新的 holdout。
```

任意关键恢复项失败时，不要自动重新训练，先报告差异。
