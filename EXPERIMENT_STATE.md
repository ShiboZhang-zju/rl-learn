# EXPERIMENT STATE

本文件是所有 Agent 的**第一入口**。在开始任何工作之前请完整阅读。

最近的完整实验记录不要复制到别处，只需要维护这一个文件。

---

## Last known good commit

```text
d3bb35c   # Complete H5 causal intervention: K=8 vs K=16
```

上面的 commit 是**本文档所描述的实验状态被产生的那个 commit**（K16 因果干预完成）。

```text
GRPO-V1 结果由 30976e5 产生。
GRPO-V2 结果由 87762e2 产生。
GRPO-V3 结果由 dc0526a 产生。
H3 诊断结果由 63abc47 产生。
H4 诊断结果由 247cb21 产生。
H5 诊断结果由 b633eda 产生。
K16 因果干预结果由 d3bb35c 产生。
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
H3 Diagnosis   COMPLETE   (structure/operator shortcut，零训练)
H4 Diagnosis   COMPLETE   (8-way probability landscape，零训练)
H5 Diagnosis   COMPLETE   (finite-K sampling feedback，零训练)
K8 vs K16      COMPLETE   (H5 因果干预，K=8 -> K=16)
GRPO-V4        NOT STARTED
```

```text
H1            NOT_SUPPORTED
best          outputs/grpo_v2_kl001/checkpoint-600   (V2 Val exact = 0.7720)

H2            NOT_SUPPORTED
best          outputs/grpo_v3_partial/checkpoint-400 (V2 Val exact = 0.7740)

H3            NOT_SUPPORTED   (零训练诊断，未训练 V4)

H4            SUPPORTED       (零训练诊断，未训练 V4)
              GRPO 主要在 8 个合法答案上重分配/sharpen 概率，
              而非广泛扩大 gold answer 的 high-rank support

H5            SUPPORTED       (零训练诊断，6/7 判据，判据 4 未检验)

              H5 is supported as a predictive/path-dependent mechanism,
              not yet established causally.

H5 CAUSAL     NOT_SUPPORTED   (K=8 -> K=16 干预，2026-08-29)
              K16 确实提高 sampling coverage，
              但 lower tail / p10 / common-eval all-wrong / Pass@8 均无改善；
              低支持区(initial q<0.20)反而显著更差
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

## H3 诊断结论（零训练，2026-08-29）

分析集：GRPO-V3 Fresh Holdout（N=2000，seed 20260902）。
复用五个模型的已有逐样本预测，**未重新推理、未训练**。
结构特征复用 `scripts/audit_dataset_features.row_features`。
脚本：`scripts/audit_h3_shortcut.py`，报告：`outputs/grpo_h3_shortcut_audit/h3_shortcut_report.md`。

### 核心结论

```text
H3 = NOT_SUPPORTED

数据 shortcut 存在，但没有证据表明 GRPO 在放大它。
```

```text
Data structure→GT shortcut exists but is weak.

No consistent evidence that:
GRPO amplifies structure→prediction association.

Conditional on GT:
GRPO does not show stronger structure dependence than SFT.

Epoch5 control explains observed generic training effects.

H3 = NOT_SUPPORTED.
```

### Q1 数据 shortcut 存在但很弱

```text
same_count        MI=0.0697  (permutation null 0.0133) -> 净 +0.0564 = GT 熵的 3.35%
different_count   MI=0.0608  (null 0.0113)             -> 净 +0.0494 = 2.92%
top_ops 序列      MI=0.7122  (null 0.4183)             -> 净 +0.2940，但偏差占 59%
not / person_is / expression_depth                     -> permutation p=0.10~0.93，无信号
```

### Q2 structure→prediction 未增强

```text
H(prediction) 五模型几乎相同（2.074~2.078），无熵混淆，可直接比较

top_ops_multiset 原始 MI:  0.3400 -> 0.3244(e5) -> 0.3355(v1) -> 0.3262(v2) -> 0.3151(v3)
                          单调下降

bootstrap vs Epoch4:  仅 v1 top_ops +0.0238 显著；其余全部 n.s.
bootstrap vs Epoch5:  仅 v1 top_ops +0.0255、same_count +0.0090 显著；v2/v3 全部 n.s.
```

### Q3 控制 GT 后：V2 显著下降，GRPO 与 Epoch5 无法区分

```text
conditional MI delta vs Epoch4:
  v2  top_ops          -0.0643  CI [-0.100, -0.026]   显著下降
  v2  top_ops_multiset -0.0341  CI [-0.067, -0.002]   显著下降
  v2  op_signature     -0.0554  CI [-0.099, -0.014]   显著下降
  v1 / v3              全部 n.s.
  Epoch5 - Epoch4      top_ops -0.0421 显著下降（纯 SFT 多训也下降）

conditional MI delta vs Epoch5:
  v1 / v2 / v3         全部 n.s.
```

**关键控制：Epoch5。** `expression_nodes` conditional NMI
Epoch4 0.1595 → **Epoch5 0.1778**，V1 0.1777 / V2 0.1819 / V3 0.1751，
GRPO 与「只是多训一个 epoch 的 SFT」完全重合。
这些变化是「训练更多」的通用效应，不是 RL 特有。

### Q4 类别异常无法归因到具体 structure

```text
V1 的 NKK 上升分散在 >=5 个 signature，每个只贡献 3-5 个样本
V2 的 KKN/KNN 上升同样分散，每个 3-5 个样本
没有任何单一结构能解释整体类别位移
```

### Q4b WC 与 CW 在结构上无分离

```text
卡方独立性检验（signature x {WC, CW}）:
  v1 p=0.9193   v2 p=0.5156   v3 p=0.6509
同一个 signature 常同时出现在 WC 与 CW 两个列表
  （如 v2 的 not|or|person_is WC=8 CW=7；different|or|same WC=10 CW=10）
```

### 唯一反向观察（如实记录）

```text
V1 的 top_ops 原始 MI 相对 Epoch4(+0.0238) 与 Epoch5(+0.0255) 都显著上升。
但原始 MI 混杂「结构→GT→预测」的间接路径（模型越准该项越大），
控制 GT 后同一对比变为 n.s.（+0.0257, CI [-0.0147, +0.0670]），
且 V2/V3 未复现。
V1 是唯一在原始指标上出现 structure 依赖上升的模型，
与 V1 最强的 NKK 类别异常在时间上吻合。
```

### 为何未判 INCONCLUSIVE

```text
- signature 样本量：top_ops 只有 5 个 signature 达 N>=20（覆盖 5.6%），
  但改用同为纯结构特征的 top_ops_multiset 后有 38 个、覆盖 91.8%，
  且 MI 分析使用全部 2000 样本，功效充足。
- 指标方向：除上述 V1 原始 MI 一项外，其余指标方向一致指向「无增强」。
```

## 当前待验证假设

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
  -> 已验证：NOT_SUPPORTED（零训练诊断，2026-08-29）
```

三条候选假设**全部不支持**：三者的干预（KL、partial reward）都真实生效，
或机制真实存在（structure→GT 相关性），但都没能解释或缓解 polarization。

H1 与 H2 **均不支持**，两者的干预（KL、partial reward）都真实生效，
但都没能减少错误方向 polarization。这提示：

```text
polarization 可能不是由「缺少约束」或「缺少信号」造成的，
而可能来自被优化目标本身与任务目标的错位（partial reward 的证据支持这一点），
或来自 generator / 数据分布层面的 shortcut。
```

## H4 诊断结论（零训练，2026-08-29）

分析集：GRPO-V3 Fresh Holdout（N=2000）+ 固定 200-prompt 诊断子集。
方法：对 8 个 canonical answer completion 做 teacher-forced scoring，
用 logsumexp 归一化得到 answer-space `q`。**不使用 temperature sampling、不训练。**
脚本：`scripts/audit_h4_probability_landscape.py`；
报告：`outputs/grpo_h4_probability_audit/h4_probability_report.md`。

### 核心结论

```text
H4 = SUPPORTED

GRPO 的主要可观测作用更接近 answer-space probability
redistribution / sharpening，
而不是广泛扩大 gold answer 的 high-rank support。
```

```text
GRPO primarily reshapes/sharpens the existing 8-way answer distribution.

V2:
Top1: 74.90% -> 77.75%
Top3: 97.70% -> 97.85%

149/152 promoted samples already had gold rank <= 3 at Epoch4.

mixed->all-correct:
gold probability increases and entropy decreases.

mixed->all-wrong:
gold probability decreases and entropy also decreases.

Therefore polarization is bidirectional probability sharpening.
```

**不要**写成「GRPO 没有学到推理」——本实验无法证明这一点。

### 关键数字

```text
normalized entropy   0.2560 -> 0.1208   (V2)   CI EXCL0
effective support    1.8446 -> 1.3527   (V2)   CI EXCL0
top1 margin          2.7796 -> 5.0221   (V2)   CI EXCL0

Top1 coverage        0.7490 -> 0.7775   +2.85pp  CI [+1.35, +4.40]pp  EXCL0
Top2 coverage        0.9240 -> 0.9185   -0.55pp
Top3 coverage        0.9770 -> 0.9785   +0.15pp  CI n.s.              <- 核心

新 rank1 正确题来源 (V2):  149/152 = 98.0% 原本已在 rank<=3
                          rank>3 -> rank1 仅 3 题

mixed->all-correct  (n=43): gold_q +0.3375, entropy -0.2756
mixed->all-wrong    (n=19): gold_q -0.2513, gold margin -0.679 -> -3.703, entropy -0.1912
   -> 两边 entropy 都降，概率质量朝相反方向集中

V2 悖论:  median gold_q 0.7735 -> 0.9698  (↑)
          p10     gold_q 0.1642 -> 0.0450  (↓)
          gold_q < 0.05   3.30%  -> 10.55%
          implied Pass@8  0.9295 -> 0.8684  贴合实际 0.9350 -> 0.8400
```

### Epoch5 控制：哪些是通用效应

```text
通用（Epoch5 同向，但幅度约为 GRPO 的 1/2.5）:
  entropy↓ / effective support↓ / top1 margin↑ / rank2/3→rank1 占 98.4%
  V2 - Epoch5 仍显著:  entropy -0.0805, support -0.2760, margin +1.4707 (均 EXCL0)

GRPO-specific（Epoch5 不出现或反向）:
  p10 gold_q       Epoch4->Epoch5 +0.0174 (改善)   vs  ->V2 -0.1192
  gold_q<0.05 占比 Epoch4->Epoch5 +1.05pp          vs  ->V2 +7.25pp
  Top2 coverage    Epoch4->Epoch5 +0.60pp          vs  ->V2 -0.55pp
```

**sharpening 本身是通用训练效应；下尾崩塌与 Top2 收窄才是 GRPO 特有，
也正是 all-wrong↑ / Pass@8↓ 的直接来源。**

### Scoring gate

```text
8-way argmax vs 已有 greedy prediction 一致率:
  Epoch4 0.9540 | Epoch5 0.9685 | V1 0.9775 | V2 0.9830 | V3 0.9790
  全部 >= 0.95 -> H4_SCORING_VALID
```

### Caveats

```text
1. q 是在 8 个 canonical legal answer 上重新归一化的 answer-space 分布，
   不是整个语言生成空间中的绝对 probability。
2. tokenization 偏差: knight=1 token, knave=2 tokens -> KKK 18 / NNN 21 tokens。
   该偏移对所有模型是常数，跨模型比较中抵消；
   spearman(gold token 长度, gold_q 变化) rho = -0.0574，可忽略。
3. mean-token-logprob 敏感性检查退化了（压缩约 20 倍，q 近乎均匀，
   normalized entropy 0.961~0.974），无法用于尾部指标；
   主结果使用真实 sequence log-probability。
4. implied Pass@8 = 1-(1-q_gold)^8 不是真实 Pass@8，
   但与实测值高度吻合，且与 rollout 正确比例 spearman rho = 0.88~0.93。
```

## H5 诊断结论（零训练，2026-08-29）

分析对象：**GRPO-V2**（标准 exact reward、beta=0.01、无 V3 代理目标混杂），
Epoch4 checkpoint-1252 → checkpoint-600。
数据：GRPO-V3 Fresh Holdout N=2000（复用 H4 的 8-way 概率）+ 200-prompt 诊断子集。
脚本：`scripts/audit_h5_sampling_feedback.py`；
报告：`outputs/grpo_h5_sampling_feedback/h5_sampling_feedback_report.md`。

### 核心结论

```text
H5 = SUPPORTED   (7 条判据中 6 条成立，判据 4 数据不足无法检验)

Finite-K on-policy sampling is consistent with a
path-dependent feedback mechanism that preferentially
reinforces already-supported answers and leaves low-support
gold answers increasingly unlikely to receive positive
reinforcement.
```

**不是因果结论。**

```text
H5 is supported as a predictive/path-dependent mechanism,
not yet established causally.
```

允许写：

```text
The observations are consistent with finite-K
on-policy sampling creating a path-dependent
rich-get-richer / poor-get-poorer feedback mechanism.
```

**禁止写：**

```text
K=8 causes polarization.
```

真正的因果实验需要 K 干预（K=8 vs 更大 K），本轮禁止实施。

### 核心证据（保留）

```text
Initial gold_q vs K=8 correct_count:
Spearman rho = +0.9323

Low-support region:
initial gold_q < 0.20

V2 shows:
- negative median gold_q change
- strong lower-tail collapse

while Epoch5 SFT does not show the same pattern.
```

关键 V2 vs Epoch5（median gold_q change）：

```text
initial q [0.05,0.10):
median Δ V2 = -0.0499
median Δ E5 = -0.0109

initial q [0.10,0.20):
median Δ V2 = -0.0896
median Δ E5 = +0.0266

initial q [0.40,0.60):
median Δ V2 = +0.2560
median Δ E5 = +0.0905

initial q [0.60,0.80):
median Δ V2 = +0.2143
median Δ E5 = +0.0939
```

低尾与分位数：

```text
gold_q < 0.05:

Epoch4 = 3.30%
Epoch5 = 4.35%
GRPO-V2 = 10.55%
```

```text
p10 gold_q:

Epoch4 = 0.1642
Epoch5 = 0.1816
GRPO-V2 = 0.0450
```

### 证据缺口（必须保留）

```text
Criterion 4 NOT TESTED:

MISS and task difficulty are strongly confounded.

Within similar initial-gold_q bins,
historical HIT/MISS sample sizes were too small
to determine whether MISS independently predicts
future deterioration.
```

```text
B3 temporal rollout alignment unavailable:

V2 probe IDs and the 200-prompt diagnostic IDs
have no overlap.

No new rollout or training was performed to fill this gap.
```

### 概率定义（必须遵守）

```text
gold_q 是 8 个 canonical legal answer 上重新归一化的概率，
不是完整生成空间中的真实 P(generate exact correct completion)。

真实 hit/miss 一律取自已保存 rollout 的 correct_count；
1-(1-q)^8 只叫 "8-way implied hit probability"，仅用于解释与预测。
实测：低 q 区真实 miss 比 implied 更严重（[0,0.05) 桶 implied 87.4% vs 实际 100%）。
```

### 关键数字

```text
Q1  spearman(initial gold_q, correct_count) = +0.9323
Q2  q<0.10: 10/15 miss；q>0.20: 1/177 miss
Q3  MISS(n=13) vs HIT(n=187):
      delta gold_q  -0.0465 vs +0.0845   (差 +0.1304, CI [+0.0836,+0.1828] EXCL0)
      P(final correct)  0.000 vs 0.829
      final all-wrong   1.000 vs 0.102
      final mean correct_count  0.0 vs 6.45
    但 MISS 的初始 mean gold_q = 0.0657 vs HIT 0.6977 -> 与难度几乎完全共线
Q4  控制初始 gold_q 后无法检验（每桶 MISS<=6、HIT<=3）
Q5  step100 future all-correct 0.8767 vs future all-wrong 0.1826（step0: 0.8305/0.2127）
Q6  low-quartile median 0.2112 -> 0.0952；低尾(q<0.05) 3.0% -> 11.5%
Q7  整体低尾 E4 3.30% / E5 4.35% / V2 10.55%（V2 是 E5 的 6.9 倍）
```

### 最重要的一张表：bin 级 V2 vs Epoch5（10000 bootstrap）

```text
bin            N    medΔV2    medΔE5    medΔ差(V2-E5)              tail<0.05 V2/E5    tail差
[0.00,0.05)    66   -0.0028   -0.0001   -0.0029 [-0.0082,-0.0002]   0.773 / 0.727    +0.046 n.s.
[0.05,0.10)    60   -0.0499   -0.0109   -0.0397 [-0.0543,-0.0288]   0.667 / 0.383    +0.282
[0.10,0.20)   118   -0.0896   +0.0266   -0.1187 [-0.1584,-0.0686]   0.449 / 0.110    +0.339
[0.20,0.40)   201   -0.0177   +0.0480   -0.0545 n.s.                0.199 / 0.010    +0.189
[0.40,0.60)   270   +0.2560   +0.0905   +0.1650 [+0.0984,+0.2172]   0.074 / 0.004    +0.071
[0.60,0.80)   332   +0.2143   +0.0939   +0.1181 [+0.0996,+0.1347]   0.012 / 0.000    +0.012
[0.80,1.00]   953   +0.0106   +0.0032   +0.0071 [+0.0036,+0.0107]   0.003 / 0.000    +0.003
```

低支持桶 V2 中位数显著为负、Epoch5 为正 → **GRPO 特有的恶化**；
中高支持桶 V2 中位数增益是 Epoch5 的 2–3 倍 → **GRPO 特有的强化**。

### 均值陷阱（必须记录）

```text
所有 bin 的 mean delta_gold_q 都是正的（含最低桶 +0.0646），
因为低桶里少数题发生极大幅度跃升（0.02 -> 0.9）把均值拉正，
而多数题其实在下降（最低桶 62.1% 下降）。

本轮结论以 median / 下降比例 / 低尾占比 为准，均值仅作对照。
```

### Empirical low-support risk region

```text
初始 gold_q < 0.20：
  - 实际 miss rate 明显上升（[0,0.10) 达 66.7%）
  - 44.9%~100% 的题最终落入 q<0.05
  - V2 median delta 转负，Epoch5 为正
初始 gold_q > 0.40：
  - 落入低尾比例 < 8%
  - V2 median delta 强正（+0.21~+0.26）

注意：correct→wrong 组（n=13）起始 median 0.5085 仍崩塌，
      说明初始支持度不是唯一决定因素。
```

### 未完成的 B3（如实报告）

```text
V2 probe rollouts 的 20 个 prompt 取自 v2_answer_only_val.jsonl，
200-prompt 诊断子集取自 grpo_v1_final_holdout，两者 id 无交集，
因此 "gold_q(t) -> correct_count(t) -> gold_q(t+1)" 的逐步对齐未能完成。
本轮不为此重新推理或重新训练。
```

## H5 因果干预结论（K=8 → K=16，2026-08-29）

设计预注册（训练前）：`outputs/grpo_k16_intervention/preregistered_design.md`
报告：`outputs/grpo_k16_analysis/k16_causal_report.md`

### 判定

```text
H5_CAUSAL_NOT_SUPPORTED
```

```text
K16 明显提高 sampling coverage（训练期 all-wrong 0.1050 -> 0.0788，相对 -25%；
零方差组 0.6200 -> 0.5563），
但 lower tail / p10 / common-eval all-wrong / Pass@8 均没有改善。
低支持区 (initial gold_q < 0.20, n=217) 反而显著更差。

之前的 H5 相关性更多反映 difficulty，而不是 K 的因果作用。
```

### 关键数字

```text
Fresh K-intervention Holdout (N=2000, seed 20260903):

                     Top1     p10 gold_q   frac q<0.05   norm entropy
  Epoch4            0.7525     0.1808        0.0330        0.2622
  Epoch5            0.7795     0.1824        0.0430        0.2077
  K8  (V2 ckpt-600) 0.7710     0.0534        0.0960        0.1235
  K16 (ckpt-625)    0.7745     0.0530        0.0995        0.1149

Primary bootstrap (10000, seed 20260903), K16 - K8:
  p10 gold_q            -0.00495  CI [-0.02279,+0.01292]  n.s.
  frac gold_q < 0.05    +0.00336  CI [-0.00750,+0.01400]  n.s.
  normalized entropy    -0.00864  CI [-0.01331,-0.00388]  EXCL0 (K16 更尖)
  Top1 accuracy         +0.00356  CI [-0.01000,+0.01750]  n.s.
  McNemar exact p = 0.6643

低支持区 (initial q<0.20, n=217):
  median Δgold_q   K8 -0.02208  K16 -0.03634  diff -0.01355 CI [-0.02760,-0.00000] EXCL0
  frac q<0.05      K8  0.379    K16  0.547    diff +0.08336 CI [+0.02304,+0.14747] EXCL0

Common K_eval=8 (200-prompt):
  Pass@8      E4 0.9350 | K8 0.8400 | K16 0.8450
  all-wrong   E4 0.0650 | K8 0.1600 | K16 0.1550
  mixed->all-wrong   K8 19  ->  K16 19     <- 完全没有变化
```

### 前置 gate 全部通过

```text
HISTORICAL_K8_CONTROL_VALID
  V2 时期脚本与当前脚本各跑 2 步，step1/2 的 loss/grad_norm/kl/entropy/
  reward_mean/num_tokens 逐位一致 -> 训练语义未变。
  唯一触及训练信号的改动是 reward 重构为 compute_reward，
  exact 模式行为由单元测试锁定为等价。

  必须记录的噪声下限：V2 原始记录与重跑相差 1 个 token (8276 vs 8277)，
  导致 loss/grad/kl 在 1e-4 量级不同。
  GRPO rollout 采样在本加速器上不是 bit-reproducible。

SAMPLER_ALIGNED
  K8 / K16 前 100 optimizer step (800 unique prompts) 顺序完全一致，
  每步 8 unique prompts，64 vs 128 completions。

K16 smoke @128 completions/step
  峰值 45.7GB，无 OOM。未改动 prompt_batch_size /
  gradient accumulation / gradient checkpointing / max_completion_length。
```

### 机制线索（不是结论）

```text
K16 的 sharpening 更强：entropy diff -0.00864 EXCL0，
KL mean 0.01544 vs K8 0.00908，但 Top1 几乎不变。

更像：更多 rollout -> 组统计量噪声更小 -> 更新更自信 -> sharpening 更强，
而不是：更多 rollout -> 低支持答案被救回来。

结合 H4：GRPO 的可观测作用是 answer-space 概率重分配 / sharpening；
本轮说明 sampling coverage 不是该 sharpening 的瓶颈。
```

### 已按规格不执行

```text
未跑 K=32（无论结果如何本轮禁止）
未改 reward / beta / LR / init / 训练数据 / epoch / generator
```

## 下一个问题

```text
H5 已在两个层面都被检验：
  - 相关性：       SUPPORTED   (predictive / path-dependent)
  - 因果 (K=8->16)：NOT_SUPPORTED

下一步未定，需人工确认。不建议继续沿 K 方向扩大（K=32），
因为 K16 已证明 coverage 提升不改变 lower-tail collapse。
```

## 下一个问题（K 干预之前的版本，保留作历史）

```text
Next:
Design causal K intervention.
Do not run yet.
```

下一实验目标：

```text
K=8 vs larger K
```

但必须先解决两个设计问题：

```text
1. difficulty stratification / initial gold_q stratification
2. rollout compute confounding
```

后续至少讨论两种对齐方式：

```text
A. equal optimizer steps
      -> K 大的组看到 2x / 4x 的 rollout
B. equal total rollout budget
      -> K 大的组 optimizer steps 减半 / 四分之一
```

**不能把：**

```text
larger K + more rollout compute
```

**的结果直接归因于 K。**

固定项（K 干预实验必须保持不变）：

```text
same SFT init (checkpoint-1252) / same exact reward /
same beta = 0.01 / same LR / same data /
same seed strategy / same number of unique prompts
```

**不要自动开始。** 需人工确认后再执行。

---

## 下一个问题（H5 之前的版本，保留作历史）

```text
H5:
Finite-K on-policy sampling may create a rich-get-richer
feedback loop that amplifies initial probability differences.
```

类型：**零训练诊断**（静态 retrospective audit + 训练轨迹 probability audit）。

```text
K=8, P(miss) = (1-p)^8:
  p=0.50 -> 0.4%    p=0.30 -> 5.8%    p=0.20 -> 16.8%
  p=0.10 -> 43.0%   p=0.05 -> 66.3%

gold probability 高 -> 容易采到 correct -> 持续正强化 -> 更高
gold probability 低 -> 经常 8 次全 miss -> 缺正强化 -> 更低
```

主分析对象：**GRPO-V2**（标准 exact reward、beta=0.01、无 V3 的代理目标混杂）。
初始化 Epoch4 checkpoint-1252，最终 checkpoint-600。
概率定义必须用 H4 的 `gold_q`（8-way 归一化），
真实 hit/miss 用已有 rollout 的 `correct_count`。

**H5 即使 SUPPORTED 也只是 path-dependence 证据，不是因果证据。**
真正的因果实验是 K=8 vs K=16/32，本轮禁止实施。

H1 / H2 / H3 均不支持，H4 支持。当前对 polarization 的最佳解释是：

```text
GRPO 主要在 answer-space 上重分配概率：
把多数题推到"几乎确定正确"（greedy↑、all-correct↑），
同时把少数困难题推到"几乎确定错误"（all-wrong↑、Pass@8↓）。

sharpening 的"量"是通用训练效应（Epoch5 也有，约 1/2.5），
但"下尾崩塌 + Top2 收窄"是 GRPO 特有的。
```

---

## 下一步实验

```text
未定。H1 / H2 / H3 全部不支持，候选假设已耗尽。
H4 待验证（零训练诊断，不启动训练）。
```

**不要自动开始 GRPO-V4。** 下一轮需要先由人工确认研究方向。

不建议继续的方向（本轮已排除）：

```text
继续调 beta / LR / reward shaping    -> H1、H2 已证明不改变 polarization
继续做 partial / Hamming reward      -> H2 已证明出现 Goodhart，且显著变差
按 structure shortcut 重构 generator -> H3 已证明 GRPO 未放大该 shortcut
```

建议的新方向（需人工确认后另立假设）：

```text
把问题从「优化过程缺什么」转向「目标与任务的错位」：
  1. 复核 polarization 是否本身就是「用 exact-match 目标做 RL」的必然结果
     —— H2 中 partial reward 的 Goodhart 证据支持这一方向
  2. 若要继续追 structure，需要更大的诊断集
     （top_ops 需 N >> 2000 才能支撑 per-signature 分析），而不是新特征
  3. 在提出独立假设之前，不建议再对 reward / beta / LR 做扫描
```

### 方法论收获（跨轮通用）

```text
1. 必须设「通用训练效应」对照。本轮用 Epoch5 证明：
   若干看起来像「RL 放大结构依赖」的变化，纯 SFT 多训一个 epoch 同样产生。
   只比较 Epoch4 vs GRPO 会得出错误结论。

2. 原始 MI(结构; 预测) 混杂了「结构→GT→预测」的间接路径，
   模型越准该项越大。判定 shortcut 放大必须用 conditional MI（控制 GT）。

3. 高基数特征（top_ops 211 类、op_signature 582 类）的 plug-in MI
   绝大部分是估计偏差（分别占 59% 与 91%）。必须用 permutation null 校准。

4. 评估 GRPO 效果时，H(prediction) 在各模型间可能不同，
   跨模型比较 MI 前应检查熵是否可比（本轮实测相同，故可直接比较）。
```

新的评测集状态：

```text
data/processed/grpo_v2_final_holdout.jsonl  已被 GRPO-V2/V3/H3 诊断使用（非 untouched）
data/processed/grpo_v3_final_holdout.jsonl  已被 GRPO-V3/H3 诊断使用（非 untouched）
  下一轮若做正式实验，需另生成新的 holdout（建议 seed 20260903）。
```

任意关键恢复项失败时，不要自动重新训练，先报告差异。
