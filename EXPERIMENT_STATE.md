# EXPERIMENT STATE

本文件是所有 Agent 的**第一入口**。在开始任何工作之前请完整阅读。

最近的完整实验记录不要复制到别处，只需要维护这一个文件。

---

## Last known good commit

```text
30976e5592ab207b7d84b0413d123a8aaee6666a   # 完成 GRPO-V1 正式实验：训练、评估与统计分析
```

上面的 commit 是**本文档所描述的实验状态被产生的那个 commit**（GRPO-V1 完成）。
它之后的 commit 只涉及文档/脚本，不改变实验状态。

恢复时先比较（避免「文档描述状态 A，但 checkout 的代码是状态 B」）：

```bash
git rev-parse HEAD
git merge-base --is-ancestor 30976e5592ab207b7d84b0413d123a8aaee6666a HEAD && echo IN_SYNC
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
GRPO-V2        NOT STARTED
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

## 当前待验证假设

按顺序：

```text
H1:
beta=0 缺少 reference KL，
导致 policy 过度 sharpening / drift。

H2:
binary exact reward 太稀疏，
all-wrong group advantage=0，无法自我纠正。

H3:
generator 中残余 structure→label shortcut
可能被 RL 放大。
```

---

## 下一步实验

```text
GRPO-V2
```

**第一轮只验证 H1。**

保持（与 GRPO-V1 完全一致）：

```text
init checkpoint = checkpoint-1252
lr = 1e-5
num_generations = 8
temperature = 0.8
top_p = 0.95
1 epoch
```

唯一计划修改：

```text
GRPO-V1 beta = 0
GRPO-V2 beta = 0.01
```

正式执行前需要先确认 TRL 0.23 的：

```text
beta semantics
reference model creation
KL implementation/logging
```

**禁止自动同时修改：**

```text
learning rate
epoch
reward
generator
curriculum
num_generations
```

任意关键恢复项失败时，不要自动重新训练，先报告差异。
