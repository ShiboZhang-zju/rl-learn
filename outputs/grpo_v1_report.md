# GRPO-V1 正式实验报告

日期：2026-08-28
状态：**GRPO_V1_COMPLETE**（含限定说明，见 §7）

## 1. 实验目标

完成一轮完整、可解释的 GRPO 训练，并验证 RL 是否能在未见 puzzle 上超过 SFT baseline。围绕六个问题：
1. GRPO 是否超过自己的初始化（SFT Epoch4）？
2. GRPO 是否超过更强的 SFT baseline（SFT Epoch5）？
3. Reward dynamics 如何变化？
4. GRPO 是否把 mixed groups 转成 all-correct？
5. 是否发生过拟合？
6. RL 改变了什么？

## 2. 环境与实现（重要前置说明）

- 仓库中**不存在** `scripts/train_grpo.py`（含 git 历史），即指令中假设的“smoke 已验证的自定义实现”并不存在。
- 按用户决定改用 **TRL GRPOTrainer**（`trl==0.23.0`，非指令假设的 0.29.0；0.23 无 FSDPModule 兼容问题，且不使用 GRPOTrainer 的 vLLM/DeepSpeed/FSDP 路径）。
- 因任务需要修复环境：peft 装回 `0.20.0`（能读 ckpt-1252 的 adapter 配置），卸载了环境里未使用的 `torchao==0.9.0`（其版本与 peft 0.20 的 torchao 检测冲突，与训练无关）。
- LoRA adapter 通过 `PeftModel.from_pretrained(..., is_trainable=True)` 加载（adapter 配置 `inference_mode: true`，否则全部参数 `requires_grad=False`）。
- 训练不使用 vLLM / DeepSpeed / FSDP / reference model / KL reward / format / length / entropy reward / reward model。

## 3. 数据

- 新生成 GRPO Train **5000**：`data/raw_grpo_v1/train.jsonl`、`data/processed/grpo_v1_train.jsonl`（仅 prompt + ground truth）。
- 新生成 Final Holdout **2000**：`data/raw_grpo_v1/final_holdout.jsonl`、`data/processed/grpo_v1_final_holdout.jsonl`。
- 使用 `generate_puzzle_v2`（statement-first），seed `20260830`（train）与 `21260833`（holdout），与 v2 的 seed `20260829` 区分。
- 去重校验（canonical key）：
  - GRPO Train ∩ (SFT Train/Val/V2 Test + legacy) = **0**
  - Final Holdout ∩ (全部 v2/legacy/GRPO Train) = **0**
  - 详细校验见 §3 数据生成时的输出与 `data/processed/grpo_v1_manifest.json`。

## 4. 训练前审计（第七节）

按指令对 GRPO 实现逐项核对（基于 TRL 0.23 源码 + 真实样本 dump，`outputs/grpo_v1/audit/debug_sample.json`）：

| 项目 | 结论 |
|---|---|
| Log probability | `old_per_token_logps` 在 `torch.no_grad()` 下计算并 detach；`new` 保留梯度；`ratio = exp(new − old)` ✓ |
| Loss mask | 仅 assistant completion tokens（含 EOS、不含 EOS 后 token），排除 system/user prompt 与 padding；真实样本：prompt_len=127，completion 21 tokens，loss 位置 127..147，均为 completion token ✓ |
| Advantage | 每组 8 个 rollout 独立 group 归一化，**population std**（TRL `scale_rewards="group"`，与 `grpo_math.py` 一致）；group std==0 时 advantage 恒为 0，无 NaN ✓ |
| Policy objective | 标准 clipped surrogate（`epsilon=0.2`）；单次更新 ratio≈1、clip≈0 属正常 ✓ |
| 其他 | `beta=0.0` → 无 reference model / 无 KL penalty ✓ |

## 5. 训练配置与规模

- 初始化：`outputs/sft_v2_5k_p800/checkpoint-1252`（SFT Epoch4，Val 72.2%）。
- 生成：`num_generations=8`，`temperature=0.8`，`top_p=0.95`，`max_completion_length=64`。
- 训练：`bf16`，`gradient_checkpointing=false`，`beta=0.0`，`lr=1e-5`，`weight_decay=0`，`max_grad_norm=1.0`，1 epoch。
- 每 optimizer step = **8 个 unique prompts × 8 rollouts = 64**（TRL 0.23 语义下 `per_device_train_batch_size=64 = prompt_batch_size(8)×num_generations(8)`）。
- 总规模：**625 optimizer steps，40,000 rollouts**。
- 训练期间评测：V2 Val 500 prompts（greedy），每 100 步一次 + Step 0 + Final。

## 6. 结果

### 6.1 训练指标（每 100 步分桶均值）

| Step | Reward mean | Mixed | All-wrong | All-correct | Zero-var | Avg unique | Avg correct/group | Format | Parse |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.730 | 0.381 | 0.098 | 0.520 | 0.619 | 1.52 | 5.84 | 1.000 | 1.000 |
| 100 | 0.729 | 0.279 | 0.144 | 0.578 | 0.720 | 1.37 | 5.83 | 0.999 | 0.999 |
| 200 | 0.754 | 0.235 | 0.138 | 0.627 | 0.765 | 1.31 | 6.03 | 0.999 | 0.999 |
| 300 | 0.748 | 0.225 | 0.136 | 0.639 | 0.775 | 1.28 | 5.98 | 0.999 | 0.999 |
| 400 | 0.755 | 0.223 | 0.138 | 0.640 | 0.776 | 1.28 | 6.04 | 1.000 | 1.000 |
| 500 | 0.755 | 0.216 | 0.149 | 0.635 | 0.786 | 1.28 | 6.04 | 0.994 | 0.994 |
| 600 | 0.749 | 0.216 | 0.163 | 0.620 | 0.774 | 1.26 | 5.99 | 0.998 | 0.998 |

注：TRL 的 loss/grad 字段在 metrics jsonl 中滞后一步；reward/group 统计为当前步。loss 量级 ≈ 0（ratio≈1、多数组零方差），grad_norm 1–3，peak memory ≈ 25–30 GB。

### 6.2 V2 Val greedy 验证轨迹

| Step | Val exact | Format | Parse | Prediction 分布（K/N 计数） |
|---:|---:|---:|---:|---|
| 0 | 0.7220 | 1.000 | 1.000 | KKK60 KKN59 KNK47 KNN56 NKK52 NKN74 NNK83 NNN69 |
| 100 | 0.7580 | 1.000 | 1.000 | KKK66 KKN63 KNK51 KNN55 NKK64 NKN69 NNK73 NNN59 |
| 200 | 0.7660 | 1.000 | 1.000 | KKK66 KKN62 KNK39 KNN46 NKK72 NKN73 NNK79 NNN63 |
| 300 | 0.7520 | 0.998 | 0.998 | KKK62 KKN59 KNK66 KNN63 NKK41 NKN50 NNK85 NNN72 (2 invalid) |
| 400 | 0.7660 | 1.000 | 1.000 | KKK63 KKN59 KNK56 KNN61 NKK66 NKN62 NNK73 NNN60 |
| 500 | 0.7420 | 1.000 | 1.000 | KKK66 KKN59 KNK51 KNN48 NKK64 NKN67 NNK80 NNN63 |
| 600 | 0.7620 | 1.000 | 1.000 | KKK69 KKN81 KNK54 KNN52 NKK61 NKN53 NNK69 NNN60 |
| 625 | 0.7660 | 1.000 | 1.000 | KKK68 KKN74 KNK54 KNN51 NKK63 NKN55 NNK72 NNN62 |

**Best checkpoint 选择**：Val exact 最高 0.7660，出现于 Step 200 / 400 / 625；按“优先选较早 checkpoint”规则 → **`outputs/grpo_v1/checkpoint-200`**。

### 6.3 固定 20-prompt Probe（V2 Val，每 prompt 8 rollouts）

| Probe step | All-correct | Mixed | All-wrong |
|---:|---:|---:|---:|
| 0 | 11/20 | 9/20 | 0/20 |
| 100 | 10/20 | 6/20 | 4/20 |
| 300 | 12/20 | 3/20 | 5/20 |
| 500 | 13/20 | 4/20 | 3/20 |
| 625 | 13/20 | 4/20 | 3/20 |

示例（correct rollouts / 8）：`kk_v2_val_000028` KKK：8→8→8→8→8；`kk_v2_val_000116` KNN：8→7→8→8→8；`kk_v2_val_000237` NNK：6→7→7→8→3（中途转 all-correct，最终回落）。

### 6.4 最终统一评估（greedy，统一 evaluator）

| Model | V2 Val | Existing V2 Test | Final Holdout |
|---|---:|---:|---:|
| SFT Epoch4 (ckpt-1252) | 72.2% | 75.7%* | 74.9% |
| SFT Epoch5 (ckpt-1565) | 74.8% | 77.9% | **77.5%** |
| GRPO Best (ckpt-200) | **76.6%** | 75.0% | 76.0% |

\* 早期记录 ckpt-1252 V2 Test greedy 为 75.1%（`outputs/v2_5k_ckpt1252_greedy_eval.jsonl`）；统一 evaluator 实测为 75.7%，按指令以统一 evaluator 为准。

格式 / parse / 8 类预测分布 / 混淆矩阵：见 `outputs/grpo_v1_final_metrics.json`（含各模型 × 各数据集）。

## 7. 六个问题的回答

### Q1. GRPO 是否超过自己的初始化（SFT Epoch4）？
部分超过：V2 Val **+4.4pp**（72.2→76.6），Final Holdout **+1.1pp**（74.9→76.0）；但 Existing V2 Test **−0.7pp**（75.7→75.0）。

### Q2. GRPO 是否超过更强的 SFT baseline（SFT Epoch5）？
**否**。V2 Val +1.8pp（76.6 vs 74.8），但 V2 Test **−2.9pp**（75.0 vs 77.9）、Final Holdout **−1.5pp**（76.0 vs 77.5）。SFT Epoch5 在 untouched Final Holdout 上仍是最强。

### Q3. Reward dynamics？
Reward mean 0.730→0.755（train 分布上 SFT 已较强，RL 增益温和）；Mixed 38.1%→21.6%，All-correct 52.0%→63.5%，Zero-var 61.9%→78.6%，Avg unique 1.52→1.26。呈典型的 **policy sharpening**。

### Q4. 是否把 mixed groups 转成 all-correct？
是（部分）：Probe 中 Mixed 9/20→4/20、All-correct 11/20→13/20；但伴随少量组转为 All-wrong（0→3/20），存在轻微不稳定。

### Q5. 是否发生过拟合？
无明显 classic overfit，但有**轻微过拟合/过锐化迹象**：train reward 提升、Val 上升；但 Existing V2 Test 相对初始化下降 0.7pp。Final Holdout 相对初始化上升，故不能断言 holdout 过拟合；V2 Test 的回落提示 1 epoch 单轮 GRPO 在高置信 prompt 上可牺牲一定泛化。

### Q6. RL 改变了什么？
- 行为变化：predict 更集中（unique answers 下降），all-correct 组增多；格式保持 ≥99.5%（仅少量 INVALID）。
- 具体样本对比见 `outputs/grpo_v1_probe_rollouts.json`（SFT 初始化 vs step100/300/500/final 完整 rollouts）。

## 8. 输出文件

```text
configs/grpo_v1_full.yaml
data/raw_grpo_v1/train.jsonl
data/raw_grpo_v1/final_holdout.jsonl
data/processed/grpo_v1_train.jsonl
data/processed/grpo_v1_final_holdout.jsonl
data/processed/grpo_v1_manifest.json
outputs/grpo_v1/checkpoint-{0,100,200,300,400,500,600,625}/
outputs/grpo_v1/grpo_v1_train_metrics.jsonl
outputs/grpo_v1/grpo_v1_val_metrics.json
outputs/grpo_v1/grpo_v1_probe_rollouts.json
outputs/grpo_v1/audit/debug_sample.json
outputs/grpo_v1_final_metrics.json
outputs/grpo_v1_final/{sft_epoch4,sft_epoch5,grpo_best}_*.jsonl
outputs/grpo_v1_report.md
```

未覆盖任何之前的 SFT 与 smoke 结果。

## 9. 结论与限定

- **GRPO_V1_COMPLETE**：GRPO 在 V2 Val 上显著超过其初始化并超过 SFT Epoch5（76.6% vs 74.8%），Final Holdout 超过其初始化；但**未超过 SFT Epoch5 在 untouched Final Holdout 上的 77.5%**。
- 主要限定：
  1. 环境与仓库现状与指令假设不符（无 train_grpo.py、trl 0.23.0 非 0.29.0），已按用户决定使用 TRL GRPOTrainer 并完成同等审计；
  2. TRL 的 loss/grad 日志滞后一步（分析已注明）；
  3. 单 checkpoint、单 seed；Best 由 V2 Val 选择。
- 下一步建议（不在本轮执行）：更长训练（多 epoch / 更大 lr 的 curriculum）、引入 KL 约束（beta>0）抑制 all-wrong 上升、或用 Epoch5 作为 RL 初始化再验证。
