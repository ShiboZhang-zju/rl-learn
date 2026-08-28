# P800 全量 SFT 训练结果报告

日期：2026-08-28

## 1. 实验目标

使用完整训练集对 `Qwen/Qwen2.5-0.5B-Instruct` 进行 LoRA SFT，并在固定测试集上比较 Base 模型与 SFT Adapter 的答案质量。

## 2. 数据集

| split | 数量 |
|---|---:|
| train | 1000 |
| validation | 200 |
| test | 500 |

数据审计结果：训练/验证/测试集之间无题目泄漏；每道题均有唯一解；8 种答案模式均有覆盖。

## 3. 训练配置

配置文件：`configs/sft.yaml`

- 设备：P800，`cuda`
- 训练轮数：1 epoch
- 训练样本：1000 条，全量使用
- `per_device_train_batch_size`：16
- `gradient_accumulation_steps`：1
- 精度：BF16
- 最大长度：512
- LoRA rank：8
- LoRA alpha：16
- LoRA dropout：0.05
- 学习率：`1e-4`
- completion-only loss：开启
- optimizer steps：63

运行环境：

- PyTorch：`2.5.1+cu118`
- Transformers：`4.57.1`
- TRL：`0.29.0`
- PEFT：`0.18.1`
- Accelerate：`1.14.0`
- Datasets：`4.0.0`

## 4. 训练结果

| 指标 | 结果 |
|---|---:|
| train loss（最终汇总） | 0.1701 |
| train mean token accuracy | 0.9734 |
| validation loss（step 50） | 0.0593 |
| validation mean token accuracy | 0.9713 |
| 训练耗时 | 51.65 秒 |
| 训练吞吐 | 19.36 samples/s |
| 可训练参数 | 4,399,104 / 498,431,872 |
| 可训练参数比例 | 0.8826% |

训练过程中 loss 从 step 10 的 `0.5982` 降至 step 60 的 `0.0593`，token-level 指标收敛明显。

正式 Adapter 输出目录：`outputs/sft_full_p800/`

最终 checkpoint：`outputs/sft_full_p800/checkpoint-63/`

## 5. 测试集评估

评估配置文件：`configs/eval_p800.yaml`

- batch size：16
- prompt 最大长度：512
- 最大生成长度：320 tokens
- 推理精度：BF16
- 两个进程并发：一个 Base、一个 SFT
- 结果按 batch 实时追加写入

将最大生成长度设为 320，是因为训练 completion 的 `</answer>` 结束位置约为 278 tokens；此前 96 tokens 会在答案块之前截断，导致评估全部被判为格式错误。

### 汇总指标

| 模型 | Exact accuracy | Format accuracy | Parse success | 平均输出字符数 |
|---|---:|---:|---:|---:|
| Base | 0/500（0.0%） | 8/500（1.6%） | 8/500（1.6%） | 817.1 |
| SFT | 81/500（16.2%） | 500/500（100.0%） | 500/500（100.0%） | 1099.2 |

相对 Base：

- Exact accuracy 提升 `16.2` 个百分点；
- Format accuracy 提升 `98.4` 个百分点；
- Parse success 提升 `98.4` 个百分点。

### 答案模式分布

SFT 的预测模式为：

- `KKN`：301
- `KKK`：189
- `KNN`：10

只覆盖 3 种模式，仍明显偏向少数模式，说明模型虽然已经稳定学会输出格式，但答案映射和逻辑泛化仍然不足，存在明显的模式偏置/坍缩现象。

Base 模型在 500 条测试样本中有 492 条无法被统一解析，只有 8 条满足当前答案解析格式。

## 6. 答案文件

- Base 答案：`outputs/base_test_eval_p800_batched.jsonl`
- SFT 答案：`outputs/sft_test_eval_p800_batched.jsonl`

每条记录包含：题目 ID、完整 prompt、模型预测、解析答案、解析原因、格式是否有效、ground truth、是否正确以及答案模式。

## 7. 结论

本次 P800 全量 SFT 训练和测试流程执行成功。SFT 明显改善了输出格式稳定性，并将测试集 exact accuracy 从 `0%` 提升到 `16.2%`，但仍未达到真正可靠的逻辑推理水平。

当前最主要的后续问题不是训练吞吐，而是：

1. 模型预测集中在 `KKN`、`KKK`、`KNN` 三种模式；
2. token-level loss 很低，但答案级 accuracy 仍然有限；
3. 训练 reasoning trace 可能过长，模型更容易学习固定模板而不是答案映射；
4. Base 模型的格式提示存在被复制的问题；
5. 在进入 GRPO 前，建议先改成更短、更明确的答案监督，并重新做 balanced overfit 和全量评估。
