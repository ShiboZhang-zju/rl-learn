# Small Reasoning SFT

这是一个用于理解 `数据 → tokenization → LoRA SFT → evaluation → rollout/reward → GRPO` 的小型实验项目。

当前冻结的 baseline：

- 任务：3-person Knights & Knaves
- 模型：`Qwen/Qwen2.5-0.5B-Instruct`
- 本地第一阶段：Mac 上运行 LoRA SFT
- 后续 RL：基于同一套 verifier 的 rule-based reward，再接 GRPO
- 参考实现：Sophie0、Logic-RL；本仓库不直接复制它们的训练代码

## Resume on a new GPU

For the latest experiment state and recovery instructions:

→ `EXPERIMENT_STATE.md`

Quick restore:

```bash
bash scripts/bootstrap_gpu.sh
```

## 目录

```text
.
├── configs/
│   ├── sft.yaml
│   └── grpo.yaml
├── data/
│   ├── raw/          # 由 generate_dataset.py 产生的可验证原始题目
│   └── processed/    # TRL 可直接读取的 prompt/completion 数据
├── docs/
│   └── plan.md
├── scripts/
│   ├── generate_dataset.py
│   ├── audit_data.py
│   ├── debug_tokenizer.py
│   ├── evaluate_model.py
│   ├── generate_balanced_overfit.py
│   ├── debug_grpo_math.py
│   └── train_sft.py
├── src/kk_sft/
│   ├── logic.py       # 表达式、求解器、题目生成
│   ├── data.py        # JSONL、SFT 样本和 reasoning trace
│   ├── audit.py       # 数据质量检查
│   ├── evaluation.py  # 统一 parse / metric / reward
│   └── grpo_math.py   # 独立的 GRPO 数学调试
└── tests/
```

## 1. 安装

如果当前环境已经有依赖，可以直接从第 2 步开始。建议在项目目录执行：

```bash
python3 -m pip install -e '.[dev]'
```

Apple Silicon 会由 PyTorch 使用 MPS；训练脚本不会引入 `bitsandbytes`、`vLLM`、`DeepSpeed` 或 `verl`。

## 2. 生成固定数据集

第一轮先生成小规模数据，确认流程后再使用默认规模：

```bash
python3 scripts/generate_dataset.py \
  --output-dir data \
  --train-size 1000 \
  --val-size 200 \
  --test-size 500 \
  --seed 42
```

生成结果：

```text
data/raw/{train,val,test}.jsonl
data/processed/{sft_train,sft_val,sft_test}.jsonl
data/processed/manifest.json
```

每道题由程序生成，并由 exact solver 穷举 `2^3=8` 种身份组合；只保留唯一解题目。SFT reasoning trace 也是程序生成的，不依赖外部 teacher model。

## 3. 审计数据

```bash
python3 scripts/audit_data.py \
  --raw-dir data/raw \
  --processed-dir data/processed \
  --output data/processed/audit_report.json
```

审计会检查唯一解、重复题、split 泄漏、答案分布、文本长度，并输出报告。训练前应先看报告和随机样本。

## 4. 调试 tokenizer

第一次运行会下载 tokenizer；若模型已在本地，可把 `--model` 换成本地目录。

```bash
python3 scripts/debug_tokenizer.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --data-file data/processed/sft_train.jsonl \
  --index 0 \
  --max-tokens 220
```

脚本会打印：原始 prompt/completion、chat template 文本、token/id，以及 completion-only loss 中哪些 label 为 `-100`。

## 5. Base evaluation

```bash
python3 scripts/evaluate_model.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --data-file data/processed/sft_test.jsonl \
  --output outputs/base_eval.jsonl
```

所有 checkpoint 都使用同一个 evaluator。默认使用贪心解码，避免 baseline 之间的 decoding 差异。

## 6. Mac 上先做 SFT smoke test

先用 32 条样本过拟合，检查数据、mask、LoRA 和 checkpoint：

```bash
python3 scripts/train_sft.py \
  --config configs/sft.yaml \
  --train-file data/processed/sft_train.jsonl \
  --eval-file data/processed/sft_val.jsonl \
  --train-limit 32 \
  --max-steps 100 \
  --output-dir outputs/sft_overfit
```

确认 loss 明显下降并能记住训练样本后，再做正式 SFT：

```bash
python3 scripts/train_sft.py \
  --config configs/sft.yaml \
  --train-file data/processed/sft_train.jsonl \
  --eval-file data/processed/sft_val.jsonl \
  --output-dir outputs/sft
```

训练完成后评估 LoRA adapter：

```bash
python3 scripts/evaluate_model.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --adapter outputs/sft \
  --data-file data/processed/sft_test.jsonl \
  --output outputs/sft_eval.jsonl
```

## 7. Balanced overfit sanity check

随机 8 条样本可能被多数答案模式误导。因此先生成覆盖全部 `2^3=8` 种答案模式的数据：

```bash
python3 scripts/generate_balanced_overfit.py
```

然后保持 SFT 参数不变，只训练 20 个 optimizer steps：

```bash
python3 scripts/train_sft.py \
  --config configs/sft.yaml \
  --train-file data/processed/sft_overfit_balanced.jsonl \
  --eval-file data/processed/sft_overfit_balanced.jsonl \
  --max-steps 20 \
  --output-dir outputs/sft_overfit_balanced
```

评估时除了 Exact Accuracy，还要检查输出模式分布：

```text
ground_truth_pattern_distribution
prediction_pattern_distribution
```

当前 20-step 实验结果是：格式 `8/8`，答案 `1/8`，预测集中为 `KKK × 7 + KKN × 1`。因此 G5 true overfit 尚未通过，不能据此声称模型已经学会逻辑映射。

## 8. RL 尚未启动前的数学调试

```bash
python3 scripts/debug_grpo_math.py
```

它只用固定 rewards `[0, 1, 0, 1]` 演示 group mean、std、advantage、ratio、clip 和 KL；这一步通过后，再实现真正的 rollout/GRPO trainer。

## 训练原则

1. 32 条不能过拟合时，不进入正式 SFT。
2. Base、SFT、后续 GRPO 必须使用同一个 test set 和 evaluator。
3. 先保存原始题目和 solver 结果，再讨论 learning rate。
4. 0.5B 的本地任务是验证 pipeline，不预设它一定能得到漂亮的逻辑推理分数。
