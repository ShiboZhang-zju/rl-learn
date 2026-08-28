# 迁移说明

这个压缩包包含项目代码、数据、测试、LoRA adapter、checkpoint 和评测结果。

## 解压后运行

```bash
cd small-reasoning-sft
python3 -m pip install -e '.[dev]'
pytest -q
```

项目没有打包 Qwen base model 权重。新设备第一次运行时需要联网下载：

```text
Qwen/Qwen2.5-0.5B-Instruct
```

如果新设备无法联网，请先把 Hugging Face 本地模型目录复制过去，然后将命令中的 `--model` 换成本地路径。

## 继续评测已有 adapter

```bash
python3 scripts/evaluate_model.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --adapter outputs/sft_smoke \
  --data-file data/processed/sft_train.jsonl \
  --output outputs/sft_smoke_train8_eval.jsonl \
  --limit 8
```

balanced 实验的 adapter 在：

```text
outputs/sft_overfit_balanced/
```

## 重新训练

训练使用目标设备自动选择 CUDA、MPS 或 CPU。Apple Silicon 当前配置固定使用 float32；NVIDIA 设备可以在 `configs/sft.yaml` 中单独调整混合精度。

