#!/usr/bin/env python3
"""Final GRPO-v1 evaluation: SFT Epoch4 / SFT Epoch5 / GRPO Best on
V2 Val, Existing V2 Test, and Final Holdout. Greedy decoding, unified evaluator.

Usage:
  python scripts/evaluate_grpo_v1.py --grpo-checkpoint outputs/grpo_v1/checkpoint-XXX \
      --output outputs/grpo_v1_final_metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kk_sft.data import read_jsonl  # noqa: E402
from kk_sft.evaluation import parse_answer  # noqa: E402

LABELS = ["KKK", "KKN", "KNK", "KNN", "NKK", "NKN", "NNK", "NNN"]


def pattern(answer: dict | None, people: list[str]) -> str:
    if not isinstance(answer, dict):
        return "INVALID"
    return "".join("K" if answer.get(person) == "knight" else "N" for person in people)


def evaluate(model, tokenizer, rows: list[dict], output: Path, device: str, batch_size: int, max_new_tokens: int) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with output.open("w", encoding="utf-8") as handle:
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            prompts = [tokenizer.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True) for row in batch]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=True,
                )
            texts = tokenizer.batch_decode(generated[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
            for row, text in zip(batch, texts):
                parsed = parse_answer(text, row["puzzle"]["people"])
                record = {
                    "id": row["id"],
                    "prediction": text,
                    "parsed_answer": parsed.parsed,
                    "parse_reason": parsed.reason,
                    "format_valid": parsed.format_valid,
                    "ground_truth": row["answer"],
                    "correct": parsed.parsed == row["answer"],
                    "ground_truth_pattern": pattern(row["answer"], row["puzzle"]["people"]),
                    "prediction_pattern": pattern(parsed.parsed, row["puzzle"]["people"]),
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                records.append(record)
            handle.flush()
    matrix = {ground: {prediction: 0 for prediction in LABELS} for ground in LABELS}
    for record in records:
        if record["prediction_pattern"] in LABELS:
            matrix[record["ground_truth_pattern"]][record["prediction_pattern"]] += 1
    return {
        "count": len(records),
        "exact_accuracy": sum(record["correct"] for record in records) / len(records),
        "format_accuracy": sum(record["format_valid"] for record in records) / len(records),
        "parse_success_rate": sum(record["parsed_answer"] is not None for record in records) / len(records),
        "ground_truth_distribution": dict(sorted(Counter(record["ground_truth_pattern"] for record in records).items())),
        "prediction_distribution": dict(sorted(Counter(record["prediction_pattern"] for record in records).items())),
        "confusion_matrix": matrix,
        "prediction_file": str(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--sft-epoch4", type=Path, default=Path("outputs/sft_v2_5k_p800/checkpoint-1252"))
    parser.add_argument("--sft-epoch5", type=Path, default=Path("outputs/sft_v2_5k_p800/checkpoint-1565"))
    parser.add_argument("--grpo-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/grpo_v1_final"))
    parser.add_argument("--output", type=Path, default=Path("outputs/grpo_v1_final_metrics.json"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    datasets = {
        "v2_val": Path("data/processed/v2_answer_only_val.jsonl"),
        "v2_test": Path("data/processed/v2_answer_only_test.jsonl"),
        "final_holdout": Path("data/processed/grpo_v1_final_holdout.jsonl"),
    }
    models = {
        "sft_epoch4": args.sft_epoch4,
        "sft_epoch5": args.sft_epoch5,
        "grpo_best": args.grpo_checkpoint,
    }
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    summary = {"models": {}}
    for model_name, adapter in models.items():
        model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.bfloat16, trust_remote_code=True)
        model = PeftModel.from_pretrained(model, adapter)
        model.to(args.device)
        model.eval()
        summary["models"][model_name] = {"adapter": str(adapter), "datasets": {}}
        for dataset_name, path in datasets.items():
            rows = read_jsonl(path)
            output = args.output_dir / f"{model_name}_{dataset_name}.jsonl"
            metrics = evaluate(model, tokenizer, rows, output, args.device, args.batch_size, args.max_new_tokens)
            summary["models"][model_name]["datasets"][dataset_name] = metrics
            print(json.dumps({"model": model_name, "dataset": dataset_name, "exact_accuracy": metrics["exact_accuracy"]}, ensure_ascii=False), flush=True)
        del model
        torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
