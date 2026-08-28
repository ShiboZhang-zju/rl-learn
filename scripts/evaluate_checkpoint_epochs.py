#!/usr/bin/env python3
"""Evaluate every saved SFT checkpoint on train and validation exact accuracy."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kk_sft.data import read_jsonl  # noqa: E402
from kk_sft.evaluation import parse_answer  # noqa: E402


def pattern(answer: dict, people: list[str]) -> str:
    return "".join("K" if answer.get(person) == "knight" else "N" for person in people)


def evaluate(model, tokenizer, rows: list[dict], device: str, batch_size: int, max_new_tokens: int, max_prompt_length: int) -> dict:
    correct = 0
    format_valid = 0
    predictions = Counter()
    records = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        prompts = [tokenizer.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True) for row in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_prompt_length).to(device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
        completion_ids = generated[:, inputs["input_ids"].shape[1] :]
        texts = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        for row, text in zip(batch, texts):
            parsed = parse_answer(text, row["puzzle"]["people"])
            gt = pattern(row["answer"], row["puzzle"]["people"])
            pred = pattern(parsed.parsed, row["puzzle"]["people"])
            correct += parsed.parsed == row["answer"]
            format_valid += parsed.format_valid
            predictions[pred] += 1
    return {
        "count": len(rows),
        "exact": correct,
        "exact_accuracy": correct / len(rows),
        "format_valid": format_valid,
        "format_accuracy": format_valid / len(rows),
        "prediction_distribution": dict(sorted(predictions.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--val-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    args = parser.parse_args()

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    train_rows = read_jsonl(args.train_file)
    val_rows = read_jsonl(args.val_file)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    checkpoints = sorted(
        [path for path in args.adapter_dir.glob("checkpoint-*") if path.is_dir()],
        key=lambda path: int(re.search(r"(\d+)$", path.name).group(1)),
    )
    results = []
    for checkpoint in checkpoints:
        model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.bfloat16, trust_remote_code=True)
        model = PeftModel.from_pretrained(model, checkpoint)
        model.to(args.device)
        model.eval()
        step = int(re.search(r"(\d+)$", checkpoint.name).group(1))
        train_metrics = evaluate(model, tokenizer, train_rows, args.device, args.batch_size, args.max_new_tokens, args.max_prompt_length)
        val_metrics = evaluate(model, tokenizer, val_rows, args.device, args.batch_size, args.max_new_tokens, args.max_prompt_length)
        results.append({"checkpoint": str(checkpoint), "step": step, "epoch": step / (len(train_rows) / 16), "train": train_metrics, "val": val_metrics})
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)
        del model
        torch.cuda.empty_cache()
    best = max(results, key=lambda item: (item["val"]["exact_accuracy"], -item["step"]))
    output = {"adapter_dir": str(args.adapter_dir), "checkpoints": results, "best_checkpoint": best}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"best_checkpoint": best["checkpoint"], "best_val_exact": best["val"]["exact_accuracy"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
