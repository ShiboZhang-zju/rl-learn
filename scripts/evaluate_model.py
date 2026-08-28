#!/usr/bin/env python3
"""Batch-evaluate a base model or PEFT adapter with the shared answer parser."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kk_sft.data import read_jsonl  # noqa: E402
from kk_sft.evaluation import aggregate_metrics, assignment_pattern, parse_answer  # noqa: E402


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def choose_dtype(name: str, device: str) -> torch.dtype:
    if name == "auto":
        return torch.bfloat16 if device == "cuda" else torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    return torch.float32


def load_config(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def build_record(row: dict, prompt_text: str, prediction: str) -> dict:
    parsed = parse_answer(prediction, row["puzzle"]["people"])
    return {
        "id": row["id"],
        "prompt": prompt_text,
        "prediction": prediction,
        "parsed_answer": parsed.parsed,
        "parse_reason": parsed.reason,
        "format_valid": parsed.format_valid,
        "ground_truth": row["answer"],
        "correct": parsed.parsed == row["answer"],
        "ground_truth_pattern": assignment_pattern(row["answer"], row["puzzle"]["people"]),
        "prediction_pattern": assignment_pattern(parsed.parsed, row["puzzle"]["people"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/eval_p800.yaml"))
    parser.add_argument("--model")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--data-file", type=Path, default=Path("data/processed/sft_test.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-prompt-length", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--flush-every", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_name = args.model or cfg.get("model_name_or_path")
    if not model_name:
        parser.error("--model or model_name_or_path in --config is required")
    batch_size = args.batch_size or int(cfg.get("batch_size", 16))
    max_prompt_length = args.max_prompt_length or int(cfg.get("max_prompt_length", 512))
    max_new_tokens = args.max_new_tokens or int(cfg.get("max_new_tokens", 320))
    dtype_name = args.dtype or cfg.get("dtype", "auto")
    stop_strings = list(cfg.get("stop_strings", []))
    device = choose_device(args.device or cfg.get("device", "auto"))
    model_dtype = choose_dtype(dtype_name, device)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = read_jsonl(args.data_file)
    start = max(0, args.offset)
    end = len(rows) if args.limit is None else min(len(rows), start + args.limit)
    rows = rows[start:end]
    if not rows:
        raise ValueError("No evaluation rows selected")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=model_dtype, trust_remote_code=True)
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model.to(device)
    model.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with args.output.open("w", encoding="utf-8") as handle:
        for batch_start in range(0, len(rows), batch_size):
            batch_rows = rows[batch_start : batch_start + batch_size]
            prompt_texts = [
                tokenizer.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True)
                for row in batch_rows
            ]
            inputs = tokenizer(
                prompt_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_prompt_length,
            ).to(device)
            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "pad_token_id": tokenizer.pad_token_id,
                "use_cache": True,
            }
            if stop_strings:
                generation_kwargs["stop_strings"] = stop_strings
                generation_kwargs["tokenizer"] = tokenizer
            with torch.inference_mode():
                generated = model.generate(**inputs, **generation_kwargs)
            completion_ids = generated[:, inputs["input_ids"].shape[1] :]
            predictions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
            batch_records = [
                build_record(row, prompt_text, prediction)
                for row, prompt_text, prediction in zip(batch_rows, prompt_texts, predictions)
            ]
            for record in batch_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            records.extend(batch_records)
            completed = batch_start + len(batch_rows)
            print(f"evaluated {completed}/{len(rows)}", flush=True)

    print(json.dumps(aggregate_metrics(records), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
