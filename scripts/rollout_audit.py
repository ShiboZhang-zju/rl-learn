#!/usr/bin/env python3
"""Audit sampled rollout reward variance for a SFT adapter."""

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
from kk_sft.evaluation import assignment_pattern, score_completion  # noqa: E402


def choose_dtype(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    return torch.float32


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--data-file", type=Path, default=Path("data/processed/answer_only_test.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    torch.manual_seed(args.seed)
    rows = read_jsonl(args.data_file)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=choose_dtype(args.dtype), trust_remote_code=True)
    model = PeftModel.from_pretrained(model, args.adapter)
    model.to("cuda")
    model.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    group_records: list[dict] = []
    with args.output.open("w", encoding="utf-8") as handle:
        for start in range(0, len(rows), args.batch_size):
            batch_rows = rows[start : start + args.batch_size]
            prompt_texts = [
                tokenizer.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True)
                for row in batch_rows
            ]
            inputs = tokenizer(
                prompt_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_prompt_length,
            ).to("cuda")
            expanded = {key: value.repeat_interleave(args.num_generations, dim=0) for key, value in inputs.items()}
            with torch.inference_mode():
                generated = model.generate(
                    **expanded,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=True,
                )
            prompt_len = expanded["input_ids"].shape[1]
            completions = tokenizer.batch_decode(generated[:, prompt_len:], skip_special_tokens=True)
            for index, row in enumerate(batch_rows):
                samples = completions[index * args.num_generations : (index + 1) * args.num_generations]
                scored = [score_completion(text, row["puzzle"]["people"], row["answer"]) for text in samples]
                rewards = [result.total for result in scored]
                parsed_patterns = [assignment_pattern(result.parsed_answer, row["puzzle"]["people"]) for result in scored]
                unique_answers = len(set(parsed_patterns))
                mean_reward = sum(rewards) / len(rewards)
                variance = sum((reward - mean_reward) ** 2 for reward in rewards) / len(rewards)
                record = {
                    "id": row["id"],
                    "ground_truth_pattern": assignment_pattern(row["answer"], row["puzzle"]["people"]),
                    "rewards": rewards,
                    "exact_correct": [result.exact_correct for result in scored],
                    "format_valid": [result.format_valid for result in scored],
                    "parsed_patterns": parsed_patterns,
                    "completions": samples,
                    "reward_mean": mean_reward,
                    "reward_std": variance**0.5,
                    "unique_answer_count": unique_answers,
                    "correct_rollout_count": sum(result.exact_correct for result in scored),
                    "all_wrong": not any(result.exact_correct for result in scored),
                    "all_correct": all(result.exact_correct for result in scored),
                "zero_variance": variance < 1e-12,
                "mixed_reward": any(result.exact_correct for result in scored) and not all(result.exact_correct for result in scored),

                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"audited {min(start + len(batch_rows), len(rows))}/{len(rows)}", flush=True)

    records = [json.loads(line) for line in args.output.open(encoding="utf-8")]
    total = len(records)
    groups_with_correct = sum(record["correct_rollout_count"] > 0 for record in records)
    mixed = sum(record["mixed_reward"] for record in records)
    zero = sum(record["zero_variance"] for record in records)
    all_wrong = sum(record["all_wrong"] for record in records)
    all_correct = sum(record["all_correct"] for record in records)
    summary = {
        "count_prompts": total,
        "num_generations": args.num_generations,
        "sampling": {"temperature": args.temperature, "top_p": args.top_p, "seed": args.seed},
        "zero_variance_group_ratio": zero / total,
        "mixed_reward_group_ratio": mixed / total,
        "all_wrong_group_ratio": all_wrong / total,
        "all_correct_group_ratio": all_correct / total,
        "groups_with_at_least_one_correct_ratio": groups_with_correct / total,
        "pass_at_8": groups_with_correct / total,
        "mean_unique_answers_per_group": sum(record["unique_answer_count"] for record in records) / total,
        "mean_correct_rollouts_per_group": sum(record["correct_rollout_count"] for record in records) / total,
        "mean_reward": sum(record["reward_mean"] for record in records) / total,
        "mean_reward_std": sum(record["reward_std"] for record in records) / total,
        "format_valid_rollout_ratio": sum(valid for record in records for valid in record["format_valid"]) / (total * args.num_generations),
        "parsed_pattern_distribution": dict(Counter(pattern for record in records for pattern in record["parsed_patterns"])),
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
