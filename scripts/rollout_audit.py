#!/usr/bin/env python3
"""Audit exact-answer reward variance for sampled SFT rollouts."""

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
from kk_sft.evaluation import assignment_pattern, parse_answer  # noqa: E402


def choose_dtype(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    return torch.float32


def canonical_answer(answer: dict[str, str] | None) -> str | None:
    if answer is None:
        return None
    return json.dumps(answer, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


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
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.num_generations < 1:
        parser.error("--num-generations must be positive")

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    rows = read_jsonl(args.data_file)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=choose_dtype(args.dtype),
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model.to(args.device)
    model.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
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
            ).to(args.device)
            expanded = {
                key: value.repeat_interleave(args.num_generations, dim=0)
                for key, value in inputs.items()
            }
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
            completions = tokenizer.batch_decode(
                generated[:, prompt_len:],
                skip_special_tokens=True,
            )
            for index, row in enumerate(batch_rows):
                samples = completions[index * args.num_generations : (index + 1) * args.num_generations]
                people = row["puzzle"]["people"]
                ground_truth = row["answer"]
                generations = []
                for text in samples:
                    parsed = parse_answer(text, people)
                    correct = parsed.parsed == ground_truth
                    generations.append(
                        {
                            "text": text,
                            "parsed_answer": parsed.parsed,
                            "parse_reason": parsed.reason,
                            "format_valid": parsed.format_valid,
                            "parse_success": parsed.parsed is not None,
                            "reward": 1 if correct else 0,
                        }
                    )
                rewards = [generation["reward"] for generation in generations]
                reward_mean = sum(rewards) / len(rewards)
                variance = sum((reward - reward_mean) ** 2 for reward in rewards) / len(rewards)
                unique_answers = {
                    canonical_answer(generation["parsed_answer"])
                    for generation in generations
                    if generation["parsed_answer"] is not None
                }
                record = {
                    "id": row["id"],
                    "ground_truth": ground_truth,
                    "ground_truth_pattern": assignment_pattern(ground_truth, people),
                    "generations": generations,
                    "reward_mean": reward_mean,
                    "reward_std": variance**0.5,
                    "correct_count": sum(rewards),
                    "unique_answer_count": len(unique_answers),
                    "format_valid_count": sum(generation["format_valid"] for generation in generations),
                    "parse_success_count": sum(generation["parse_success"] for generation in generations),
                    "all_wrong": all(reward == 0 for reward in rewards),
                    "all_correct": all(reward == 1 for reward in rewards),
                    "zero_variance": variance < 1e-12,
                    "mixed_reward": any(reward == 0 for reward in rewards) and any(reward == 1 for reward in rewards),
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"audited {min(start + len(batch_rows), len(rows))}/{len(rows)}", flush=True)

    records = [json.loads(line) for line in args.output.open(encoding="utf-8")]
    total = len(records)
    rollout_total = total * args.num_generations
    groups_with_correct = sum(record["correct_count"] > 0 for record in records)
    mixed = sum(record["mixed_reward"] for record in records)
    zero = sum(record["zero_variance"] for record in records)
    all_wrong = sum(record["all_wrong"] for record in records)
    all_correct = sum(record["all_correct"] for record in records)
    reward_count_distribution = {
        str(count): sum(record["correct_count"] == count for record in records)
        for count in range(args.num_generations + 1)
    }
    summary = {
        "count_prompts": total,
        "num_generations": args.num_generations,
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
            "dtype": args.dtype,
        },
        "mean_rollout_reward": sum(record["reward_mean"] for record in records) / total,
        "average_reward_std_per_group": sum(record["reward_std"] for record in records) / total,
        "zero_variance_group_ratio": zero / total,
        "mixed_reward_group_ratio": mixed / total,
        "all_wrong_group_ratio": all_wrong / total,
        "all_correct_group_ratio": all_correct / total,
        "pass_at_8": groups_with_correct / total,
        "groups_with_at_least_one_correct_ratio": groups_with_correct / total,
        "average_correct_rollouts_per_group": sum(record["correct_count"] for record in records) / total,
        "average_unique_answers_per_group": sum(record["unique_answer_count"] for record in records) / total,
        "format_valid_rollout_ratio": sum(record["format_valid_count"] for record in records) / rollout_total,
        "parse_success_rollout_ratio": sum(record["parse_success_count"] for record in records) / rollout_total,
        "reward_count_distribution": reward_count_distribution,
        "parsed_pattern_distribution": dict(
            Counter(
                assignment_pattern(generation["parsed_answer"], record["ground_truth"].keys())
                for record in records
                for generation in record["generations"]
                if generation["parsed_answer"] is not None
            )
        ),
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
