#!/usr/bin/env python3
"""GRPO-V2 behavioural comparison on the fixed 200-prompt diagnostic subset.

Reuses the exact subset GRPO-V1 used (first 200 prompts of the GRPO-V1 final holdout,
8 rollouts, temperature 0.8, top_p 0.95, max_new_tokens 64, seed 20260828):

  A = SFT Epoch4   outputs/grpo_v1_analysis/rollout_A_final200.jsonl
  C = GRPO-V1      outputs/grpo_v1_analysis/rollout_C_final200.jsonl
  D = GRPO-V2      outputs/grpo_v2_analysis/rollout_D_final200.jsonl

Writes outputs/grpo_v2_analysis/behavior_comparison_200.json
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V1_ANALYSIS = ROOT / "outputs" / "grpo_v1_analysis"
OUT_DIR = ROOT / "outputs" / "grpo_v2_analysis"


def load(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        out[record["id"]] = record
    return out


def state(record: dict) -> str:
    if record["all_correct"]:
        return "all-correct"
    if record["all_wrong"]:
        return "all-wrong"
    return "mixed"


def aggregate(records: dict[str, dict]) -> dict:
    n = len(records)
    values = list(records.values())
    return {
        "n": n,
        "mean_reward": sum(r["reward_mean"] for r in values) / n,
        "pass_at_8": sum(r["correct_count"] > 0 for r in values) / n,
        "mixed": sum(state(r) == "mixed" for r in values) / n,
        "all_correct": sum(state(r) == "all-correct" for r in values) / n,
        "all_wrong": sum(state(r) == "all-wrong" for r in values) / n,
        "avg_unique": sum(r["unique_answer_count"] for r in values) / n,
        "avg_correct_per_group": sum(r["correct_count"] for r in values) / n,
        "format_valid_ratio": sum(r["format_valid_count"] for r in values) / (n * 8),
        "parse_success_ratio": sum(r["parse_success_count"] for r in values) / (n * 8),
        "state_counts": dict(Counter(state(r) for r in values)),
    }


def transitions(base: dict[str, dict], target: dict[str, dict]) -> dict:
    counter = Counter(f"{state(base[i])} -> {state(target[i])}" for i in base)
    return dict(sorted(counter.items()))


def class_accuracy(records: dict[str, dict]) -> dict:
    per_class: dict[str, list[int]] = {}
    for record in records.values():
        cls = record["ground_truth_pattern"]
        bucket = per_class.setdefault(cls, [0, 0])
        bucket[0] += record["correct_count"]
        bucket[1] += 8
    return {
        cls: {"rollout_accuracy": total_correct / (total * 8), "support_prompts": total // 8}
        for cls, (total_correct, total) in sorted(per_class.items())
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    a = load(V1_ANALYSIS / "rollout_A_final200.jsonl")
    c = load(V1_ANALYSIS / "rollout_C_final200.jsonl")
    d = load(OUT_DIR / "rollout_D_final200.jsonl")
    assert set(a) == set(c) == set(d), "rollout subsets do not match"

    result = {
        "subset": "grpo_v1_final_holdout first 200 prompts",
        "sampling": {"num_generations": 8, "temperature": 0.8, "top_p": 0.95, "max_new_tokens": 64, "seed": 20260828},
        "models": {
            "sft_epoch4": aggregate(a),
            "grpo_v1": aggregate(c),
            "grpo_v2": aggregate(d),
        },
        "transitions": {
            "epoch4_to_grpo_v1": transitions(a, c),
            "epoch4_to_grpo_v2": transitions(a, d),
        },
        "per_prompt": [
            {
                "id": key,
                "gt_pattern": a[key]["ground_truth_pattern"],
                "state_epoch4": state(a[key]),
                "state_v1": state(c[key]),
                "state_v2": state(d[key]),
                "reward_mean_epoch4": a[key]["reward_mean"],
                "reward_mean_v1": c[key]["reward_mean"],
                "reward_mean_v2": d[key]["reward_mean"],
                "unique_epoch4": a[key]["unique_answer_count"],
                "unique_v1": c[key]["unique_answer_count"],
                "unique_v2": d[key]["unique_answer_count"],
            }
            for key in a
        ],
        "class_rollout_accuracy": {
            "sft_epoch4": class_accuracy(a),
            "grpo_v1": class_accuracy(c),
            "grpo_v2": class_accuracy(d),
        },
    }
    out = OUT_DIR / "behavior_comparison_200.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"{'metric':<24}{'Epoch4':>12}{'GRPO-V1':>12}{'GRPO-V2':>12}")
    for field in ("mean_reward", "pass_at_8", "mixed", "all_correct", "all_wrong", "avg_unique", "avg_correct_per_group"):
        row = result["models"]
        print(f"{field:<24}{row['sft_epoch4'][field]:>12.4f}{row['grpo_v1'][field]:>12.4f}{row['grpo_v2'][field]:>12.4f}")
    print("\nEpoch4 -> GRPO-V1:", json.dumps(result["transitions"]["epoch4_to_grpo_v1"], ensure_ascii=False))
    print("Epoch4 -> GRPO-V2:", json.dumps(result["transitions"]["epoch4_to_grpo_v2"], ensure_ascii=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
