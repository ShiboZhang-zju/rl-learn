#!/usr/bin/env python3
"""GRPO-V3 behavioural comparison on the fixed 200-prompt diagnostic subset.

Reuses the exact subset used since GRPO-V1 (first 200 prompts of the GRPO-V1 final
holdout; 8 rollouts, temperature 0.8, top_p 0.95, max_new_tokens 64, seed 20260828):

  A = SFT Epoch4   outputs/grpo_v1_analysis/rollout_A_final200.jsonl
  C = GRPO-V1      outputs/grpo_v1_analysis/rollout_C_final200.jsonl
  D = GRPO-V2      outputs/grpo_v2_analysis/rollout_D_final200.jsonl
  E = GRPO-V3      outputs/grpo_v3_analysis/rollout_E_final200.jsonl

Also reports, for every model, how many exact-all-wrong groups would regain non-zero
variance under the partial reward (the H2 mechanism, measured on real rollouts).

Writes outputs/grpo_v3_analysis/behavior_comparison_200.json
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kk_sft.reward import EXACT, PARTIAL, compute_reward, trl_group_advantages  # noqa: E402

V1_ANALYSIS = ROOT / "outputs" / "grpo_v1_analysis"
V2_ANALYSIS = ROOT / "outputs" / "grpo_v2_analysis"
OUT_DIR = ROOT / "outputs" / "grpo_v3_analysis"
PEOPLE = ["Alice", "Bob", "Carol"]


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


def rescue_stats(records: dict[str, dict]) -> dict:
    """How many exact-all-wrong groups regain non-zero variance under the partial reward."""
    all_wrong = 0
    rescued = 0
    advantage_magnitudes = []
    for record in records.values():
        if not record["all_wrong"]:
            continue
        all_wrong += 1
        gt = record["ground_truth"]
        exact, partial = [], []
        for generation in record["generations"]:
            parsed = generation["parsed_answer"]
            exact.append(compute_reward(parsed, gt, PEOPLE, EXACT))
            partial.append(compute_reward(parsed, gt, PEOPLE, PARTIAL))
        if len({round(value, 6) for value in partial}) > 1:
            advantages, _ = trl_group_advantages(partial)
            if float(advantages.abs().max()) > 1e-3:
                rescued += 1
                advantage_magnitudes.append(float(advantages.abs().max()))
    return {
        "exact_all_wrong_groups": all_wrong,
        "rescued": rescued,
        "still_zero_variance": all_wrong - rescued,
        "rescue_rate": (rescued / all_wrong) if all_wrong else None,
        "mean_max_abs_advantage_when_rescued": (sum(advantage_magnitudes) / len(advantage_magnitudes)) if advantage_magnitudes else None,
    }


def transitions(base: dict[str, dict], target: dict[str, dict]) -> dict:
    counter = Counter(f"{state(base[i])} -> {state(target[i])}" for i in base)
    return dict(sorted(counter.items()))


def class_accuracy(records: dict[str, dict]) -> dict:
    per_class: dict[str, list[int]] = {}
    for record in records.values():
        bucket = per_class.setdefault(record["ground_truth_pattern"], [0, 0])
        bucket[0] += record["correct_count"]
        bucket[1] += 8
    return {
        cls: {"rollout_accuracy": correct / (total * 8), "support_prompts": total // 8}
        for cls, (correct, total) in sorted(per_class.items())
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    a = load(V1_ANALYSIS / "rollout_A_final200.jsonl")
    c = load(V1_ANALYSIS / "rollout_C_final200.jsonl")
    d = load(V2_ANALYSIS / "rollout_D_final200.jsonl")
    e = load(OUT_DIR / "rollout_E_final200.jsonl")
    assert set(a) == set(c) == set(d) == set(e), "rollout subsets do not match"

    result = {
        "subset": "grpo_v1_final_holdout first 200 prompts",
        "sampling": {"num_generations": 8, "temperature": 0.8, "top_p": 0.95, "max_new_tokens": 64, "seed": 20260828},
        "models": {
            "sft_epoch4": aggregate(a),
            "grpo_v1": aggregate(c),
            "grpo_v2": aggregate(d),
            "grpo_v3": aggregate(e),
        },
        "rescue": {
            "sft_epoch4": rescue_stats(a),
            "grpo_v1": rescue_stats(c),
            "grpo_v2": rescue_stats(d),
            "grpo_v3": rescue_stats(e),
        },
        "transitions": {
            "epoch4_to_grpo_v1": transitions(a, c),
            "epoch4_to_grpo_v2": transitions(a, d),
            "epoch4_to_grpo_v3": transitions(a, e),
            "grpo_v2_to_grpo_v3": transitions(d, e),
        },
        "per_prompt": [
            {
                "id": key,
                "gt_pattern": a[key]["ground_truth_pattern"],
                "state_epoch4": state(a[key]),
                "state_v1": state(c[key]),
                "state_v2": state(d[key]),
                "state_v3": state(e[key]),
                "reward_mean_epoch4": a[key]["reward_mean"],
                "reward_mean_v1": c[key]["reward_mean"],
                "reward_mean_v2": d[key]["reward_mean"],
                "reward_mean_v3": e[key]["reward_mean"],
                "unique_epoch4": a[key]["unique_answer_count"],
                "unique_v1": c[key]["unique_answer_count"],
                "unique_v2": d[key]["unique_answer_count"],
                "unique_v3": e[key]["unique_answer_count"],
            }
            for key in a
        ],
        "class_rollout_accuracy": {
            "sft_epoch4": class_accuracy(a),
            "grpo_v1": class_accuracy(c),
            "grpo_v2": class_accuracy(d),
            "grpo_v3": class_accuracy(e),
        },
    }
    (OUT_DIR / "behavior_comparison_200.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    order = ["sft_epoch4", "grpo_v1", "grpo_v2", "grpo_v3"]
    print(f"{'metric':<24}" + "".join(f"{m:>12}" for m in order))
    for field in ("mean_reward", "pass_at_8", "mixed", "all_correct", "all_wrong", "avg_unique", "avg_correct_per_group"):
        print(f"{field:<24}" + "".join(f"{result['models'][m][field]:>12.4f}" for m in order))
    print("\nH2 mechanism on real rollouts (exact-all-wrong groups):")
    for name in order:
        r = result["rescue"][name]
        rate = f"{r['rescue_rate'] * 100:.1f}%" if r["rescue_rate"] is not None else "n/a"
        print(f"  {name:<12} all_wrong={r['exact_all_wrong_groups']:>3}  rescued={r['rescued']:>3}  rescue_rate={rate}")
    print("\nEpoch4 -> V1:", json.dumps(result["transitions"]["epoch4_to_grpo_v1"], ensure_ascii=False))
    print("Epoch4 -> V2:", json.dumps(result["transitions"]["epoch4_to_grpo_v2"], ensure_ascii=False))
    print("Epoch4 -> V3:", json.dumps(result["transitions"]["epoch4_to_grpo_v3"], ensure_ascii=False))
    print("V2 -> V3    :", json.dumps(result["transitions"]["grpo_v2_to_grpo_v3"], ensure_ascii=False))


if __name__ == "__main__":
    main()
