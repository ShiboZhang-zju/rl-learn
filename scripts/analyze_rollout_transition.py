#!/usr/bin/env python3
"""Build per-prompt rollout-state transitions (Epoch4 -> GRPO) for the fixed 200-prompt subset."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "outputs" / "grpo_v1_analysis"


def load(path: Path) -> dict[str, dict]:
    out = {}
    for line in path.open(encoding="utf-8"):
        r = json.loads(line)
        out[r["id"]] = r
    return out


def state(r: dict) -> str:
    if r["all_correct"]:
        return "all-correct"
    if r["all_wrong"]:
        return "all-wrong"
    return "mixed"


def main() -> None:
    A = load(OUT / "rollout_A_final200.jsonl")
    C = load(OUT / "rollout_C_final200.jsonl")
    assert set(A) == set(C), "id mismatch between rollout files"
    n = len(A)

    # per-prompt stats for both models
    per_prompt = []
    transition = Counter()
    for i in A:
        a, c = A[i], C[i]
        transition[f"{state(a)} -> {state(c)}"] += 1
        per_prompt.append(
            {
                "id": i,
                "gt_pattern": a["ground_truth_pattern"],
                "state_A": state(a),
                "state_C": state(c),
                "reward_mean_A": a["reward_mean"],
                "reward_mean_C": c["reward_mean"],
                "unique_A": a["unique_answer_count"],
                "unique_C": c["unique_answer_count"],
                "correct_count_A": a["correct_count"],
                "correct_count_C": c["correct_count"],
            }
        )

    def agg(rows, key):
        vals = [r[key] for r in rows]
        return {"mean": sum(vals) / len(vals), "min": min(vals), "max": max(vals)}

    result = {
        "N": n,
        "sampling": {
            "subset": "final_holdout first 200 prompts",
            "num_generations": 8,
            "temperature": 0.8,
            "top_p": 0.95,
            "max_new_tokens": 64,
            "seed": 20260828,
        },
        "state_transition": dict(sorted(transition.items())),
        "state_counts": {
            "A": {s: sum(1 for r in per_prompt if r["state_A"] == s) for s in ("all-correct", "mixed", "all-wrong")},
            "C": {s: sum(1 for r in per_prompt if r["state_C"] == s) for s in ("all-correct", "mixed", "all-wrong")},
        },
        "aggregate_A": {
            "all_correct": sum(r["state_A"] == "all-correct" for r in per_prompt) / n,
            "mixed": sum(r["state_A"] == "mixed" for r in per_prompt) / n,
            "all_wrong": sum(r["state_A"] == "all-wrong" for r in per_prompt) / n,
            "mean_reward": agg(per_prompt, "reward_mean_A")["mean"],
            "avg_unique": agg(per_prompt, "unique_A")["mean"],
            "pass_at_8": sum(r["correct_count_A"] > 0 for r in per_prompt) / n,
        },
        "aggregate_C": {
            "all_correct": sum(r["state_C"] == "all-correct" for r in per_prompt) / n,
            "mixed": sum(r["state_C"] == "mixed" for r in per_prompt) / n,
            "all_wrong": sum(r["state_C"] == "all-wrong" for r in per_prompt) / n,
            "mean_reward": agg(per_prompt, "reward_mean_C")["mean"],
            "avg_unique": agg(per_prompt, "unique_C")["mean"],
            "pass_at_8": sum(r["correct_count_C"] > 0 for r in per_prompt) / n,
        },
        "per_prompt": per_prompt,
    }
    (OUT / "rollout_transition_200.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"N": n, "state_transition": result["state_transition"], "state_counts": result["state_counts"]}, indent=2))
    print(json.dumps({"A": result["aggregate_A"], "C": result["aggregate_C"]}, indent=2))


if __name__ == "__main__":
    main()
