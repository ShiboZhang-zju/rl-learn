#!/usr/bin/env python3
"""GRPO-V3 statistical attribution: paired McNemar + bootstrap, and H2 behavioural tests.

Greedy comparisons use outputs/grpo_v3_final/{model}_{dataset}.jsonl (identical sample ids
per dataset, so every comparison is paired). The primary causal comparison is

    V3 (partial reward) vs V2 (exact reward)

because every other setting is locked identical. It is read on the freshly generated
GRPO-V3 holdout (N=2000, seed 20260902), which no earlier round has seen.

The behavioural tests use the fixed 200-prompt rollout subset and test the H2 quantities
directly: all-wrong, pass@8, all-correct, and mixed->all-wrong transitions.

Writes outputs/grpo_v3_analysis/grpo_v3_statistical_analysis.json
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kk_sft.data import read_jsonl  # noqa: E402

FINAL = ROOT / "outputs" / "grpo_v3_final"
OUT = ROOT / "outputs" / "grpo_v3_analysis"
LABELS = ["KKK", "KKN", "KNK", "KNN", "NKK", "NKN", "NNK", "NNN"]
MODELS = ["sft_epoch4", "sft_epoch5", "grpo_v1_best", "grpo_v2_best", "grpo_v3_best"]


def exact_mcnemar(b: int, c: int) -> dict:
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "n_discordant": 0, "exact_p": 1.0}
    from scipy.stats import binomtest

    return {"b": b, "c": c, "n_discordant": n, "exact_p": float(binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue)}


def paired_bootstrap(delta_fn, idx: np.ndarray, iterations: int, rng) -> dict:
    deltas = np.empty(iterations, dtype=np.float64)
    for i in range(iterations):
        sample = rng.choice(idx, size=len(idx), replace=True)
        deltas[i] = delta_fn(sample)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"mean_delta": float(deltas.mean()), "pct_2_5": float(lo), "pct_97_5": float(hi), "ci_crosses_zero": bool(lo < 0 < hi), "seed": 20260829}


def load_predictions(dataset: str) -> dict[str, dict[str, dict]]:
    return {model: {row["id"]: row for row in read_jsonl(FINAL / f"{model}_{dataset}.jsonl")} for model in MODELS}


def compare(data, dataset: str, left: str, right: str, rng) -> dict:
    """right - left, paired."""
    left_map, right_map = data[left], data[right]
    ids = sorted(set(left_map) & set(right_map))
    assert len(ids) == len(left_map) == len(right_map), f"id mismatch on {dataset}"
    b = c = 0
    for i in ids:
        lc, rc = bool(left_map[i]["correct"]), bool(right_map[i]["correct"])
        if lc and not rc:
            b += 1
        elif rc and not lc:
            c += 1
    acc_left = np.array([bool(left_map[i]["correct"]) for i in ids], dtype=np.float64)
    acc_right = np.array([bool(right_map[i]["correct"]) for i in ids], dtype=np.float64)
    idx = np.arange(len(ids))
    bootstrap = paired_bootstrap(lambda s: acc_right[s].mean() - acc_left[s].mean(), idx, 10000, rng)

    by_gt: dict[str, list[str]] = defaultdict(list)
    for i in ids:
        by_gt[left_map[i]["ground_truth_pattern"]].append(i)
    per_class = {}
    for gt in LABELS:
        group = by_gt.get(gt, [])
        if not group:
            continue
        la = sum(bool(left_map[i]["correct"]) for i in group) / len(group)
        ra = sum(bool(right_map[i]["correct"]) for i in group) / len(group)
        per_class[gt] = {"N": len(group), "acc_left": la, "acc_right": ra, "delta_pp": (ra - la) * 100}

    return {
        "dataset": dataset,
        "comparison": f"{right}_minus_{left}",
        "N": len(ids),
        "acc_left": float(acc_left.mean()),
        "acc_right": float(acc_right.mean()),
        "delta": float(acc_right.mean() - acc_left.mean()),
        "mcnemar": exact_mcnemar(b, c),
        "bootstrap": bootstrap,
        "per_class": per_class,
    }


def behaviour_tests(behavior: dict) -> dict:
    per_prompt = behavior["per_prompt"]
    n = len(per_prompt)

    def indicator_test(name: str, field_a: str, field_b: str, predicate) -> dict:
        b = c = 0
        for row in per_prompt:
            left, right = predicate(row[field_a]), predicate(row[field_b])
            if left and not right:
                b += 1
            elif right and not left:
                c += 1
        left_rate = sum(predicate(row[field_a]) for row in per_prompt) / n
        right_rate = sum(predicate(row[field_b]) for row in per_prompt) / n
        return {
            "metric": name,
            "v2_rate": left_rate,
            "v3_rate": right_rate,
            "delta": right_rate - left_rate,
            "n_v2_only": b,
            "n_v3_only": c,
            "mcnemar": exact_mcnemar(b, c),
        }

    is_wrong = lambda s: s == "all-wrong"  # noqa: E731
    is_correct = lambda s: s == "all-correct"  # noqa: E731
    is_mixed = lambda s: s == "mixed"  # noqa: E731

    return {
        "N": n,
        "tests": [
            indicator_test("all_wrong", "state_v2", "state_v3", is_wrong),
            indicator_test("all_correct", "state_v2", "state_v3", is_correct),
            indicator_test("mixed", "state_v2", "state_v3", is_mixed),
        ],
        "pass_at_8": {
            "v2_rate": behavior["models"]["grpo_v2"]["pass_at_8"],
            "v3_rate": behavior["models"]["grpo_v3"]["pass_at_8"],
            "delta": behavior["models"]["grpo_v3"]["pass_at_8"] - behavior["models"]["grpo_v2"]["pass_at_8"],
        },
        "epoch4_to_v2_mixed_to_all_wrong": behavior["transitions"]["epoch4_to_grpo_v2"].get("mixed -> all-wrong", 0),
        "epoch4_to_v3_mixed_to_all_wrong": behavior["transitions"]["epoch4_to_grpo_v3"].get("mixed -> all-wrong", 0),
        "epoch4_to_v2_mixed_to_all_correct": behavior["transitions"]["epoch4_to_grpo_v2"].get("mixed -> all-correct", 0),
        "epoch4_to_v3_mixed_to_all_correct": behavior["transitions"]["epoch4_to_grpo_v3"].get("mixed -> all-correct", 0),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260829)

    greedy = {}
    for dataset, pairs in (
        ("grpo_v3_holdout", [("grpo_v2_best", "grpo_v3_best"), ("grpo_v1_best", "grpo_v3_best"), ("sft_epoch5", "grpo_v3_best"), ("sft_epoch4", "grpo_v3_best")]),
        ("grpo_v2_holdout", [("grpo_v2_best", "grpo_v3_best")]),
    ):
        data = load_predictions(dataset)
        greedy[dataset] = [compare(data, dataset, left, right, rng) for left, right in pairs]

    behavior = json.loads((OUT / "behavior_comparison_200.json").read_text(encoding="utf-8"))
    result = {"greedy": greedy, "behaviour_200": behaviour_tests(behavior)}
    (OUT / "grpo_v3_statistical_analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for dataset, comparisons in greedy.items():
        print(f"\n=== {dataset} (greedy, paired) ===")
        for comp in comparisons:
            bs = comp["bootstrap"]
            print(
                f"  {comp['comparison']}: {comp['acc_left']:.4f} -> {comp['acc_right']:.4f} "
                f"delta={comp['delta'] * 100:+.2f}pp  mcnemar_p={comp['mcnemar']['exact_p']:.4f} "
                f"95%CI=[{bs['pct_2_5'] * 100:+.2f},{bs['pct_97_5'] * 100:+.2f}]pp crosses0={bs['ci_crosses_zero']}"
            )
            print("    per-class delta(pp): " + json.dumps({k: round(v["delta_pp"], 2) for k, v in comp["per_class"].items()}))

    print("\n=== behavioural tests on the 200-prompt subset (V2 -> V3) ===")
    for test in result["behaviour_200"]["tests"]:
        print(
            f"  {test['metric']:<12} {test['v2_rate']:.4f} -> {test['v3_rate']:.4f} delta={test['delta'] * 100:+.2f}pp "
            f"(v2_only={test['n_v2_only']}, v3_only={test['n_v3_only']}) p={test['mcnemar']['exact_p']:.4f}"
        )
    pa = result["behaviour_200"]["pass_at_8"]
    print(f"  pass_at_8     {pa['v2_rate']:.4f} -> {pa['v3_rate']:.4f} delta={pa['delta'] * 100:+.2f}pp")
    print(f"  mixed->all-wrong   Epoch4->V2 = {result['behaviour_200']['epoch4_to_v2_mixed_to_all_wrong']}"
          f" | Epoch4->V3 = {result['behaviour_200']['epoch4_to_v3_mixed_to_all_wrong']}")
    print(f"  mixed->all-correct Epoch4->V2 = {result['behaviour_200']['epoch4_to_v2_mixed_to_all_correct']}"
          f" | Epoch4->V3 = {result['behaviour_200']['epoch4_to_v3_mixed_to_all_correct']}")


if __name__ == "__main__":
    main()
