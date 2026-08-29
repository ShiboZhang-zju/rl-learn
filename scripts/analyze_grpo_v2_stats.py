#!/usr/bin/env python3
"""GRPO-V2 statistical attribution: paired McNemar + bootstrap, and behavioural tests.

Greedy comparisons reuse outputs/grpo_v2_final/{model}_{dataset}.jsonl (same sample ids
per dataset, so every comparison is paired):
  V2 vs V1, V2 vs Epoch4, V2 vs Epoch5   on the fresh GRPO-V2 holdout (primary)
  V2 vs V1                               on the GRPO-V1 holdout (secondary)

Behavioural comparisons use the fixed 200-prompt rollout subset and test the H1
quantities directly: all-wrong, pass@8 and all-correct indicators.

Writes outputs/grpo_v2_analysis/grpo_v2_statistical_analysis.json
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys_path = ROOT / "src"
import sys

sys.path.insert(0, str(sys_path))

from kk_sft.data import read_jsonl  # noqa: E402

FINAL = ROOT / "outputs" / "grpo_v2_final"
OUT = ROOT / "outputs" / "grpo_v2_analysis"
LABELS = ["KKK", "KKN", "KNK", "KNN", "NKK", "NKN", "NNK", "NNN"]
MODELS = ["sft_epoch4", "sft_epoch5", "grpo_v1_best", "grpo_v2_best"]


def exact_mcnemar(b: int, c: int) -> dict:
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "n_discordant": 0, "exact_p": 1.0, "chi2_corrected_p": 1.0}
    from scipy.stats import binomtest, chi2

    exact_p = binomtest(k=min(b, c), n=n, p=0.5, alternative="two-sided").pvalue
    chi2_stat = (abs(b - c) - 1.0) ** 2 / n
    return {
        "b": b,
        "c": c,
        "n_discordant": n,
        "exact_p": float(exact_p),
        "chi2_corrected_p": float(chi2.sf(chi2_stat, df=1)),
        "chi2_statistic": float(chi2_stat),
    }


def paired_bootstrap(delta_fn, idx: np.ndarray, iterations: int, rng) -> dict:
    deltas = np.empty(iterations, dtype=np.float64)
    for i in range(iterations):
        sample = rng.choice(idx, size=len(idx), replace=True)
        deltas[i] = delta_fn(sample)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {
        "mean_delta": float(deltas.mean()),
        "pct_2_5": float(lo),
        "pct_97_5": float(hi),
        "ci_crosses_zero": bool(lo < 0 < hi),
        "iterations": iterations,
        "seed": 20260829,
    }


def load_predictions(dataset: str) -> dict[str, dict[str, dict]]:
    data = {}
    for model in MODELS:
        path = FINAL / f"{model}_{dataset}.jsonl"
        rows = read_jsonl(path)
        data[model] = {row["id"]: row for row in rows}
    return data


def compare(data: dict[str, dict[str, dict]], dataset: str, left: str, right: str, rng) -> dict:
    """right - left, paired."""
    left_map, right_map = data[left], data[right]
    ids = sorted(set(left_map) & set(right_map))
    assert len(ids) == len(left_map) == len(right_map), f"id mismatch on {dataset}"
    n = len(ids)

    b_only = c_only = 0  # left correct & right wrong / left wrong & right correct
    for i in ids:
        lc = bool(left_map[i]["correct"])
        rc = bool(right_map[i]["correct"])
        if lc and not rc:
            b_only += 1
        elif rc and not lc:
            c_only += 1

    acc_left = np.array([bool(left_map[i]["correct"]) for i in ids], dtype=np.float64)
    acc_right = np.array([bool(right_map[i]["correct"]) for i in ids], dtype=np.float64)
    idx = np.arange(n)
    bootstrap = paired_bootstrap(lambda s: acc_right[s].mean() - acc_left[s].mean(), idx, 10000, rng)

    # per-class delta on the ground-truth pattern
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
        "N": n,
        "acc_left": float(acc_left.mean()),
        "acc_right": float(acc_right.mean()),
        "delta": float(acc_right.mean() - acc_left.mean()),
        "mcnemar": exact_mcnemar(b_only, c_only),
        "bootstrap": bootstrap,
        "per_class": per_class,
    }


def behaviour_tests(behavior: dict) -> dict:
    per_prompt = behavior["per_prompt"]
    n = len(per_prompt)

    def indicator_test(name: str, fn) -> dict:
        b = c = 0
        for row in per_prompt:
            v1 = fn(row["state_v1"])
            v2 = fn(row["state_v2"])
            if v1 and not v2:
                b += 1
            elif v2 and not v1:
                c += 1
        v1_rate = sum(fn(row["state_v1"]) for row in per_prompt) / n
        v2_rate = sum(fn(row["state_v2"]) for row in per_prompt) / n
        return {
            "metric": name,
            "v1_rate": v1_rate,
            "v2_rate": v2_rate,
            "delta": v2_rate - v1_rate,
            "n_v1_only": b,
            "n_v2_only": c,
            "mcnemar": exact_mcnemar(b, c),
        }

    return {
        "N": n,
        "tests": [
            indicator_test("all_wrong", lambda s: s == "all-wrong"),
            indicator_test("all_correct", lambda s: s == "all-correct"),
            indicator_test("mixed", lambda s: s == "mixed"),
        ],
        "pass_at_8": {
            "v1_rate": behavior["models"]["grpo_v1"]["pass_at_8"],
            "v2_rate": behavior["models"]["grpo_v2"]["pass_at_8"],
            "delta": behavior["models"]["grpo_v2"]["pass_at_8"] - behavior["models"]["grpo_v1"]["pass_at_8"],
        },
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260829)

    greedy = {}
    for dataset in ("grpo_v2_holdout", "grpo_v1_holdout"):
        data = load_predictions(dataset)
        pairs = [("grpo_v1_best", "grpo_v2_best"), ("sft_epoch4", "grpo_v2_best"), ("sft_epoch5", "grpo_v2_best")]
        if dataset == "grpo_v1_holdout":
            pairs = [("grpo_v1_best", "grpo_v2_best")]
        greedy[dataset] = [compare(data, dataset, left, right, rng) for left, right in pairs]

    behavior = json.loads((OUT / "behavior_comparison_200.json").read_text(encoding="utf-8"))
    result = {
        "greedy": greedy,
        "behaviour_200": behaviour_tests(behavior),
    }
    (OUT / "grpo_v2_statistical_analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for dataset, comparisons in greedy.items():
        print(f"\n=== {dataset} (greedy, paired) ===")
        for comp in comparisons:
            m = comp["mcnemar"]
            bs = comp["bootstrap"]
            print(
                f"  {comp['comparison']}: {comp['acc_left']:.4f} -> {comp['acc_right']:.4f} "
                f"delta={comp['delta'] * 100:+.2f}pp  mcnemar_p={m['exact_p']:.4f} "
                f"95%CI=[{bs['pct_2_5'] * 100:+.2f},{bs['pct_97_5'] * 100:+.2f}]pp crosses0={bs['ci_crosses_zero']}"
            )
            print("    per-class delta(pp): " + json.dumps({k: round(v["delta_pp"], 2) for k, v in comp["per_class"].items()}))

    print("\n=== behavioural tests on the 200-prompt subset (V1 -> V2) ===")
    for test in result["behaviour_200"]["tests"]:
        m = test["mcnemar"]
        print(
            f"  {test['metric']:<12} {test['v1_rate']:.4f} -> {test['v2_rate']:.4f} "
            f"delta={test['delta'] * 100:+.2f}pp  (v1_only={test['n_v1_only']}, v2_only={test['n_v2_only']})  mcnemar_p={m['exact_p']:.4f}"
        )
    pa = result["behaviour_200"]["pass_at_8"]
    print(f"  pass_at_8     {pa['v1_rate']:.4f} -> {pa['v2_rate']:.4f} delta={pa['delta'] * 100:+.2f}pp")


if __name__ == "__main__":
    main()
