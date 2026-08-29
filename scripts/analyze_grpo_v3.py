#!/usr/bin/env python3
"""Summarize GRPO-V3 training dynamics and compare them against GRPO-V2 (and V1).

Normalization note: GRPO-V2 was trained with `reward.mode: exact`, so its log has no
dual-track fields (`exact_*` / `shaped_*`). For V2 the shaped reward *is* the exact
reward, so those fields are reconstructed:

    exact_reward_mean   = reward_mean
    shaped_reward_mean  = reward_mean
    exact_mixed_ratio   = mixed_group_ratio
    exact_all_*_ratio   = all_*_ratio
    *_zero_variance*    = zero_variance_group_ratio
    n_exact_all_wrong   = round(all_wrong_ratio * groups_per_step)
    rescue ratio        = 0.0   (by construction: shaped == exact)

V1 is only comparable on the reward-function statistics (see the V2 report erratum).

Writes outputs/grpo_v3_analysis/grpo_v3_training_summary.json
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
V3_DIR = ROOT / "outputs" / "grpo_v3_partial"
V2_DIR = ROOT / "outputs" / "grpo_v2_kl001"
V1_DIR = ROOT / "outputs" / "grpo_v1"
OUT_DIR = ROOT / "outputs" / "grpo_v3_analysis"

BUCKETS = [(1, 100), (101, 200), (201, 300), (301, 400), (401, 500), (501, 600), (601, 625)]
GROUPS_PER_STEP = 8

FIELDS = [
    "exact_reward_mean",
    "shaped_reward_mean",
    "shaped_reward_std",
    "exact_mixed_ratio",
    "exact_all_correct_ratio",
    "exact_all_wrong_ratio",
    "exact_zero_variance_ratio",
    "shaped_zero_variance_ratio",
    "exact_all_wrong_but_shaped_nonzero_variance_ratio",
    "avg_unique_answers",
    "entropy",
    "kl",
    "loss",
    "grad_norm",
    "peak_memory_allocated_gb",
]


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize_v2(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        entry = dict(row)
        reward_mean = row.get("reward_mean")
        entry["exact_reward_mean"] = reward_mean
        entry["shaped_reward_mean"] = reward_mean
        entry["shaped_reward_std"] = None
        entry["exact_mixed_ratio"] = row.get("mixed_group_ratio")
        entry["exact_all_correct_ratio"] = row.get("all_correct_ratio")
        entry["exact_all_wrong_ratio"] = row.get("all_wrong_ratio")
        entry["exact_zero_variance_ratio"] = row.get("zero_variance_group_ratio")
        entry["shaped_zero_variance_ratio"] = row.get("zero_variance_group_ratio")
        entry["exact_all_wrong_but_shaped_nonzero_variance_ratio"] = 0.0
        entry["n_exact_all_wrong_groups"] = (
            round(row["all_wrong_ratio"] * GROUPS_PER_STEP) if row.get("all_wrong_ratio") is not None else 0
        )
        entry["n_rescued_groups"] = 0
        out.append(entry)
    return out


def bucket_stats(rows: list[dict], low: int, high: int) -> dict:
    selected = [row for row in rows if low <= row.get("step", -1) <= high]
    out = {"steps": f"{low}-{high}", "n": len(selected)}
    for field in FIELDS:
        values = [row[field] for row in selected if row.get(field) is not None]
        out[field] = mean(values) if values else None
    all_wrong = sum(int(row.get("n_exact_all_wrong_groups") or 0) for row in selected)
    rescued = sum(int(row.get("n_rescued_groups") or 0) for row in selected)
    out["n_exact_all_wrong_groups"] = all_wrong
    out["n_rescued_groups"] = rescued
    out["pooled_rescue_rate"] = (rescued / all_wrong) if all_wrong else None
    return out


def val_table(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        {
            "step": row["step"],
            "exact_accuracy": row["exact_accuracy"],
            "format_accuracy": row["format_accuracy"],
            "parse_success_rate": row["parse_success_rate"],
        }
        for row in json.loads(path.read_text(encoding="utf-8"))
    ]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    v3 = load_jsonl(V3_DIR / "train_metrics.jsonl")
    v2 = normalize_v2(load_jsonl(V2_DIR / "train_metrics.jsonl"))
    v1 = normalize_v2(load_jsonl(V1_DIR / "grpo_v1_train_metrics.jsonl"))

    summary = {
        "v3_steps": len(v3),
        "v2_steps": len(v2),
        "v3_train_buckets": [bucket_stats(v3, low, high) for low, high in BUCKETS],
        "v2_train_buckets": [bucket_stats(v2, low, high) for low, high in BUCKETS],
        "v1_train_buckets": [bucket_stats(v1, low, high) for low, high in BUCKETS],
        "v3_val": val_table(V3_DIR / "val_metrics.json"),
        "v2_val": val_table(V2_DIR / "val_metrics.json"),
        "v3_kl_trajectory": [
            {"step": row["step"], "kl": row.get("kl"), "loss": row.get("loss"), "grad_norm": row.get("grad_norm")}
            for row in v3
            if row.get("kl") is not None
        ],
    }
    kl = [row["kl"] for row in v3 if row.get("kl") is not None]
    if kl:
        summary["v3_kl_summary"] = {"n": len(kl), "first": kl[0], "last": kl[-1], "min": min(kl), "max": max(kl), "mean": mean(kl)}
    (OUT_DIR / "grpo_v3_training_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    cols = ["exact_reward_mean", "shaped_reward_mean", "exact_mixed_ratio", "exact_all_correct_ratio", "exact_all_wrong_ratio", "pooled_rescue_rate", "kl"]
    print("=== V3 (partial reward) ===")
    print(f"{'bucket':<12}" + "".join(f"{c[:16]:>18}" for c in cols))
    for b in summary["v3_train_buckets"]:
        print(f"{b['steps']:<12}" + "".join(f"{(b[c] if b[c] is not None else 0):>18.4f}" for c in cols))
    print("\n=== V2 (exact reward) ===")
    print(f"{'bucket':<12}" + "".join(f"{c[:16]:>18}" for c in cols))
    for b in summary["v2_train_buckets"]:
        print(f"{b['steps']:<12}" + "".join(f"{(b[c] if b[c] is not None else 0):>18.4f}" for c in cols))

    print("\n=== V3 val ===")
    for row in summary["v3_val"]:
        print(f"  step {row['step']:>4}  exact={row['exact_accuracy']:.4f}  format={row['format_accuracy']:.4f}  parse={row['parse_success_rate']:.4f}")
    print("\n=== V2 val (reference) ===")
    for row in summary["v2_val"]:
        print(f"  step {row['step']:>4}  exact={row['exact_accuracy']:.4f}")
    print("\nV3 KL summary:", json.dumps(summary.get("v3_kl_summary"), ensure_ascii=False))


if __name__ == "__main__":
    main()
