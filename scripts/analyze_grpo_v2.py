#!/usr/bin/env python3
"""Summarize GRPO-V2 training dynamics and compare them against GRPO-V1.

Reads (never writes into the V1 directories):
  outputs/grpo_v2_kl001/train_metrics.jsonl
  outputs/grpo_v2_kl001/val_metrics.json
  outputs/grpo_v1/grpo_v1_train_metrics.jsonl      (V1 reference)
  outputs/grpo_v1/grpo_v1_val_metrics.json         (V1 reference)

Writes:
  outputs/grpo_v2_analysis/grpo_v2_training_summary.json

Note on V1 comparability: V1's per-step file was written from `on_step_end`, i.e. one
step behind TRL's own metrics, so V1 `loss` / `grad_norm` / `entropy` are shifted by one
step and V1 has no `kl` (beta=0). The reward-function statistics (reward_mean, mixed,
all_correct, all_wrong, unique) were recorded in-step in both runs and are comparable.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
V2_DIR = ROOT / "outputs" / "grpo_v2_kl001"
V1_DIR = ROOT / "outputs" / "grpo_v1"
OUT_DIR = ROOT / "outputs" / "grpo_v2_analysis"

BUCKETS = [(1, 100), (101, 200), (201, 300), (301, 400), (401, 500), (501, 600), (601, 625)]

TRAIN_FIELDS = [
    "reward_mean",
    "mixed_group_ratio",
    "all_correct_ratio",
    "all_wrong_ratio",
    "zero_variance_group_ratio",
    "avg_correct_per_group",
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


def bucket_stats(rows: list[dict], low: int, high: int) -> dict:
    selected = [row for row in rows if low <= row.get("step", -1) <= high]
    out = {"steps": f"{low}-{high}", "n": len(selected)}
    for field in TRAIN_FIELDS:
        values = [row[field] for row in selected if row.get(field) is not None]
        out[field] = mean(values) if values else None
    return out


def val_table(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [
        {
            "step": row["step"],
            "exact_accuracy": row["exact_accuracy"],
            "format_accuracy": row["format_accuracy"],
            "parse_success_rate": row["parse_success_rate"],
        }
        for row in rows
    ]


def kl_trajectory(rows: list[dict]) -> list[dict]:
    return [
        {"step": row["step"], "kl": row["kl"], "loss": row.get("loss"), "grad_norm": row.get("grad_norm"), "reward_mean": row.get("reward_mean")}
        for row in rows
        if row.get("kl") is not None
    ]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    v2_train = load_jsonl(V2_DIR / "train_metrics.jsonl")
    v1_train = load_jsonl(V1_DIR / "grpo_v1_train_metrics.jsonl")

    summary = {
        "v2_steps": len(v2_train),
        "v1_steps": len(v1_train),
        "v2_train_buckets": [bucket_stats(v2_train, low, high) for low, high in BUCKETS],
        "v1_train_buckets": [bucket_stats(v1_train, low, high) for low, high in BUCKETS],
        "v2_val": val_table(V2_DIR / "val_metrics.json"),
        "v1_val": val_table(V1_DIR / "grpo_v1_val_metrics.json"),
        "v2_kl_trajectory": kl_trajectory(v2_train),
    }
    kl_values = [row["kl"] for row in v2_train if row.get("kl") is not None]
    summary["v2_kl_summary"] = {
        "n": len(kl_values),
        "first": kl_values[0] if kl_values else None,
        "last": kl_values[-1] if kl_values else None,
        "min": min(kl_values) if kl_values else None,
        "max": max(kl_values) if kl_values else None,
        "mean": mean(kl_values) if kl_values else None,
    }
    out = OUT_DIR / "grpo_v2_training_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"v2 steps={summary['v2_steps']} v1 steps={summary['v1_steps']}")
    header = f"{'bucket':<12}" + "".join(f"{f[:14]:>16}" for f in ("reward_mean", "mixed", "all_correct", "all_wrong", "kl"))
    print("\nV2 train dynamics")
    print(header)
    for bucket in summary["v2_train_buckets"]:
        print(
            f"{bucket['steps']:<12}"
            + f"{bucket['reward_mean'] or 0:>16.4f}"
            + f"{bucket['mixed_group_ratio'] or 0:>16.4f}"
            + f"{bucket['all_correct_ratio'] or 0:>16.4f}"
            + f"{bucket['all_wrong_ratio'] or 0:>16.4f}"
            + f"{bucket['kl'] or 0:>16.6f}"
        )
    print("\nV1 train dynamics (reference)")
    print(header)
    for bucket in summary["v1_train_buckets"]:
        print(
            f"{bucket['steps']:<12}"
            + f"{bucket['reward_mean'] or 0:>16.4f}"
            + f"{bucket['mixed_group_ratio'] or 0:>16.4f}"
            + f"{bucket['all_correct_ratio'] or 0:>16.4f}"
            + f"{bucket['all_wrong_ratio'] or 0:>16.4f}"
            + f"{0.0:>16.6f}"
        )
    print("\nV2 val")
    for row in summary["v2_val"]:
        print(f"  step {row['step']:>4}  exact={row['exact_accuracy']:.4f}  format={row['format_accuracy']:.4f}  parse={row['parse_success_rate']:.4f}")
    print("\nV2 KL summary:", json.dumps(summary["v2_kl_summary"], ensure_ascii=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
