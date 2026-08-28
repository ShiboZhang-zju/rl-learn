#!/usr/bin/env python3
"""Generate GRPO-v1 train (5000) and final holdout (2000) datasets.

Uses the same statement-first generator (generate_puzzle_v2) as kk-v2.
Both splits are de-duplicated against every existing puzzle key:
  - legacy v1 raw train/val/test (data/raw/*.jsonl, if present)
  - kk-v2 raw train/val/test (data/raw_v2/*.jsonl)
  - already-generated GRPO-v1 rows in this run

Processed files contain only prompt + ground truth (no SFT completion).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kk_sft.data import format_answer_only_prompt, read_jsonl, write_jsonl  # noqa: E402
from kk_sft.logic import canonical_puzzle_key, generate_puzzle_v2  # noqa: E402


def puzzle_of(row: dict) -> dict:
    return row.get("puzzle", row)


def collect_existing_keys(*paths: Path) -> set[str]:
    keys: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        for row in read_jsonl(path):
            keys.add(canonical_puzzle_key(puzzle_of(row)))
    return keys


def generate_split(split: str, count: int, seed: int, used_keys: set[str]) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []
    attempts = 0
    while len(rows) < count:
        attempts += 1
        puzzle_seed = rng.randrange(0, 2**63)
        puzzle = generate_puzzle_v2(puzzle_seed)
        key = canonical_puzzle_key(puzzle)
        if key in used_keys:
            continue
        used_keys.add(key)
        puzzle["id"] = f"kk_grpo_v1_{split}_{len(rows):06d}"
        rows.append(puzzle)
        if len(rows) % 500 == 0:
            print(f"generated {split}: {len(rows)}/{count} attempts={attempts}", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--train-size", type=int, default=5000)
    parser.add_argument("--holdout-size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()

    raw_dir = args.output_dir / "raw_grpo_v1"
    processed_dir = args.output_dir / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    legacy_files = [args.output_dir / "raw" / f"{name}.jsonl" for name in ("train", "val", "test")]
    v2_files = [args.output_dir / "raw_v2" / f"{name}.jsonl" for name in ("train", "val", "test")]
    universe = collect_existing_keys(*legacy_files, *v2_files)
    print(f"existing puzzle keys loaded: {len(universe)}")

    used_keys: set[str] = set(universe)

    manifest = {
        "dataset_version": "kk-v2-statement-first",
        "generator_version": "kk-v2-statement-first",
        "seed": args.seed,
        "people": ["Alice", "Bob", "Carol"],
        "generation_rule": "statement-first, exact solve, retain solution_count == 1; dedup against legacy+v2+grpo-v1",
        "dedup_universe": {
            "legacy_raw": [str(path) for path in legacy_files],
            "v2_raw": [str(path) for path in v2_files],
            "universe_key_count": len(universe),
        },
        "splits": {},
    }

    # 1) GRPO train
    train_rows = generate_split("train", args.train_size, args.seed, used_keys)
    write_jsonl(raw_dir / "train.jsonl", train_rows)
    processed_train = [
        {
            "id": row["id"],
            "prompt": [
                {"role": "system", "content": "You solve Knights and Knaves logic puzzles. Output only one answer block."},
                {"role": "user", "content": format_answer_only_prompt(row)},
            ],
            "answer": row["answer"],
            "puzzle": row,
        }
        for row in train_rows
    ]
    write_jsonl(processed_dir / "grpo_v1_train.jsonl", processed_train)
    manifest["splits"]["train"] = {
        "count": args.train_size,
        "raw_file": str(raw_dir / "train.jsonl"),
        "processed_file": str(processed_dir / "grpo_v1_train.jsonl"),
        "seed": args.seed,
    }

    # 2) Final holdout (train on nothing; only opened once after best checkpoint chosen)
    holdout_rows = generate_split("final_holdout", args.holdout_size, args.seed + 1_000_003, used_keys)
    write_jsonl(raw_dir / "final_holdout.jsonl", holdout_rows)
    processed_holdout = [
        {
            "id": row["id"],
            "prompt": [
                {"role": "system", "content": "You solve Knights and Knaves logic puzzles. Output only one answer block."},
                {"role": "user", "content": format_answer_only_prompt(row)},
            ],
            "answer": row["answer"],
            "puzzle": row,
        }
        for row in holdout_rows
    ]
    write_jsonl(processed_dir / "grpo_v1_final_holdout.jsonl", processed_holdout)
    manifest["splits"]["final_holdout"] = {
        "count": args.holdout_size,
        "raw_file": str(raw_dir / "final_holdout.jsonl"),
        "processed_file": str(processed_dir / "grpo_v1_final_holdout.jsonl"),
        "seed": args.seed + 1_000_003,
    }

    manifest["total_unique_puzzles"] = len(used_keys)
    (processed_dir / "grpo_v1_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"total unique puzzles in universe after generation: {len(used_keys)}")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
