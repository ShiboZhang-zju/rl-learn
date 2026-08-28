#!/usr/bin/env python3
"""Generate statement-first K&K v2 raw and answer-only SFT datasets."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kk_sft.data import build_answer_only_sft_example, write_jsonl  # noqa: E402
from kk_sft.logic import canonical_puzzle_key, generate_puzzle_v2  # noqa: E402


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
        puzzle["id"] = f"kk_v2_{split}_{len(rows):06d}"
        rows.append(puzzle)
        if len(rows) % 500 == 0:
            print(f"generated {split}: {len(rows)}/{count} attempts={attempts}", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--train-size", type=int, default=5000)
    parser.add_argument("--val-size", type=int, default=500)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--train-1k-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260829)
    args = parser.parse_args()

    raw_dir = args.output_dir / "raw_v2"
    processed_dir = args.output_dir / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    used_keys: set[str] = set()
    specs = [("train", args.train_size), ("val", args.val_size), ("test", args.test_size)]
    raw_splits: dict[str, list[dict]] = {}
    manifest = {
        "dataset_version": "kk-v2",
        "generator_version": "kk-v2-statement-first",
        "seed": args.seed,
        "people": ["Alice", "Bob", "Carol"],
        "generation_rule": "random statements first, exact solve, retain solution_count == 1",
        "splits": {},
    }
    for offset, (split, count) in enumerate(specs):
        rows = generate_split(split, count, args.seed + offset * 1_000_003, used_keys)
        raw_splits[split] = rows
        raw_path = raw_dir / f"{split}.jsonl"
        write_jsonl(raw_path, rows)
        processed_rows = [build_answer_only_sft_example(row, row["id"]) for row in rows]
        processed_path = processed_dir / f"v2_answer_only_{split}.jsonl"
        write_jsonl(processed_path, processed_rows)
        manifest["splits"][split] = {"count": count, "raw_file": str(raw_path), "processed_file": str(processed_path)}
        print(f"completed {split}: {count}", flush=True)

    train_1k = raw_splits["train"][: args.train_1k_size]
    write_jsonl(raw_dir / "train_1k.jsonl", train_1k)
    write_jsonl(
        processed_dir / "v2_answer_only_train_1k.jsonl",
        [build_answer_only_sft_example(row, row["id"]) for row in train_1k],
    )
    manifest["train_1k"] = {
        "count": len(train_1k),
        "selection": "first 1000 rows of deterministic v2 train split",
        "raw_file": str(raw_dir / "train_1k.jsonl"),
        "processed_file": str(processed_dir / "v2_answer_only_train_1k.jsonl"),
    }
    manifest["total_unique_puzzles"] = len(used_keys)
    (processed_dir / "v2_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"unique puzzles across v2 splits: {len(used_keys)}")


if __name__ == "__main__":
    main()
