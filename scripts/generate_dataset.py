#!/usr/bin/env python3
"""Generate raw and TRL-ready Knights & Knaves datasets."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kk_sft.data import build_sft_example, write_jsonl  # noqa: E402
from kk_sft.logic import canonical_puzzle_key, generate_puzzle  # noqa: E402


def generate_split(split: str, count: int, seed: int, used_keys: set[str]) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    while len(rows) < count:
        puzzle_seed = rng.randrange(0, 2**63)
        puzzle = generate_puzzle(puzzle_seed)
        key = canonical_puzzle_key(puzzle)
        if key in used_keys:
            continue
        used_keys.add(key)
        puzzle["id"] = f"kk_{split}_{len(rows):06d}"
        rows.append(puzzle)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--train-size", type=int, default=1000)
    parser.add_argument("--val-size", type=int, default=200)
    parser.add_argument("--test-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Allow replacing generated JSONL files")
    args = parser.parse_args()

    raw_dir = args.output_dir / "raw"
    processed_dir = args.output_dir / "processed"
    paths = [
        raw_dir / "train.jsonl",
        raw_dir / "val.jsonl",
        raw_dir / "test.jsonl",
        processed_dir / "sft_train.jsonl",
        processed_dir / "sft_val.jsonl",
        processed_dir / "sft_test.jsonl",
    ]
    if not args.force:
        existing = [str(path) for path in paths if path.exists()]
        if existing:
            raise SystemExit(
                "Files already exist; use --force only when replacement is intentional: " + ", ".join(existing)
            )

    used_keys: set[str] = set()
    split_specs = [("train", args.train_size), ("val", args.val_size), ("test", args.test_size)]
    manifest = {"dataset_version": "kk-v1", "seed": args.seed, "splits": {}}
    for offset, (split, count) in enumerate(split_specs):
        raw_rows = generate_split(split, count, args.seed + offset * 1_000_003, used_keys)
        processed_rows = [build_sft_example(row, row["id"]) for row in raw_rows]
        write_jsonl(raw_dir / f"{split}.jsonl", raw_rows)
        write_jsonl(processed_dir / f"sft_{split}.jsonl", processed_rows)
        manifest["splits"][split] = {"count": count, "file": f"sft_{split}.jsonl"}
        print(f"generated {split}: {count}")
    processed_dir.mkdir(parents=True, exist_ok=True)
    (processed_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"unique puzzles across splits: {len(used_keys)}")


if __name__ == "__main__":
    main()

