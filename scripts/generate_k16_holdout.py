#!/usr/bin/env python3
"""Generate the K-intervention fresh final holdout (2000, seed 20260903).

Every earlier holdout has already been opened for analysis, so the K=8 vs K=16 causal
comparison needs its own untouched set. Same statement-first generator
(generate_puzzle_v2) as every earlier split, de-duplicated against:

  - legacy v1 raw train/val/test       (data/raw/*.jsonl, if present)
  - kk-v2 raw train/val/test           (data/raw_v2/*.jsonl)
  - GRPO-V1 train + final holdout      (data/raw_grpo_v1/*.jsonl)
  - GRPO-V2 final holdout              (data/raw_grpo_v2/final_holdout.jsonl)
  - GRPO-V3 final holdout              (data/raw_grpo_v3/final_holdout.jsonl)

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

SYSTEM_PROMPT = "You solve Knights and Knaves logic puzzles. Output only one answer block."


def puzzle_of(row: dict) -> dict:
    return row.get("puzzle", row)


def collect_existing_keys(*paths: Path) -> dict[str, int]:
    keys: dict[str, int] = {}
    for path in paths:
        if not path.exists():
            continue
        count = 0
        for row in read_jsonl(path):
            key = canonical_puzzle_key(puzzle_of(row))
            keys[key] = keys.get(key, 0) + 1
            count += 1
        print(f"loaded {path}: {count} rows, {len(keys)} unique keys so far", flush=True)
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
        puzzle["id"] = f"kk_k16_{split}_{len(rows):06d}"
        rows.append(puzzle)
        if len(rows) % 500 == 0:
            print(f"generated {split}: {len(rows)}/{count} attempts={attempts}", flush=True)
    print(f"generated {split}: {len(rows)}/{count} attempts={attempts}", flush=True)
    return rows


def to_processed(rows: list[dict]) -> list[dict]:
    return [
        {
            "id": row["id"],
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": format_answer_only_prompt(row)},
            ],
            "answer": row["answer"],
            "puzzle": row,
        }
        for row in rows
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--holdout-size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260903)
    args = parser.parse_args()

    raw_dir = args.output_dir / "raw_grpo_k16"
    processed_dir = args.output_dir / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    sources = {
        "legacy_raw": [args.output_dir / "raw" / f"{name}.jsonl" for name in ("train", "val", "test")],
        "v2_raw": [args.output_dir / "raw_v2" / f"{name}.jsonl" for name in ("train", "val", "test")],
        "grpo_v1_raw": [args.output_dir / "raw_grpo_v1" / f"{name}.jsonl" for name in ("train", "final_holdout")],
        "grpo_v2_raw": [args.output_dir / "raw_grpo_v2" / "final_holdout.jsonl"],
        "grpo_v3_raw": [args.output_dir / "raw_grpo_v3" / "final_holdout.jsonl"],
    }
    all_paths = [path for group in sources.values() for path in group]
    existing = collect_existing_keys(*all_paths)
    used_keys: set[str] = set(existing)
    print(f"existing puzzle keys loaded: {len(used_keys)}", flush=True)

    holdout_rows = generate_split("final_holdout", args.holdout_size, args.seed, used_keys)
    raw_path = raw_dir / "final_holdout.jsonl"
    write_jsonl(raw_path, holdout_rows)
    processed_path = processed_dir / "grpo_k16_final_holdout.jsonl"
    write_jsonl(processed_path, to_processed(holdout_rows))

    def count_overlap(path: Path) -> int:
        if not path.exists():
            return -1
        other = {canonical_puzzle_key(puzzle_of(row)) for row in read_jsonl(path)}
        return len({canonical_puzzle_key(row) for row in holdout_rows} & other)

    overlap = {name: {str(path): count_overlap(path) for path in paths} for name, paths in sources.items()}
    internal_dupes = args.holdout_size - len({canonical_puzzle_key(row) for row in holdout_rows})

    manifest = {
        "dataset_version": "kk-v2-statement-first",
        "generator_version": "kk-v2-statement-first",
        "seed": args.seed,
        "people": ["Alice", "Bob", "Carol"],
        "generation_rule": "statement-first, exact solve, retain solution_count == 1; dedup against legacy+v2+grpo-v1+grpo-v2+grpo-v3+self",
        "dedup_universe": {
            "sources": {name: [str(path) for path in paths] for name, paths in sources.items()},
            "universe_key_count": len(existing),
        },
        "splits": {
            "final_holdout": {
                "count": len(holdout_rows),
                "raw_file": str(raw_path),
                "processed_file": str(processed_path),
                "seed": args.seed,
            }
        },
        "overlap": overlap,
        "internal_duplicates": internal_dupes,
        "total_unique_puzzles": len(used_keys),
    }
    (processed_dir / "grpo_k16_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
