#!/usr/bin/env python3
"""Create one uniquely solvable puzzle for each 3-person K/N answer pattern."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kk_sft.data import build_sft_example, write_jsonl  # noqa: E402
from kk_sft.logic import KNIGHT, canonical_puzzle_key, generate_puzzle  # noqa: E402


def pattern(answer: dict[str, str], people: list[str]) -> str:
    return "".join("K" if answer[person] == KNIGHT else "N" for person in people)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    people = ["Alice", "Bob", "Carol"]
    raw_path = args.output_dir / "raw" / "overfit_balanced.jsonl"
    processed_path = args.output_dir / "processed" / "sft_overfit_balanced.jsonl"
    if not args.force and (raw_path.exists() or processed_path.exists()):
        raise SystemExit("Balanced dataset exists; use --force only when replacement is intentional")

    targets = ["KKK", "KKN", "KNK", "KNN", "NKK", "NKN", "NNK", "NNN"]
    rng = random.Random(args.seed)
    rows = []
    used_keys: set[str] = set()
    for target in targets:
        while True:
            puzzle = generate_puzzle(rng.randrange(0, 2**63), people=people)
            if pattern(puzzle["answer"], people) == target:
                key = canonical_puzzle_key(puzzle)
                if key not in used_keys:
                    used_keys.add(key)
                    puzzle["id"] = f"kk_balanced_{len(rows):03d}_{target}"
                    puzzle["metadata"]["balanced_target_pattern"] = target
                    rows.append(puzzle)
                    break

    processed = [build_sft_example(row, row["id"]) for row in rows]
    write_jsonl(raw_path, rows)
    write_jsonl(processed_path, processed)
    print(f"wrote {raw_path}")
    print(f"wrote {processed_path}")
    print("patterns:", [pattern(row["answer"], people) for row in rows])


if __name__ == "__main__":
    main()

