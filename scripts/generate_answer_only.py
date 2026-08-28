#!/usr/bin/env python3
"""Create answer-only SFT datasets while preserving the original splits."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kk_sft.data import read_jsonl, write_jsonl  # noqa: E402


def answer_completion(row: dict) -> str:
    people = row["puzzle"]["people"]
    lines = ["<answer>"]
    lines.extend(f"{person}: {row['answer'][person]}" for person in people)
    lines.append("</answer>")
    return "\n".join(lines)


def convert(row: dict) -> dict:
    result = copy.deepcopy(row)
    completion = [{"role": "assistant", "content": answer_completion(row)}]
    result["completion"] = completion
    result["messages"] = result["prompt"] + completion
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    for split in ("train", "val", "test"):
        rows = read_jsonl(args.input_dir / f"sft_{split}.jsonl")
        output = args.output_dir / f"answer_only_{split}.jsonl"
        write_jsonl(output, (convert(row) for row in rows))
        print(f"{split}: {len(rows)} -> {output}")


if __name__ == "__main__":
    main()
