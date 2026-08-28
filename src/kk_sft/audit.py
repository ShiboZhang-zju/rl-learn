"""Data quality checks used before any model training."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .logic import canonical_puzzle_key, solve_puzzle


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _raw_signature(row: dict[str, Any]) -> str:
    return canonical_puzzle_key(row)


def _char_stats(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"min": 0, "max": 0, "mean": 0.0}
    return {"min": min(values), "max": max(values), "mean": sum(values) / len(values)}


def audit_raw_splits(raw_dir: str | Path) -> dict[str, Any]:
    raw_dir = Path(raw_dir)
    split_rows: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "val", "test"):
        path = raw_dir / f"{split}.jsonl"
        if path.exists():
            split_rows[split] = _read_jsonl(path)
    all_signatures: dict[str, set[str]] = {}
    report: dict[str, Any] = {"raw_dir": str(raw_dir), "splits": {}, "cross_split_overlap": {}}
    for split, rows in split_rows.items():
        signatures = [_raw_signature(row) for row in rows]
        all_signatures[split] = set(signatures)
        solution_counts = [len(solve_puzzle(row)) for row in rows]
        answer_keys = ["|".join(row["answer"][person] for person in row["people"]) for row in rows]
        report["splits"][split] = {
            "count": len(rows),
            "unique_ids": len({row.get("id") for row in rows}),
            "unique_puzzles": len(set(signatures)),
            "duplicate_count": len(signatures) - len(set(signatures)),
            "unique_solution_count": sum(count == 1 for count in solution_counts),
            "bad_solution_count": sum(count != 1 for count in solution_counts),
            "answer_distribution": dict(Counter(answer_keys)),
            "statement_chars": _char_stats(
                [sum(len(statement["text"]) for statement in row["statements"]) for row in rows]
            ),
        }
    splits = list(all_signatures)
    for index, left in enumerate(splits):
        for right in splits[index + 1 :]:
            report["cross_split_overlap"][f"{left}__{right}"] = len(all_signatures[left] & all_signatures[right])
    return report


def audit_processed_splits(processed_dir: str | Path) -> dict[str, Any]:
    processed_dir = Path(processed_dir)
    report: dict[str, Any] = {"processed_dir": str(processed_dir), "splits": {}}
    for split in ("train", "val", "test"):
        path = processed_dir / f"sft_{split}.jsonl"
        if not path.exists():
            continue
        rows = _read_jsonl(path)
        prompt_chars = [len(json.dumps(row.get("prompt", ""), ensure_ascii=False)) for row in rows]
        completion_chars = [len(json.dumps(row.get("completion", ""), ensure_ascii=False)) for row in rows]
        report["splits"][split] = {
            "count": len(rows),
            "unique_ids": len({row.get("id") for row in rows}),
            "prompt_chars": _char_stats(prompt_chars),
            "completion_chars": _char_stats(completion_chars),
        }
    return report


def audit_dataset(raw_dir: str | Path, processed_dir: str | Path | None = None) -> dict[str, Any]:
    report = audit_raw_splits(raw_dir)
    if processed_dir is not None:
        report["processed"] = audit_processed_splits(processed_dir)
    return report

