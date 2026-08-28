#!/usr/bin/env python3
"""Audit dataset balance, structural difficulty proxies, and parser samples."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kk_sft.data import read_jsonl  # noqa: E402
from kk_sft.evaluation import parse_answer  # noqa: E402
from kk_sft.logic import solve_puzzle  # noqa: E402

LABELS = ["KKK", "KKN", "KNK", "KNN", "NKK", "NKN", "NNK", "NNN"]
OPS = ("person_is", "same", "different", "not", "and", "or")


def pattern(row: dict) -> str:
    return "".join("K" if row["answer"][person] == "knight" else "N" for person in row["puzzle"]["people"])


def walk_expr(expr: dict[str, Any], ops: Counter[str], depths: list[int], depth: int = 1) -> int:
    op = expr["op"]
    ops[op] += 1
    depths.append(depth)
    if op == "not":
        return walk_expr(expr["expr"], ops, depths, depth + 1)
    if op in ("and", "or"):
        left = walk_expr(expr["left"], ops, depths, depth + 1)
        right = walk_expr(expr["right"], ops, depths, depth + 1)
        return max(left, right)
    return depth


def row_features(row: dict) -> dict[str, Any]:
    ops: Counter[str] = Counter()
    depths: list[int] = []
    top_ops = []
    statement_chars = 0
    speakers = []
    for statement in row["puzzle"]["statements"]:
        top_ops.append(statement["expr"]["op"])
        speakers.append(statement["speaker"])
        statement_chars += len(statement["text"])
        walk_expr(statement["expr"], ops, depths)
    return {
        "pattern": pattern(row),
        "top_ops": "+".join(top_ops),
        "op_signature": "+".join(f"{op}:{ops[op]}" for op in OPS if ops[op]),
        "statement_chars": statement_chars,
        "expression_nodes": sum(ops.values()),
        "expression_depth": max(depths),
        "person_is_count": ops["person_is"],
        "same_count": ops["same"],
        "different_count": ops["different"],
        "not_count": ops["not"],
        "and_count": ops["and"],
        "or_count": ops["or"],
        "speaker_order": "+".join(speakers),
    }


def crosstab(rows: list[dict], feature: str) -> dict[str, dict[str, int]]:
    table = {label: Counter() for label in LABELS}
    for row in rows:
        table[row["pattern"]][row[feature]] += 1
    return {label: dict(sorted(table[label].items())) for label in LABELS}


def numeric_summary(rows: list[dict], feature: str) -> dict[str, float]:
    values = [float(row[feature]) for row in rows]
    return {"mean": sum(values) / len(values), "min": min(values), "max": max(values)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--parser-file", type=Path, default=Path("outputs/answer_only_3ep_test_eval.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("outputs/dataset_feature_audit.json"))
    parser.add_argument("--parser-sample-size", type=int, default=50)
    args = parser.parse_args()

    split_rows: dict[str, list[dict]] = {}
    split_features: dict[str, list[dict]] = {}
    for split in ("train", "val", "test"):
        rows = read_jsonl(args.data_dir / f"sft_{split}.jsonl")
        split_rows[split] = rows
        split_features[split] = [row_features(row) for row in rows]

    all_rows = [row for rows in split_rows.values() for row in rows]
    all_keys = [json.dumps(row["puzzle"], sort_keys=True, separators=(",", ":")) for row in all_rows]
    duplicate_counts = Counter(all_keys)
    duplicate_examples = [count for count in duplicate_counts.values() if count > 1]
    split_result: dict[str, Any] = {}
    for split, features in split_features.items():
        split_result[split] = {
            "count": len(features),
            "label_distribution": dict(sorted(Counter(row["pattern"] for row in features).items())),
            "solution_count_distribution": dict(sorted(Counter(row["puzzle"]["solution_count"] for row in split_rows[split]).items())),
            "metadata_max_depth": dict(sorted(Counter(row["puzzle"]["metadata"].get("max_depth") for row in split_rows[split]).items())),
            "numeric_features": {feature: numeric_summary(features, feature) for feature in ("statement_chars", "expression_nodes", "expression_depth", "person_is_count", "same_count", "different_count", "not_count", "and_count", "or_count")},
            "answer_by_top_ops": crosstab(features, "top_ops"),
            "answer_by_op_signature": crosstab(features, "op_signature"),
            "answer_by_speaker_order": crosstab(features, "speaker_order"),
        }

    parser_rows = read_jsonl(args.parser_file)
    puzzle_by_id = {row["id"]: row["puzzle"] for row in all_rows}
    rng = random.Random(20260828)
    sample = rng.sample(parser_rows, min(args.parser_sample_size, len(parser_rows)))
    parser_checks = []
    for row in sample:
        puzzle = puzzle_by_id[row["id"]]
        parsed = parse_answer(row["prediction"], puzzle["people"])
        parser_checks.append({
            "id": row["id"],
            "prediction": row["prediction"],
            "parsed_answer": parsed.parsed,
            "ground_truth": row["ground_truth"],
            "format_valid": parsed.format_valid,
            "correct": parsed.parsed == row["ground_truth"],
            "record_correct": row["correct"],
            "consistent_with_record": parsed.parsed == row["parsed_answer"] and (parsed.parsed == row["ground_truth"]) == row["correct"],
        })

    result = {
        "splits": split_result,
        "duplicate_audit": {
            "total_rows": len(all_rows),
            "unique_puzzle_keys": len(duplicate_counts),
            "duplicate_key_count": len(duplicate_examples),
            "duplicate_extra_rows": sum(count - 1 for count in duplicate_examples),
        },
        "parser_sample_audit": {
            "sample_size": len(parser_checks),
            "format_valid_count": sum(row["format_valid"] for row in parser_checks),
            "correct_count": sum(row["correct"] for row in parser_checks),
            "record_consistency_count": sum(row["consistent_with_record"] for row in parser_checks),
            "records": parser_checks,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "duplicate_audit": result["duplicate_audit"],
        "parser_sample_audit": {key: value for key, value in result["parser_sample_audit"].items() if key != "records"},
        "split_labels": {split: value["label_distribution"] for split, value in split_result.items()},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
