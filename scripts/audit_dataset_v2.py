#!/usr/bin/env python3
"""Audit kk-v2 integrity, labels, structural features, and answer-only examples."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kk_sft.data import read_jsonl  # noqa: E402
from kk_sft.logic import canonical_puzzle_key, solve_puzzle  # noqa: E402

LABELS = ["KKK", "KKN", "KNK", "KNN", "NKK", "NKN", "NNK", "NNN"]
FEATURES = ("same_count", "different_count", "and_count", "or_count", "not_count")


def puzzle_of(row: dict) -> dict:
    return row.get("puzzle", row)


def label(row: dict) -> str:
    puzzle = puzzle_of(row)
    return "".join("K" if row["answer"][person] == "knight" else "N" for person in puzzle["people"])


def expr_stats(expr: dict, counts: Counter, depths: list[int], depth: int = 1) -> None:
    counts[expr["op"]] += 1
    depths.append(depth)
    if expr["op"] == "not":
        expr_stats(expr["expr"], counts, depths, depth + 1)
    elif expr["op"] in ("and", "or"):
        expr_stats(expr["left"], counts, depths, depth + 1)
        expr_stats(expr["right"], counts, depths, depth + 1)


def features(row: dict) -> dict:
    counts = Counter()
    depths: list[int] = []
    top_ops: list[str] = []
    speakers: list[str] = []
    chars = 0
    for statement in puzzle_of(row)["statements"]:
        top_ops.append(statement["expr"]["op"])
        speakers.append(statement["speaker"])
        chars += len(statement["text"])
        expr_stats(statement["expr"], counts, depths)
    return {
        "label": label(row),
        "top_ops": "+".join(top_ops),
        "speaker_order": "+".join(speakers),
        "statement_chars": chars,
        "expression_nodes": sum(counts.values()),
        "expression_depth": max(depths),
        **{f"{op}_count": counts[op] for op in ("same", "different", "and", "or", "not")},
    }


def contingency(rows: list[dict], feature: str) -> dict[str, dict[str, int]]:
    table = {label_name: Counter() for label_name in LABELS}
    for row in rows:
        table[row["label"]][row[feature]] += 1
    return {label_name: dict(sorted(table[label_name].items())) for label_name in LABELS}


def mutual_information(table: dict[str, dict[str, int]]) -> float:
    total = sum(sum(values.values()) for values in table.values())
    x_counts = Counter()
    y_counts = Counter()
    for y, values in table.items():
        y_counts[y] += sum(values.values())
        for x, value in values.items():
            x_counts[x] += value
    result = 0.0
    for y, values in table.items():
        for x, value in values.items():
            if value:
                result += value / total * math.log2(value * total / (y_counts[y] * x_counts[x]))
    return result


def audit_split(raw_rows: list[dict], processed_rows: list[dict]) -> dict:
    feats = [features(row) for row in raw_rows]
    prompt_lengths = [len(row["prompt"][1]["content"]) for row in processed_rows]
    answer_lengths = [len(row["completion"][0]["content"]) for row in processed_rows]
    solution_counts = [len(solve_puzzle(puzzle_of(row))) for row in raw_rows]
    return {
        "count": len(raw_rows),
        "unique_ids": len({row["id"] for row in raw_rows}),
        "label_distribution": dict(sorted(Counter(row["label"] for row in feats).items())),
        "solution_count_distribution": dict(sorted(Counter(solution_counts).items())),
        "unique_solution_rate": sum(value == 1 for value in solution_counts) / len(solution_counts),
        "prompt_chars": {"mean": mean(prompt_lengths), "min": min(prompt_lengths), "max": max(prompt_lengths)},
        "answer_chars": {"mean": mean(answer_lengths), "min": min(answer_lengths), "max": max(answer_lengths)},
        "difficulty": {
            feature: {"mean": mean(row[feature] for row in feats), "min": min(row[feature] for row in feats), "max": max(row[feature] for row in feats)}
            for feature in ("statement_chars", "expression_nodes", "expression_depth")
        },
        "feature_means_by_label": {
            label_name: {
                feature: mean(row[feature] for row in feats if row["label"] == label_name)
                for feature in FEATURES
            }
            for label_name in LABELS
        },
        "shortcut_audit": {
            feature: {"mutual_information_bits": mutual_information(contingency(feats, feature)), "table": contingency(feats, feature)}
            for feature in FEATURES
        },
        "top_op_label_mutual_information_bits": mutual_information(contingency(feats, "top_ops")),
        "speaker_order_label_mutual_information_bits": mutual_information(contingency(feats, "speaker_order")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw_v2"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("outputs/dataset_v2_feature_audit.json"))
    parser.add_argument("--report", type=Path, default=Path("outputs/dataset_v2_audit_report.md"))
    parser.add_argument("--sample-size", type=int, default=30)
    args = parser.parse_args()

    splits: dict[str, dict] = {}
    all_keys: list[str] = []
    for split in ("train", "val", "test"):
        raw = read_jsonl(args.raw_dir / f"{split}.jsonl")
        processed = read_jsonl(args.processed_dir / f"v2_answer_only_{split}.jsonl")
        if len(raw) != len(processed):
            raise ValueError(f"raw/processed count mismatch for {split}")
        all_keys.extend(canonical_puzzle_key(puzzle_of(row)) for row in raw)
        splits[split] = audit_split(raw, processed)

    key_counts = Counter(all_keys)
    duplicate_values = [value for value in key_counts.values() if value > 1]
    train_rows = read_jsonl(args.raw_dir / "train.jsonl")
    processed_train = read_jsonl(args.processed_dir / "v2_answer_only_train.jsonl")
    rng = random.Random(20260829)
    sample_indices = rng.sample(range(len(train_rows)), min(args.sample_size, len(train_rows)))
    manual_samples = []
    for index in sample_indices:
        raw = train_rows[index]
        processed = processed_train[index]
        manual_samples.append({
            "id": raw["id"],
            "statements": puzzle_of(raw)["statements"],
            "solver_answer": raw["answer"],
            "sft_prompt": processed["prompt"],
            "sft_completion": processed["completion"],
            "solver_matches_record": solve_puzzle(puzzle_of(raw))[0] == raw["answer"],
        })

    result = {
        "dataset_version": "kk-v2",
        "generator_version": "kk-v2-statement-first",
        "splits": splits,
        "duplicate_audit": {
            "total_rows": len(all_keys),
            "unique_puzzle_keys": len(key_counts),
            "duplicate_key_count": len(duplicate_values),
            "duplicate_extra_rows": sum(value - 1 for value in duplicate_values),
        },
        "manual_sample_audit": {"sample_size": len(manual_samples), "samples": manual_samples},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# kk-v2 数据审计报告",
        "",
        "## 完整性",
        "",
        f"- Generator：`kk-v2-statement-first`；seed：`20260829`。",
        f"- 总样本：`{len(all_keys)}`；唯一 puzzle：`{len(key_counts)}`。",
        f"- duplicate puzzle：`{len(duplicate_values)}`，重复多出的行数：`{sum(value - 1 for value in duplicate_values)}`。",
        "- 生成方式：先随机生成 statements，再调用 exact solver，仅保留唯一解。",
        "",
        "## Split 与标签分布",
        "",
        "| Split | Count | KKK | KKN | KNK | KNN | NKK | NKN | NNK | NNN | Unique solution rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, audit in splits.items():
        labels = audit["label_distribution"]
        lines.append("| " + " | ".join([split, str(audit["count"])] + [str(labels.get(name, 0)) for name in LABELS] + [f"{audit['unique_solution_rate']:.3f}"]) + " |")
    lines.extend(["", "## 难度与长度", "", "| Split | Prompt chars mean | Answer chars mean | Expr nodes mean | Expr depth mean |", "|---|---:|---:|---:|---:|"])
    for split, audit in splits.items():
        lines.append(f"| {split} | {audit['prompt_chars']['mean']:.1f} | {audit['answer_chars']['mean']:.1f} | {audit['difficulty']['expression_nodes']['mean']:.2f} | {audit['difficulty']['expression_depth']['mean']:.2f} |")
    lines.extend(["", "## Shortcut audit", "", "互信息越低表示该结构特征对答案标签的直接关联越弱；它不是独立性证明。", "", "| Split | same | different | and | or | not | top-op signature |", "|---|---:|---:|---:|---:|---:|---:|"])
    for split, audit in splits.items():
        values = [audit["shortcut_audit"][feature]["mutual_information_bits"] for feature in FEATURES]
        lines.append(f"| {split} | " + " | ".join(f"{value:.4f}" for value in values) + f" | {audit['top_op_label_mutual_information_bits']:.4f} |")
    lines.extend(["", "## 人工抽查", "", f"固定随机抽查 `{len(manual_samples)}` 条 train 样本；每条同时保存 raw statements、solver answer、SFT prompt、SFT completion，并检查 solver answer 与记录一致。", "", "明细保存在本报告对应的 JSON 文件 `dataset_v2_feature_audit.json`。", ""])
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"duplicate_audit": result["duplicate_audit"], "split_labels": {key: value["label_distribution"] for key, value in splits.items()}, "report": str(args.report)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
