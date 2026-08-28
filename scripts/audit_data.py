#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kk_sft.audit import audit_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/audit_report.json"))
    args = parser.parse_args()
    report = audit_dataset(args.raw_dir, args.processed_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    bad = []
    for split, stats in report["splits"].items():
        if stats["bad_solution_count"] or stats["duplicate_count"]:
            bad.append(split)
    if any(value for value in report["cross_split_overlap"].values()):
        bad.append("cross_split_overlap")
    if bad:
        raise SystemExit("Audit failed: " + ", ".join(bad))


if __name__ == "__main__":
    main()

