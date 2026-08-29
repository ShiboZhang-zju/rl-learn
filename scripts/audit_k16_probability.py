#!/usr/bin/env python3
"""Teacher-forced 8-way probability scoring for the K=8 vs K=16 intervention.

Reuses the H4 scoring code verbatim (same canonical completions, same completion-only
sequence log-prob, same 8-way logsumexp normalisation), so K8 and K16 numbers are
directly comparable with the historical H4 numbers.

Two modes:

  --dataset diag200      200-prompt diagnostic subset, scored under every
                         K8 checkpoint (100..600) and K16 checkpoint (100..600)
                         plus the shared Epoch4 starting point (step 0).
                         Used for the equal-step and equal-rollout-budget comparisons.

  --dataset k16_holdout  fresh K-intervention holdout (N=2000), scored under
                         Epoch4 / Epoch5 / K8 best / K16 best.
                         Used for the primary mechanism endpoints and the
                         stratified low-support analysis.

Writes to outputs/grpo_k16_analysis/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_h4_probability_landscape import (  # noqa: E402
    BASE_MODEL,
    landscape_from_scores,
    score_dataset,
)
from kk_sft.data import read_jsonl  # noqa: E402

OUT_DIR = ROOT / "outputs" / "grpo_k16_analysis"
DIAG_DATA = ROOT / "outputs" / "grpo_v3_analysis/diagnostic_200.jsonl"
K16_HOLDOUT = ROOT / "data/processed/grpo_k16_final_holdout.jsonl"

EPOCH4 = "outputs/sft_v2_5k_p800/checkpoint-1252"
EPOCH5 = "outputs/sft_v2_5k_p800/checkpoint-1565"
K8_STEPS = [100, 200, 300, 400, 500, 600]
K16_STEPS = [100, 200, 300, 400, 500, 600]


def pattern_of(answer: dict, people: list) -> str:
    return "".join("K" if answer[p] == "knight" else "N" for p in people)


def load_models(path: Path) -> list[tuple[int, str]]:
    """(step, adapter) pairs; step 0 means the shared Epoch4 starting point."""
    return [(0, EPOCH4)] + [(step, str(path / f"checkpoint-{step}")) for step in K8_STEPS]


@torch.inference_mode()
def score(rows: list[dict], checkpoints: list[tuple[int, str]], batch_size: int, device: str) -> dict:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    gold = [pattern_of(row["answer"], row["puzzle"]["people"]) for row in rows]

    out: dict[str, dict] = {}
    for label, adapter in checkpoints:
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16, trust_remote_code=True)
        model = PeftModel.from_pretrained(model, adapter)
        model.to(device)
        model.eval()
        scores, lengths = score_dataset(model, tokenizer, rows, batch_size, device)
        land = landscape_from_scores(scores, lengths, gold)
        out[str(label)] = {
            "gold_q": land["gold_q"].tolist(),
            "gold_rank": land["gold_rank"].astype(int).tolist(),
            "gold_margin": land["gold_margin"].tolist(),
            "top1_margin": land["top1_margin"].tolist(),
            "top1_pattern": land["top1_pattern"].tolist(),
            "normalized_entropy": land["normalized_entropy"].tolist(),
            "effective_support": land["effective_support"].tolist(),
        }
        del model
        torch.cuda.empty_cache()
        print(f"  scored {label} <- {adapter}", flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["diag200", "k16_holdout"], required=True)
    parser.add_argument("--k8-dir", type=Path, default=ROOT / "outputs" / "grpo_v2_kl001")
    parser.add_argument("--k16-dir", type=Path, default=ROOT / "outputs" / "grpo_k16_intervention")
    parser.add_argument("--k16-best", type=int, default=625, help="best K16 step chosen on V2 Val")
    parser.add_argument("--k8-best", type=int, default=600, help="historical K8 best step")
    parser.add_argument(
        "--extra-k16-steps",
        type=int,
        nargs="*",
        default=[600],
        help="additional K16 checkpoints to score, so a same-step K8-vs-K16 comparison is possible",
    )
    parser.add_argument("--extra-k8-steps", type=int, nargs="*", default=[], help="additional K8 checkpoints to score")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.dataset == "diag200":
        rows = read_jsonl(DIAG_DATA)
        checkpoints = [("e4_step0", EPOCH4)]
        checkpoints += [(f"k8_{step}", str(args.k8_dir / f"checkpoint-{step}")) for step in K8_STEPS]
        checkpoints += [(f"k16_{step}", str(args.k16_dir / f"checkpoint-{step}")) for step in K16_STEPS]
        result = score(rows, checkpoints, args.batch_size, args.device)
        payload = {
            "dataset": str(DIAG_DATA.relative_to(ROOT)),
            "n": len(rows),
            "ids": [row["id"] for row in rows],
            "gold": [pattern_of(row["answer"], row["puzzle"]["people"]) for row in rows],
            "models": result,
        }
        out_path = OUT_DIR / "diag200_probability.json"
    else:
        rows = read_jsonl(K16_HOLDOUT)
        checkpoints = [
            ("epoch4", EPOCH4),
            ("epoch5", EPOCH5),
            ("k8_best", str(args.k8_dir / f"checkpoint-{args.k8_best}")),
            ("k16_best", str(args.k16_dir / f"checkpoint-{args.k16_best}")),
        ]
        checkpoints += [(f"k8_{step}", str(args.k8_dir / f"checkpoint-{step}")) for step in args.extra_k8_steps]
        checkpoints += [(f"k16_{step}", str(args.k16_dir / f"checkpoint-{step}")) for step in args.extra_k16_steps]
        result = score(rows, checkpoints, args.batch_size, args.device)
        payload = {
            "dataset": str(K16_HOLDOUT.relative_to(ROOT)),
            "n": len(rows),
            "ids": [row["id"] for row in rows],
            "gold": [pattern_of(row["answer"], row["puzzle"]["people"]) for row in rows],
            "k8_best_step": args.k8_best,
            "k16_best_step": args.k16_best,
            "models": result,
        }
        out_path = OUT_DIR / "fresh_holdout_probability.json"

    out_path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")

    # quick console summary
    print(f"\n{'model':<12}{'top1':>8}{'mean_q':>9}{'p10_q':>9}{'q<.05':>8}{'top3':>8}{'n_ent':>8}")
    gold = np.array(payload["gold"])
    for name, data in payload["models"].items():
        q = np.array(data["gold_q"])
        rank = np.array(data["gold_rank"])
        top1 = np.array(data["top1_pattern"])
        print(
            f"{name:<12}{(top1 == gold).mean():>8.4f}{q.mean():>9.4f}{np.percentile(q, 10):>9.4f}"
            f"{(q < 0.05).mean():>8.4f}{(rank <= 3).mean():>8.4f}{np.mean(data['normalized_entropy']):>8.4f}"
        )


if __name__ == "__main__":
    main()
