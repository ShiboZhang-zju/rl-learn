#!/usr/bin/env python3
"""H5 zero-training audit: finite-K on-policy sampling feedback.

Hypothesis: GRPO's bidirectional polarization comes from finite K=8 on-policy
sampling. Prompts whose gold answer already has high probability get sampled
correctly often and keep receiving positive reinforcement; prompts with low gold
probability frequently miss gold in all 8 rollouts and therefore receive no direct
positive reinforcement, drifting into stable-wrong.

    P(miss) = (1 - p)^K,  K = 8
    p=0.50 -> 0.4%   p=0.30 -> 5.8%   p=0.20 -> 16.8%
    p=0.10 -> 43.0%  p=0.05 -> 66.3%

Probability definition caveat (inherited from H4): `gold_q` is the model's
distribution re-normalised over the 8 canonical legal answers, NOT the true
P(generate the exact correct completion) in the full generation space. Therefore
`1 - (1 - gold_q)^8` is only an **8-way implied hit probability**. Real hit/miss
always comes from the saved rollouts (`correct_count`), never from theory.

This is a retrospective/path-dependence audit, NOT a causal experiment.

Writes to outputs/grpo_h5_sampling_feedback/.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_h4_probability_landscape import (  # noqa: E402
    BASE_MODEL,
    OUT_DIR as H4_OUT,
    landscape_from_scores,
    score_dataset,
)
from kk_sft.data import read_jsonl  # noqa: E402

OUT_DIR = ROOT / "outputs" / "grpo_h5_sampling_feedback"
DIAG_DATA = ROOT / "outputs" / "grpo_v3_analysis/diagnostic_200.jsonl"
ROLLOUT_E4 = ROOT / "outputs/grpo_v1_analysis/rollout_A_final200.jsonl"
ROLLOUT_V2 = ROOT / "outputs/grpo_v2_analysis/rollout_D_final200.jsonl"
PROBE = ROOT / "outputs/grpo_v2_kl001/probe_rollouts.json"

INIT_MODEL = ("epoch4", "outputs/sft_v2_5k_p800/checkpoint-1252")
FINAL_MODEL = ("v2", "outputs/grpo_v2_kl001/checkpoint-600")
CONTROL_MODEL = "epoch5"

BINS = [(0.0, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.40), (0.40, 0.60), (0.60, 0.80), (0.80, 1.01)]
MIN_BIN_N = 30
K = 8
BOOTSTRAP = 10000
SEED = 20260831
TRAJECTORY_STEPS = [0, 100, 200, 300, 400, 500, 600]


def bin_label(low: float, high: float) -> str:
    return f"[{low:.2f},{high:.2f})" if high <= 1.0 else f"[{low:.2f},1.00]"


def assign_bins(values: np.ndarray) -> tuple[list[str], list[tuple[float, float]]]:
    edges = list(BINS)
    # merge from the bottom until every bin has enough samples
    counts = [int(((values >= lo) & (values < hi)).sum()) for lo, hi in edges]
    merges = []
    while len(edges) > 1 and counts[0] < MIN_BIN_N:
        lo0, hi0 = edges[0]
        lo1, hi1 = edges[1]
        edges[0] = (lo0, hi1)
        edges.pop(1)
        counts = [int(((values >= lo) & (values < hi)).sum()) for lo, hi in edges]
        merges.append(f"merged {bin_label(lo0, hi0)} into {bin_label(lo0, hi1)} (N<{MIN_BIN_N})")
    labels = [bin_label(lo, hi) for lo, hi in edges]
    return labels, edges, merges


def pattern_of(answer: dict[str, str], people: list[str]) -> str:
    return "".join("K" if answer[person] == "knight" else "N" for person in people)


def load_h4() -> dict:
    """Per-sample arrays for the N=2000 fresh holdout, straight from the H4 audit."""
    ids, gold = [], []
    data: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for line in (H4_OUT / "sample_probability_landscape.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        ids.append(record["id"])
        gold.append(record["gold_pattern"])
        for model, values in record["models"].items():
            for key in ("gold_q", "gold_rank", "gold_margin", "normalized_entropy", "effective_support", "top1_margin"):
                data[model][key].append(values[key])
            data[model]["top1_pattern"].append(values["predicted_pattern_8way"])
    out = {"ids": np.array(ids), "gold": np.array(gold)}
    for model, values in data.items():
        out[model] = {key: np.array(v) for key, v in values.items()}
    return out


def score_trajectory(rows: list[dict], checkpoints: list[tuple[int, str]], batch_size: int, device: str) -> dict[int, dict]:
    """Teacher-forced 8-way landscape of one dataset under several checkpoints."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    gold = ["".join("K" if row["answer"][p] == "knight" else "N" for p in row["puzzle"]["people"]) for row in rows]

    result: dict[int, dict] = {}
    for step, adapter in checkpoints:
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16, trust_remote_code=True)
        model = PeftModel.from_pretrained(model, adapter)
        model.to(device)
        model.eval()
        scores, lengths = score_dataset(model, tokenizer, rows, batch_size, device)
        result[step] = landscape_from_scores(scores, lengths, gold)
        del model
        torch.cuda.empty_cache()
        print(f"  scored step {step}", flush=True)
    return result


def spearman(x: np.ndarray, y: np.ndarray) -> dict:
    from scipy.stats import spearmanr

    rho, p = spearmanr(x, y)
    return {"rho": float(rho), "p": float(p), "n": int(len(x))}


def boot_ci(a: np.ndarray, b: np.ndarray, indices: np.ndarray = None, indices_b: np.ndarray = None, seed: int = SEED) -> dict:
    """Bootstrap CI for mean(a) - mean(b).

    Two modes:
      - paired:   pass `indices` only; the same resample is applied to both (same length).
      - two independent groups (n can differ): pass `indices` and `indices_b`.
    """
    if indices is None:
        indices = np.random.default_rng(seed).choice(len(a), size=(BOOTSTRAP, len(a)), replace=True)
    if indices_b is None:
        indices_b = indices if len(a) == len(b) else np.random.default_rng(seed + 1).choice(len(b), size=(BOOTSTRAP, len(b)), replace=True)
    da = a[indices].mean(axis=1)
    db = b[indices_b].mean(axis=1)
    delta = da - db
    lo, hi = np.percentile(delta, [2.5, 97.5])
    return {"mean_delta": float(delta.mean()), "ci_2_5": float(lo), "ci_97_5": float(hi), "ci_crosses_zero": bool(lo < 0 < hi)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    h4 = load_h4()
    init, final = INIT_MODEL[0], FINAL_MODEL[0]
    q0 = h4[init]["gold_q"]
    q1 = h4[final]["gold_q"]
    delta_q = q1 - q0
    correct0 = h4[init]["top1_pattern"] == h4["gold"]
    correct1 = h4[final]["top1_pattern"] == h4["gold"]
    n = len(q0)
    print(f"N = {n}  ({init} -> {final})")

    boot_indices = np.random.default_rng(SEED).choice(n, size=(args.bootstrap, n), replace=True)

    # ------------------------------------------------------------------ A1: initial gold_q bins
    labels, edges, merges = assign_bins(q0)
    bin_index = np.full(n, -1, dtype=int)
    for i, (lo, hi) in enumerate(edges):
        bin_index[(q0 >= lo) & (q0 < hi)] = i

    bins_rows = []
    for i, (label, (lo, hi)) in enumerate(zip(labels, edges)):
        mask = bin_index == i
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        bins_rows.append(
            {
                "bin": label,
                "N": cnt,
                "initial_mean_gold_q": float(q0[mask].mean()),
                "final_mean_gold_q": float(q1[mask].mean()),
                "mean_delta_gold_q": float(delta_q[mask].mean()),
                "initial_mean_gold_rank": float(h4[init]["gold_rank"][mask].mean()),
                "final_mean_gold_rank": float(h4[final]["gold_rank"][mask].mean()),
                "final_top1_accuracy": float(correct1[mask].mean()),
                "initial_top1_accuracy": float(correct0[mask].mean()),
                "frac_gold_q_increased": float((delta_q[mask] > 0).mean()),
                "frac_gold_q_decreased": float((delta_q[mask] < 0).mean()),
                "implied_hit_rate": float((1.0 - (1.0 - q0[mask]) ** K).mean()),
                "implied_miss_rate": float(((1.0 - q0[mask]) ** K).mean()),
                "epoch5_mean_delta_gold_q": float((h4[CONTROL_MODEL]["gold_q"][mask] - q0[mask]).mean()),
            }
        )
    write_csv(
        OUT_DIR / "initial_goldq_bins.csv",
        list(bins_rows[0].keys()),
        [[row[key] for key in bins_rows[0]] for row in bins_rows],
    )

    # ------------------------------------------------------------------ A2: does initial state predict fate?
    outcome_groups = {}
    for name, mask in (
        ("e4_correct_to_v2_correct", correct0 & correct1),
        ("e4_wrong_to_v2_correct", (~correct0) & correct1),
        ("e4_correct_to_v2_wrong", correct0 & (~correct1)),
        ("e4_wrong_to_v2_wrong", (~correct0) & (~correct1)),
    ):
        outcome_groups[name] = {
            "n": int(mask.sum()),
            "initial_mean_gold_q": float(q0[mask].mean()),
            "initial_mean_gold_rank": float(h4[init]["gold_rank"][mask].mean()),
            "initial_mean_gold_margin": float(h4[init]["gold_margin"][mask].mean()),
            "final_mean_gold_q": float(q1[mask].mean()),
            "mean_delta_gold_q": float(delta_q[mask].mean()),
        }

    a2 = {
        "spearman_initial_vs_final_gold_q": spearman(q0, q1),
        "spearman_initial_gold_q_vs_delta_gold_q": spearman(q0, delta_q),
        "spearman_initial_gold_q_vs_final_gold_margin": spearman(q0, h4[final]["gold_margin"]),
        "outcome_groups": outcome_groups,
    }

    # ------------------------------------------- A3-A5: real K=8 hit/miss on the 200-prompt subset
    diag_rows = read_jsonl(DIAG_DATA)
    rollout_e4 = {row["id"]: row for row in read_jsonl(ROLLOUT_E4)}
    rollout_v2 = {row["id"]: row for row in read_jsonl(ROLLOUT_V2)}
    diag_ids = [row["id"] for row in diag_rows]

    checkpoints = [(0, INIT_MODEL[1]), (600, FINAL_MODEL[1])]
    land = score_trajectory(diag_rows, checkpoints, args.batch_size, args.device)
    gold_diag = np.array([pattern_of(row["answer"], row["puzzle"]["people"]) for row in diag_rows])
    dq0, dq1 = land[0]["gold_q"], land[600]["gold_q"]
    ddelta = dq1 - dq0
    dcorrect1 = land[600]["top1_pattern"] == gold_diag

    cc = np.array([rollout_e4[i]["correct_count"] for i in diag_ids])
    hit = cc > 0
    miss = ~hit
    final_all_wrong = np.array([bool(rollout_v2[i]["all_wrong"]) for i in diag_ids])
    final_all_correct = np.array([bool(rollout_v2[i]["all_correct"]) for i in diag_ids])
    final_cc = np.array([rollout_v2[i]["correct_count"] for i in diag_ids])

    implied_hit = 1.0 - (1.0 - dq0) ** K
    a3 = {
        "n": len(diag_ids),
        "K": K,
        "spearman_initial_gold_q_vs_correct_count": spearman(dq0, cc.astype(float)),
        "spearman_initial_gold_q_vs_hit": spearman(dq0, hit.astype(float)),
        "mean_implied_hit_rate": float(implied_hit.mean()),
        "actual_hit_rate": float(hit.mean()),
        "actual_miss_rate": float(miss.mean()),
        "mean_correct_count": float(cc.mean()),
        "by_bin": [],
    }
    for i, (label, (lo, hi)) in enumerate(zip(labels, edges)):
        mask = (dq0 >= lo) & (dq0 < hi)
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        a3["by_bin"].append(
            {
                "bin": label,
                "N": cnt,
                "mean_initial_gold_q": float(dq0[mask].mean()),
                "implied_hit_rate": float(implied_hit[mask].mean()),
                "actual_hit_rate": float(hit[mask].mean()),
                "actual_miss_rate": float(miss[mask].mean()),
                "mean_correct_count": float(cc[mask].mean()),
            }
        )
    write_csv(OUT_DIR / "hit_miss_analysis.csv", list(a3["by_bin"][0].keys()), [[row[k] for k in a3["by_bin"][0]] for row in a3["by_bin"]])

    # ---- A4: does an initial MISS predict later deterioration?
    a4 = {
        "n_hit": int(hit.sum()),
        "n_miss": int(miss.sum()),
        "mean_delta_gold_q_HIT": float(ddelta[hit].mean()),
        "mean_delta_gold_q_MISS": float(ddelta[miss].mean()),
        "delta_diff_HIT_minus_MISS": boot_ci(
            ddelta[hit],
            ddelta[miss],
            np.random.default_rng(SEED + 2).choice(int(hit.sum()), size=(args.bootstrap, int(hit.sum())), replace=True),
            np.random.default_rng(SEED + 2).choice(int(miss.sum()), size=(args.bootstrap, int(miss.sum())), replace=True),
        ),
        "p_final_correct_given_HIT": float(dcorrect1[hit].mean()),
        "p_final_correct_given_MISS": float(dcorrect1[miss].mean()),
        "final_all_wrong_rate_given_HIT": float(final_all_wrong[hit].mean()),
        "final_all_wrong_rate_given_MISS": float(final_all_wrong[miss].mean()),
        "final_all_correct_rate_given_HIT": float(final_all_correct[hit].mean()),
        "final_all_correct_rate_given_MISS": float(final_all_correct[miss].mean()),
        "mean_initial_gold_q_HIT": float(dq0[hit].mean()),
        "mean_initial_gold_q_MISS": float(dq0[miss].mean()),
        "mean_final_correct_count_HIT": float(final_cc[hit].mean()),
        "mean_final_correct_count_MISS": float(final_cc[miss].mean()),
    }

    # ---- A5: control for initial gold_q
    controlled = []
    for i, (label, (lo, hi)) in enumerate(zip(labels, edges)):
        mask = (dq0 >= lo) & (dq0 < hi)
        h = mask & hit
        m = mask & miss
        if int(h.sum()) < 5 or int(m.sum()) < 5:
            controlled.append({"bin": label, "N": int(mask.sum()), "n_hit": int(h.sum()), "n_miss": int(m.sum()), "note": "insufficient"})
            continue
        controlled.append(
            {
                "bin": label,
                "N": int(mask.sum()),
                "n_hit": int(h.sum()),
                "n_miss": int(m.sum()),
                "mean_initial_gold_q_HIT": float(dq0[h].mean()),
                "mean_initial_gold_q_MISS": float(dq0[m].mean()),
                "mean_delta_gold_q_HIT": float(ddelta[h].mean()),
                "mean_delta_gold_q_MISS": float(ddelta[m].mean()),
                "delta_gold_q_diff": float(ddelta[h].mean() - ddelta[m].mean()),
                "final_accuracy_HIT": float(dcorrect1[h].mean()),
                "final_accuracy_MISS": float(dcorrect1[m].mean()),
                "final_all_wrong_HIT": float(final_all_wrong[h].mean()),
                "final_all_wrong_MISS": float(final_all_wrong[m].mean()),
            }
        )
    write_csv(
        OUT_DIR / "controlled_hit_miss.csv",
        ["bin", "N", "n_hit", "n_miss", "mean_initial_gold_q_HIT", "mean_initial_gold_q_MISS", "mean_delta_gold_q_HIT", "mean_delta_gold_q_MISS", "delta_gold_q_diff", "final_accuracy_HIT", "final_accuracy_MISS", "final_all_wrong_HIT", "final_all_wrong_MISS"],
        [[row.get(k, "") for k in ["bin", "N", "n_hit", "n_miss", "mean_initial_gold_q_HIT", "mean_initial_gold_q_MISS", "mean_delta_gold_q_HIT", "mean_delta_gold_q_MISS", "delta_gold_q_diff", "final_accuracy_HIT", "final_accuracy_MISS", "final_all_wrong_HIT", "final_all_wrong_MISS"]] for row in controlled],
    )

    # ---- A6: tipping region (on the N=2000 set, where we have power)
    tipping = []
    for row in bins_rows:
        tipping.append(
            {
                "bin": row["bin"],
                "N": row["N"],
                "implied_miss_rate": row["implied_miss_rate"],
                "mean_delta_gold_q": row["mean_delta_gold_q"],
                "final_top1_accuracy": row["final_top1_accuracy"],
                "frac_gold_q_decreased": row["frac_gold_q_decreased"],
                "epoch5_mean_delta_gold_q": row["epoch5_mean_delta_gold_q"],
            }
        )
    write_csv(OUT_DIR / "tipping_region.csv", list(tipping[0].keys()), [[row[k] for k in tipping[0]] for row in tipping])

    # ------------------------------------------------------------------ Epoch5 control
    q5 = h4[CONTROL_MODEL]["gold_q"]
    epoch5_control = {
        "description": "static Epoch4 -> Epoch5 change, same initial gold_q bins",
        "bins": [
            {
                "bin": row["bin"],
                "N": row["N"],
                "v2_mean_delta_gold_q": row["mean_delta_gold_q"],
                "epoch5_mean_delta_gold_q": float((q5[bin_index == i] - q0[bin_index == i]).mean()),
                "v2_frac_decreased": row["frac_gold_q_decreased"],
                "epoch5_frac_decreased": float(((q5 - q0)[bin_index == i] < 0).mean()),
                "v2_final_top1": row["final_top1_accuracy"],
                "epoch5_final_top1": float((h4[CONTROL_MODEL]["top1_pattern"][bin_index == i] == h4["gold"][bin_index == i]).mean()),
            }
            for i, row in enumerate(bins_rows)
        ],
    }

    bootstrap = {
        "v2_minus_epoch4": {
            "gold_q": boot_ci(q1, q0, boot_indices),
            "top1_accuracy": boot_ci(correct1.astype(float), correct0.astype(float), boot_indices),
        },
        "high_vs_low_quartile_delta_gold_q": {},
        "v2_vs_epoch5_low_support_bin": {},
    }
    order = np.argsort(q0)
    low_q = order[: n // 4]
    high_q = order[-n // 4 :]
    bootstrap["high_vs_low_quartile_delta_gold_q"] = {
        "low_quartile_initial_mean_gold_q": float(q0[low_q].mean()),
        "high_quartile_initial_mean_gold_q": float(q0[high_q].mean()),
        "low_quartile_mean_delta": float(delta_q[low_q].mean()),
        "high_quartile_mean_delta": float(delta_q[high_q].mean()),
        "diff": boot_ci(
            delta_q[high_q],
            delta_q[low_q],
            np.random.default_rng(SEED + 3).choice(len(high_q), size=(args.bootstrap, len(high_q)), replace=True),
            np.random.default_rng(SEED + 3).choice(len(low_q), size=(args.bootstrap, len(low_q)), replace=True),
        ),
        "low_quartile_epoch5_delta": float((q5[low_q] - q0[low_q]).mean()),
        "high_quartile_epoch5_delta": float((q5[high_q] - q0[high_q]).mean()),
    }
    lowest = bin_index == 0
    if lowest.sum() > 0:
        bootstrap["v2_vs_epoch5_low_support_bin"] = {
            "bin": labels[0],
            "N": int(lowest.sum()),
            "v2_vs_epoch5_delta_gold_q": boot_ci(
                (q1 - q0)[lowest],
                (q5 - q0)[lowest],
                np.random.default_rng(SEED + 4).choice(int(lowest.sum()), size=(args.bootstrap, int(lowest.sum())), replace=True),
            ),
        }

    # Mean delta rises in every bin (a few large positive jumps dominate it), so the
    # rich-get-richer signature lives in medians and tail fractions. Bootstrap those.
    bin_boot: dict[str, dict] = {}
    for i, (label, (lo, hi)) in enumerate(zip(labels, edges)):
        mask_idx = np.where(bin_index == i)[0]
        if len(mask_idx) < 10:
            continue
        d2 = delta_q[mask_idx]
        d5 = (q5 - q0)[mask_idx]
        tail2 = (q1[mask_idx] < 0.05).astype(float)
        tail5 = (q5[mask_idx] < 0.05).astype(float)
        resample = np.random.default_rng(SEED + 10 + i).choice(len(mask_idx), size=(args.bootstrap, len(mask_idx)), replace=True)

        def ci(values: np.ndarray) -> dict:
            lo_p, hi_p = np.percentile(values, [2.5, 97.5])
            return {"mean": float(values.mean()), "ci_2_5": float(lo_p), "ci_97_5": float(hi_p), "ci_crosses_zero": bool(lo_p < 0 < hi_p)}

        med_diff = np.median(d2[resample], axis=1) - np.median(d5[resample], axis=1)
        frac_diff = (d2[resample] < 0).mean(axis=1) - (d5[resample] < 0).mean(axis=1)
        tail_diff = tail2[resample].mean(axis=1) - tail5[resample].mean(axis=1)
        bin_boot[label] = {
            "N": int(len(mask_idx)),
            "median_delta_v2": float(np.median(d2)),
            "median_delta_epoch5": float(np.median(d5)),
            "median_delta_diff_v2_minus_e5": ci(med_diff),
            "frac_decreased_v2": float((d2 < 0).mean()),
            "frac_decreased_epoch5": float((d5 < 0).mean()),
            "frac_decreased_diff": ci(frac_diff),
            "tail_q_lt_05_v2": float(tail2.mean()),
            "tail_q_lt_05_epoch5": float(tail5.mean()),
            "tail_diff": ci(tail_diff),
        }
    bootstrap["bin_level_v2_vs_epoch5"] = bin_boot

    (OUT_DIR / "epoch5_control.json").write_text(json.dumps(epoch5_control, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # ------------------------------------------------------------------ Phase B: trajectory over checkpoints
    print("\n=== Phase B: trajectory over GRPO-V2 checkpoints ===")
    checkpoints = [(0, INIT_MODEL[1])] + [(step, f"outputs/grpo_v2_kl001/checkpoint-{step}") for step in TRAJECTORY_STEPS[1:]]
    traj = score_trajectory(diag_rows, checkpoints, args.batch_size, args.device)
    with (OUT_DIR / "trajectory_200.jsonl").open("w", encoding="utf-8") as handle:
        for i, pid in enumerate(diag_ids):
            record = {"id": pid, "gold_pattern": str(gold_diag[i]), "steps": {}}
            for step in TRAJECTORY_STEPS:
                land_step = traj[step]
                record["steps"][step] = {
                    "gold_q": float(land_step["gold_q"][i]),
                    "gold_rank": int(land_step["gold_rank"][i]),
                    "gold_margin": float(land_step["gold_margin"][i]),
                    "entropy": float(land_step["entropy"][i]),
                    "normalized_entropy": float(land_step["normalized_entropy"][i]),
                    "effective_support": float(land_step["effective_support"][i]),
                    "top1_pattern": str(land_step["top1_pattern"][i]),
                    "top1_margin": float(land_step["top1_margin"][i]),
                    "correct": bool(land_step["top1_pattern"][i] == gold_diag[i]),
                }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    # B1: trajectory classification
    final_step = TRAJECTORY_STEPS[-1]
    c0 = np.array([traj[0]["top1_pattern"] == gold_diag])[0]
    c1 = np.array([traj[final_step]["top1_pattern"] == gold_diag])[0]
    classes = {
        "stable_correct": c0 & c1,
        "stable_wrong": (~c0) & (~c1),
        "wrong_to_correct": (~c0) & c1,
        "correct_to_wrong": c0 & (~c1),
    }
    traj_summary = []
    for name, mask in classes.items():
        row = {"class": name, "n": int(mask.sum())}
        for step in TRAJECTORY_STEPS:
            land_step = traj[step]
            row[f"gold_q_step{step}"] = float(land_step["gold_q"][mask].mean()) if mask.sum() else None
            row[f"median_gold_q_step{step}"] = float(np.median(land_step["gold_q"][mask])) if mask.sum() else None
            row[f"gold_margin_step{step}"] = float(land_step["gold_margin"][mask].mean()) if mask.sum() else None
            row[f"norm_entropy_step{step}"] = float(land_step["normalized_entropy"][mask].mean()) if mask.sum() else None
        traj_summary.append(row)
    write_csv(OUT_DIR / "trajectory_summary.csv", list(traj_summary[0].keys()), [[row[k] for k in traj_summary[0]] for row in traj_summary])

    # B2: early divergence
    early_delta = traj[100]["gold_q"] - traj[0]["gold_q"]
    final_delta = traj[final_step]["gold_q"] - traj[0]["gold_q"]
    future_all_correct = final_all_correct
    future_all_wrong = final_all_wrong
    early_divergence = {
        "spearman_early_delta_vs_final_delta": spearman(early_delta, final_delta),
        "step100_gold_q_future_all_correct": float(traj[100]["gold_q"][future_all_correct].mean()),
        "step100_gold_q_future_all_wrong": float(traj[100]["gold_q"][future_all_wrong].mean()),
        "step0_gold_q_future_all_correct": float(traj[0]["gold_q"][future_all_correct].mean()),
        "step0_gold_q_future_all_wrong": float(traj[0]["gold_q"][future_all_wrong].mean()),
        "by_step_gold_q": {
            "future_all_correct": {step: float(traj[step]["gold_q"][future_all_correct].mean()) for step in TRAJECTORY_STEPS},
            "future_all_wrong": {step: float(traj[step]["gold_q"][future_all_wrong].mean()) for step in TRAJECTORY_STEPS},
        },
        "by_step_norm_entropy": {
            "future_all_correct": {step: float(traj[step]["normalized_entropy"][future_all_correct].mean()) for step in TRAJECTORY_STEPS},
            "future_all_wrong": {step: float(traj[step]["normalized_entropy"][future_all_wrong].mean()) for step in TRAJECTORY_STEPS},
        },
    }

    # B4: quartile trajectories
    q0_diag = traj[0]["gold_q"]
    order_diag = np.argsort(q0_diag)
    low_d = order_diag[: len(order_diag) // 4]
    high_d = order_diag[-len(order_diag) // 4 :]
    quartiles = {
        "low_quartile_initial_mean_gold_q": float(q0_diag[low_d].mean()),
        "high_quartile_initial_mean_gold_q": float(q0_diag[high_d].mean()),
        "by_step": {
            "low": {step: {"gold_q": float(traj[step]["gold_q"][low_d].mean()), "top1_margin": float(traj[step]["top1_margin"][low_d].mean()), "norm_entropy": float(traj[step]["normalized_entropy"][low_d].mean())} for step in TRAJECTORY_STEPS},
            "high": {step: {"gold_q": float(traj[step]["gold_q"][high_d].mean()), "top1_margin": float(traj[step]["top1_margin"][high_d].mean()), "norm_entropy": float(traj[step]["normalized_entropy"][high_d].mean())} for step in TRAJECTORY_STEPS},
        },
    }

    # B3: probe alignment (V2 mid-training real rollouts)
    probe = json.loads(PROBE.read_text(encoding="utf-8"))
    probe_alignment = {
        "note": "V2 probe rollouts: 20 prompts x K=8 at each probe step. Sample size is small; reported as descriptive.",
        "n_prompts": len(probe["groups"][0]["prompts"]),
        "probe_steps": [group["probe_step"] for group in probe["groups"]],
        "by_step": [],
    }
    for group in probe["groups"]:
        step = group["probe_step"]
        pids = [p["id"] for p in group["prompts"]]
        step_cc = np.array(
            [sum(1 for r in p["rollouts"] if r.get("parse_success") and r.get("pattern") == p["ground_truth_pattern"]) for p in group["prompts"]]
        )
        idx = [diag_ids.index(pid) for pid in pids if pid in diag_ids]
        if not idx:
            continue
        probe_alignment["by_step"].append(
            {
                "probe_step": step,
                "n": len(idx),
                "mean_correct_count": float(step_cc.mean()),
                "hit_rate": float((step_cc > 0).mean()),
                "mean_gold_q_at_step": float(traj[step]["gold_q"][idx].mean()) if step in traj else None,
                "implied_hit_rate": float((1.0 - (1.0 - traj[step]["gold_q"][idx]) ** K).mean()) if step in traj else None,
            }
        )

    (OUT_DIR / "early_divergence.json").write_text(json.dumps(early_divergence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT_DIR / "bootstrap_results.json").write_text(json.dumps(bootstrap, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    json.dump(
        {
            "a2_initial_state_predicts_fate": a2,
            "a3_real_hit_miss": a3,
            "a4_miss_predicts_deterioration": a4,
            "a5_controlled_hit_miss": controlled,
            "bin_merges": merges,
            "b1_class_counts": {name: int(mask.sum()) for name, mask in classes.items()},
            "b3_probe_alignment": probe_alignment,
            "b4_quartiles": quartiles,
        },
        (OUT_DIR / "h5_analysis.json").open("w", encoding="utf-8"),
        ensure_ascii=False,
        indent=2,
    )

    # ------------------------------------------------------------------ console summary
    print("\n=== A1: initial gold_q bins (N=2000) ===")
    print(f"{'bin':<16}{'N':>6}{'init_q':>9}{'final_q':>9}{'delta':>9}{'final_top1':>11}{'decr%':>8}{'impl_miss':>11}{'e5_delta':>10}")
    for row in bins_rows:
        print(f"{row['bin']:<16}{row['N']:>6}{row['initial_mean_gold_q']:>9.4f}{row['final_mean_gold_q']:>9.4f}"
              f"{row['mean_delta_gold_q']:>+9.4f}{row['final_top1_accuracy']:>11.4f}{row['frac_gold_q_decreased'] * 100:>8.1f}"
              f"{row['implied_miss_rate']:>11.4f}{row['epoch5_mean_delta_gold_q']:>+10.4f}")
    if merges:
        print("  bin merges:", merges)

    print("\n=== A2 ===")
    for key in ("spearman_initial_vs_final_gold_q", "spearman_initial_gold_q_vs_delta_gold_q", "spearman_initial_gold_q_vs_final_gold_margin"):
        print(f"  {key:<48} rho={a2[key]['rho']:+.4f} p={a2[key]['p']:.3e}")
    for name, g in outcome_groups.items():
        print(f"  {name:<28} n={g['n']:<5} init_q={g['initial_mean_gold_q']:.4f} init_rank={g['initial_mean_gold_rank']:.3f} init_margin={g['initial_mean_gold_margin']:+.3f}")

    print("\n=== A3: real K=8 hit/miss (200-prompt) ===")
    print(f"  spearman(gold_q, correct_count) rho={a3['spearman_initial_gold_q_vs_correct_count']['rho']:+.4f}")
    print(f"  implied hit {a3['mean_implied_hit_rate']:.4f} vs actual hit {a3['actual_hit_rate']:.4f} (miss {a3['actual_miss_rate']:.4f})")
    for row in a3["by_bin"]:
        print(f"  {row['bin']:<16} N={row['N']:<5} q0={row['mean_initial_gold_q']:.4f} implied={row['implied_hit_rate']:.4f} actual_hit={row['actual_hit_rate']:.4f} miss={row['actual_miss_rate']:.4f}")

    print("\n=== A4: MISS vs HIT ===")
    print(f"  n_hit={a4['n_hit']} n_miss={a4['n_miss']}")
    print(f"  delta_gold_q  HIT={a4['mean_delta_gold_q_HIT']:+.4f}  MISS={a4['mean_delta_gold_q_MISS']:+.4f}")
    print(f"  final correct P(|HIT)={a4['p_final_correct_given_HIT']:.4f}  P(|MISS)={a4['p_final_correct_given_MISS']:.4f}")
    print(f"  final all-wrong HIT={a4['final_all_wrong_rate_given_HIT']:.4f}  MISS={a4['final_all_wrong_rate_given_MISS']:.4f}")

    print("\n=== A5: controlled HIT vs MISS ===")
    for row in controlled:
        if "delta_gold_q_diff" in row:
            print(f"  {row['bin']:<16} nH={row['n_hit']:<4} nM={row['n_miss']:<4} q0 H={row['mean_initial_gold_q_HIT']:.4f} M={row['mean_initial_gold_q_MISS']:.4f} "
                  f"delta H={row['mean_delta_gold_q_HIT']:+.4f} M={row['mean_delta_gold_q_MISS']:+.4f}")
        else:
            print(f"  {row['bin']:<16} insufficient (n_hit={row.get('n_hit')}, n_miss={row.get('n_miss')})")

    print("\n=== B1: trajectory classes (200-prompt) ===")
    for name, mask in classes.items():
        print(f"  {name:<18} n={int(mask.sum())}")
    print("  gold_q by step:")
    for row in traj_summary:
        print(f"    {row['class']:<18}" + "".join(f"{row[f'gold_q_step{s}']:>9.4f}" if row[f"gold_q_step{s}"] is not None else f"{'-':>9}" for s in TRAJECTORY_STEPS))

    print("\n=== B2: early divergence ===")
    print(f"  spearman(early delta@100, final delta) rho={early_divergence['spearman_early_delta_vs_final_delta']['rho']:+.4f}")
    print("  future all-correct gold_q by step:", {k: round(v, 4) for k, v in early_divergence["by_step_gold_q"]["future_all_correct"].items()})
    print("  future all-wrong   gold_q by step:", {k: round(v, 4) for k, v in early_divergence["by_step_gold_q"]["future_all_wrong"].items()})

    print("\n=== B4: quartiles ===")
    print("  low  ", {s: round(quartiles["by_step"]["low"][s]["gold_q"], 4) for s in TRAJECTORY_STEPS})
    print("  high ", {s: round(quartiles["by_step"]["high"][s]["gold_q"], 4) for s in TRAJECTORY_STEPS})

    print("\n=== Epoch5 control (mean delta; means rise in every bin) ===")
    for row in epoch5_control["bins"]:
        print(f"  {row['bin']:<16} N={row['N']:<6} v2_delta={row['v2_mean_delta_gold_q']:+.4f}  e5_delta={row['epoch5_mean_delta_gold_q']:+.4f}  "
              f"v2_decr={row['v2_frac_decreased']:.3f}  e5_decr={row['epoch5_frac_decreased']:.3f}")

    print("\n=== bin-level V2 vs Epoch5 (median / fraction / tail, 10000 bootstrap) ===")
    print(f"{'bin':<16}{'N':>6}{'medV2':>10}{'medE5':>10}{'medDiff':>20}{'tailV2':>9}{'tailE5':>9}{'tailDiff':>20}")
    for label, row in bootstrap["bin_level_v2_vs_epoch5"].items():
        md, td = row["median_delta_diff_v2_minus_e5"], row["tail_diff"]
        print(f"{label:<16}{row['N']:>6}{row['median_delta_v2']:>+10.4f}{row['median_delta_epoch5']:>+10.4f}"
              f"  {md['mean']:+.4f} [{md['ci_2_5']:+.4f},{md['ci_97_5']:+.4f}]"
              f"{row['tail_q_lt_05_v2']:>9.3f}{row['tail_q_lt_05_epoch5']:>9.3f}"
              f"  {td['mean']:+.4f} [{td['ci_2_5']:+.4f},{td['ci_97_5']:+.4f}]")


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


if __name__ == "__main__":
    main()
