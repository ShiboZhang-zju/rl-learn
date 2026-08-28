#!/usr/bin/env python3
"""Zero-training-cost statistical attribution analysis for GRPO-V1.

Reuses existing per-sample predictions for A=SFT Epoch4, B=SFT Epoch5,
C=GRPO Best on the Final Holdout (2000). Produces:
  paired_correctness.json, mcnemar_results.json, bootstrap_results.json,
  class_transition.csv, prediction_transition.csv, feature_transition.json,
  fixed_samples.json, broken_samples.json, rollout_transition_200.json (placeholder),
  and prints the summary used in the final report.
"""

from __future__ import annotations

import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kk_sft.data import read_jsonl  # noqa: E402

OUT = ROOT / "outputs" / "grpo_v1_analysis"
OUT.mkdir(parents=True, exist_ok=True)

LABELS = ["KKK", "KKN", "KNK", "KNN", "NKK", "NKN", "NNK", "NNN"]
FILES = {
    "A": ("sft_epoch4", ROOT / "outputs/grpo_v1_final/sft_epoch4_final_holdout.jsonl"),
    "B": ("sft_epoch5", ROOT / "outputs/grpo_v1_final/sft_epoch5_final_holdout.jsonl"),
    "C": ("grpo_best", ROOT / "outputs/grpo_v1_final/grpo_best_final_holdout.jsonl"),
}


def expr_stats(expr: dict, counts: Counter, depths: list[int], depth: int = 1) -> None:
    counts[expr["op"]] += 1
    depths.append(depth)
    if expr["op"] == "not":
        expr_stats(expr["expr"], counts, depths, depth + 1)
    elif expr["op"] in ("and", "or"):
        expr_stats(expr["left"], counts, depths, depth + 1)
        expr_stats(expr["right"], counts, depths, depth + 1)


def puzzle_features(row: dict) -> dict:
    puzzle = row["puzzle"]
    counts = Counter()
    depths: list[int] = []
    top_ops: list[str] = []
    speakers: list[str] = []
    chars = 0
    for statement in puzzle["statements"]:
        top_ops.append(statement["expr"]["op"])
        speakers.append(statement["speaker"])
        chars += len(statement["text"])
        expr_stats(statement["expr"], counts, depths)
    return {
        "same_count": counts["same"],
        "different_count": counts["different"],
        "and_count": counts["and"],
        "or_count": counts["or"],
        "not_count": counts["not"],
        "expression_nodes": sum(counts.values()),
        "expression_depth": max(depths),
        "statement_chars": chars,
        "top_ops": "+".join(top_ops),
        "speaker_order": "+".join(speakers),
    }


def exact_mcnemar(b: int, c: int) -> dict:
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "n_discordant": 0, "exact_p": 1.0, "chi2_corrected_p": 1.0}
    from scipy.stats import binomtest, chi2

    exact_p = binomtest(k=min(b, c), n=n, p=0.5, alternative="two-sided").pvalue
    chi2_stat = (abs(b - c) - 1.0) ** 2 / n
    chi2_p = chi2.sf(chi2_stat, df=1)
    return {
        "b": b,
        "c": c,
        "n_discordant": n,
        "exact_p": float(exact_p),
        "chi2_corrected_p": float(chi2_p),
        "chi2_statistic": float(chi2_stat),
    }


def paired_bootstrap(delta_fn, idx: np.ndarray, iterations: int, rng) -> dict:
    deltas = np.empty(iterations, dtype=np.float64)
    for i in range(iterations):
        sample = rng.choice(idx, size=len(idx), replace=True)
        deltas[i] = delta_fn(sample)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {
        "mean_delta": float(deltas.mean()),
        "pct_2_5": float(lo),
        "pct_97_5": float(hi),
        "ci_crosses_zero": bool(lo < 0 < hi),
        "iterations": iterations,
        "seed": 20260828,
    }


def main() -> None:
    # ---- Load + alignment ----------------------------------------------------------
    data: dict[str, dict[str, dict]] = {}
    for key, (name, path) in FILES.items():
        rows = read_jsonl(path)
        assert len(rows) == 2000, f"{name}: expected 2000 rows, got {len(rows)}"
        ids = [r["id"] for r in rows]
        assert len(set(ids)) == len(ids), f"{name}: duplicate ids"
        data[key] = {r["id"]: r for r in rows}
        print(f"[load] {name}: {len(data[key])} samples")

    ids_A = set(data["A"])
    ids_B = set(data["B"])
    ids_C = set(data["C"])
    assert ids_A == ids_B == ids_C, "sample id sets differ across models"
    N = len(ids_A)
    id_list = sorted(ids_A)

    gt_diff = sum(1 for i in id_list if data["A"][i]["ground_truth"] != data["B"][i]["ground_truth"] or data["A"][i]["ground_truth"] != data["C"][i]["ground_truth"])
    assert gt_diff == 0, f"ground truth mismatch across models on {gt_diff} samples"
    print(f"[align] N={N} id_sets_equal=True gt_consistent=True")

    # puzzles are not stored in the eval jsonl; join from the holdout data file
    holdout_rows = read_jsonl(ROOT / "data/processed/grpo_v1_final_holdout.jsonl")
    puzzle_by_id = {r["id"]: r["puzzle"] for r in holdout_rows}
    assert len(puzzle_by_id) == N, f"holdout puzzle lookup mismatch: {len(puzzle_by_id)} vs {N}"
    missing = [i for i in id_list if i not in puzzle_by_id]
    assert not missing, f"missing puzzles for {len(missing)} ids"

    # per-sample correctness + features
    rows = []
    for i in id_list:
        a, b, c = data["A"][i], data["B"][i], data["C"][i]
        puzzle = puzzle_by_id[i]
        rows.append(
            {
                "id": i,
                "ground_truth": a["ground_truth"],
                "ground_truth_pattern": a["ground_truth_pattern"],
                "correct_A": int(bool(a["correct"])),
                "correct_B": int(bool(b["correct"])),
                "correct_C": int(bool(c["correct"])),
                "pred_A": a["prediction_pattern"],
                "pred_B": b["prediction_pattern"],
                "pred_C": c["prediction_pattern"],
                "fmt_A": bool(a["format_valid"]),
                "fmt_B": bool(b["format_valid"]),
                "fmt_C": bool(c["format_valid"]),
                "puzzle": puzzle,
                **puzzle_features({"puzzle": puzzle}),
            }
        )

    # ---- 1. Paired correctness transitions (A -> C) -------------------------------
    cc = cw = wc = ww = 0
    for r in rows:
        if r["correct_A"] and r["correct_C"]:
            cc += 1
        elif r["correct_A"] and not r["correct_C"]:
            cw += 1
        elif not r["correct_A"] and r["correct_C"]:
            wc += 1
        else:
            ww += 1
    acc_A = sum(r["correct_A"] for r in rows) / N
    acc_C = sum(r["correct_C"] for r in rows) / N
    delta_AC = acc_C - acc_A
    assert abs(delta_AC - (wc - cw) / N) < 1e-12, "delta identity failed"
    paired = {
        "N": N,
        "CC": cc, "CW": cw, "WC": wc, "WW": ww,
        "CC_ratio": cc / N, "CW_ratio": cw / N, "WC_ratio": wc / N, "WW_ratio": ww / N,
        "acc_A": acc_A, "acc_C": acc_C, "delta_AC": delta_AC,
        "delta_identity_check": f"{delta_AC:.10f} == {(wc - cw) / N:.10f}",
    }
    (OUT / "paired_correctness.json").write_text(json.dumps(paired, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[paired] CC={cc} CW={cw} WC={wc} WW={ww}  acc_A={acc_A:.4f} acc_C={acc_C:.4f} delta={delta_AC:+.4f}")

    # ---- 2. McNemar ------------------------------------------------------------------
    def mcnemar(pair_ab: str, name_a: str, name_b: str) -> dict:
        b_only = c_only = 0
        for r in rows:
            ca = r[f"correct_{pair_ab[0]}"]
            cb = r[f"correct_{pair_ab[1]}"]
            if ca and not cb:
                b_only += 1
            elif not ca and cb:
                c_only += 1
        return {"comparison": f"{name_a}_vs_{name_b}", **exact_mcnemar(b_only, c_only)}

    mcnemar_results = [
        mcnemar("AC", "SFT_Epoch4", "GRPO_Best"),
        mcnemar("BC", "SFT_Epoch5", "GRPO_Best"),
        mcnemar("AB", "SFT_Epoch4", "SFT_Epoch5"),
    ]
    (OUT / "mcnemar_results.json").write_text(json.dumps(mcnemar_results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for m in mcnemar_results:
        print(f"[mcnemar] {m['comparison']}: b={m['b']} c={m['c']} exact_p={m['exact_p']:.6f} chi2_p={m['chi2_corrected_p']:.6f}")

    # ---- 3. Paired bootstrap -------------------------------------------------------------
    rng = np.random.default_rng(20260828)
    idx = np.arange(N)
    acc = {k: np.array([r[f"correct_{k}"] for r in rows], dtype=np.float64) for k in ("A", "B", "C")}
    bs = {
        "GRPO_minus_Epoch4": paired_bootstrap(lambda s: acc["C"][s].mean() - acc["A"][s].mean(), idx, 10000, rng),
        "GRPO_minus_Epoch5": paired_bootstrap(lambda s: acc["C"][s].mean() - acc["B"][s].mean(), idx, 10000, rng),
        "Epoch5_minus_Epoch4": paired_bootstrap(lambda s: acc["B"][s].mean() - acc["A"][s].mean(), idx, 10000, rng),
    }
    for k, v in bs.items():
        print(f"[bootstrap] {k}: mean={v['mean_delta']:+.4f} 95%CI=[{v['pct_2_5']:+.4f},{v['pct_97_5']:+.4f}] crosses_zero={v['ci_crosses_zero']}")
    (OUT / "bootstrap_results.json").write_text(json.dumps(bs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # ---- 4. Class transition (GT pattern), A -> C and B -> C ---------------------------
    def class_table(pair_ab: str) -> list[dict]:
        out = []
        by_gt = defaultdict(list)
        for r in rows:
            by_gt[r["ground_truth_pattern"]].append(r)
        for gt in LABELS:
            grp = by_gt.get(gt, [])
            n = len(grp)
            if n == 0:
                continue
            cA = sum(r[f"correct_{pair_ab[0]}"] for r in grp)
            cB = sum(r[f"correct_{pair_ab[1]}"] for r in grp)
            wc_n = sum(not r[f"correct_{pair_ab[0]}"] and r[f"correct_{pair_ab[1]}"] for r in grp)
            cw_n = sum(r[f"correct_{pair_ab[0]}"] and not r[f"correct_{pair_ab[1]}"] for r in grp)
            out.append(
                {
                    "gt_pattern": gt,
                    "N": n,
                    f"acc_{pair_ab[0]}": cA / n,
                    f"acc_{pair_ab[1]}": cB / n,
                    "delta": (cB - cA) / n,
                    "WC": wc_n,
                    "CW": cw_n,
                    "net": wc_n - cw_n,
                }
            )
        return out

    ct_AC = class_table("AC")
    ct_BC = class_table("BC")
    with (OUT / "class_transition.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ct_AC[0].keys()))
        w.writeheader()
        w.writerows(ct_AC)
        w.writerow({"gt_pattern": "A->C", **{k: "" for k in ct_AC[0] if k != "gt_pattern"}})
        for r in ct_BC:
            w.writerow(
                {
                    "gt_pattern": f"B->C|{r['gt_pattern']}",
                    "N": r["N"],
                    "acc_A": r["acc_B"],
                    "acc_C": r["acc_C"],
                    "delta": r["delta"],
                    "WC": r["WC"],
                    "CW": r["CW"],
                    "net": r["net"],
                }
            )
    for r in ct_AC:
        print(f"[class A->C] {r['gt_pattern']}: N={r['N']} accA={r['acc_A']:.3f} accC={r['acc_C']:.3f} delta={r['delta']:+.3f} WC={r['WC']} CW={r['CW']} net={r['net']:+d}")

    # ---- 5. Feature analysis (A -> C), WC vs CW vs WW ------------------------------------
    groups = {"WC": [], "CW": [], "WW": [], "CC": []}
    for r in rows:
        if r["correct_A"] and r["correct_C"]:
            groups["CC"].append(r)
        elif r["correct_A"] and not r["correct_C"]:
            groups["CW"].append(r)
        elif not r["correct_A"] and r["correct_C"]:
            groups["WC"].append(r)
        else:
            groups["WW"].append(r)
    feature_names = ["same_count", "different_count", "and_count", "or_count", "not_count", "expression_nodes", "expression_depth", "statement_chars"]
    feature_transition = {}
    for gname in ("WC", "CW", "WW", "CC"):
        grp = groups[gname]
        feature_transition[gname] = {"count": len(grp)}
        for feat in feature_names:
            vals = [r[feat] for r in grp]
            feature_transition[gname][feat] = {
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "min": float(min(vals)),
                "max": float(max(vals)),
                "hist": dict(sorted(Counter(vals).items())),
            }
    (OUT / "feature_transition.json").write_text(json.dumps(feature_transition, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\n[features] means: " + json.dumps({g: {f: round(v["mean"], 2) for f in feature_names for v in [feature_transition[g][f]]} for g in ("WC", "CW", "WW", "CC")}))

    # ---- 6. Prediction transition (A pred -> C pred) + subsets ---------------------------
    def pred_matrix(rows_subset, key_a: str, key_c: str, label: str) -> dict:
        matrix = {p: {q: 0 for q in LABELS + ["INVALID"]} for p in LABELS + ["INVALID"]}
        for r in rows_subset:
            p = r[key_a] if r[key_a] in LABELS else "INVALID"
            q = r[key_c] if r[key_c] in LABELS else "INVALID"
            matrix[p][q] += 1
        return {"label": label, "count": len(rows_subset), "matrix": matrix}

    all_matrix = pred_matrix(rows, "pred_A", "pred_C", "all")
    wc_matrix = pred_matrix(groups["WC"], "pred_A", "pred_C", "WC")
    cw_matrix = pred_matrix(groups["CW"], "pred_A", "pred_C", "CW")
    with (OUT / "prediction_transition.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subset", "from_pattern", "to_pattern", "count"])
        for m in (all_matrix, wc_matrix, cw_matrix):
            for p in LABELS + ["INVALID"]:
                for q in LABELS + ["INVALID"]:
                    w.writerow([m["label"], p, q, m["matrix"][p][q]])
    print("\n[pred transition A->C] (to counts for each from row)")
    for p in LABELS:
        print(f"  {p}: " + json.dumps({q: v for q, v in all_matrix["matrix"][p].items() if v}))
    print("  INVALID row:", json.dumps(all_matrix["matrix"]["INVALID"]))

    # ---- 7. Prompt bias: GT freq vs prediction freq per model -----------------------------
    gt_freq = Counter(r["ground_truth_pattern"] for r in rows)
    pred_freq = {k: Counter(r[f"pred_{k}"] if r[f"pred_{k}"] in LABELS else "INVALID" for r in rows) for k in ("A", "B", "C")}
    prompt_bias = {"gt_freq": dict(sorted(gt_freq.items())), "models": {}}
    for k in ("A", "B", "C"):
        prompt_bias["models"][k] = {
            "pred_freq": dict(sorted(pred_freq[k].items())),
            "diff": {p: pred_freq[k].get(p, 0) / N - gt_freq.get(p, 0) / N for p in LABELS},
        }
    print("\n[prompt bias] pred-GT diff by pattern:")
    for k in ("A", "B", "C"):
        print(f"  {k}: " + json.dumps({p: round(v, 4) for p, v in prompt_bias["models"][k]["diff"].items()}))
    (OUT / "prompt_bias.json").write_text(json.dumps(prompt_bias, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # ---- 8. Fixed / broken sample export --------------------------------------------------
    rng_s = random.Random(20260828)
    wc_samples = rng_s.sample(groups["WC"], min(20, len(groups["WC"])))
    cw_samples = rng_s.sample(groups["CW"], min(20, len(groups["CW"])))
    for name, sel in (("fixed_samples", wc_samples), ("broken_samples", cw_samples)):
        out_list = []
        for r in sel:
            out_list.append(
                {
                    "id": r["id"],
                    "ground_truth": r["ground_truth"],
                    "ground_truth_pattern": r["ground_truth_pattern"],
                    "puzzle": r["puzzle"],
                    "features": {f: r[f] for f in feature_names + ["top_ops", "speaker_order"]},
                    "epoch4_prediction": data["A"][r["id"]]["prediction"],
                    "epoch5_prediction": data["B"][r["id"]]["prediction"],
                    "grpo_prediction": data["C"][r["id"]]["prediction"],
                    "epoch4_pattern": r["pred_A"],
                    "grpo_pattern": r["pred_C"],
                }
            )
        (OUT / f"{name}.json").write_text(json.dumps(out_list, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n[samples] WC selected={len(wc_samples)} CW selected={len(cw_samples)}")


if __name__ == "__main__":
    main()
