#!/usr/bin/env python3
"""Analysis for the K=8 vs K=16 causal intervention.

Inputs (all produced by audit_k16_probability.py, except the rollout files):
  outputs/grpo_k16_analysis/diag200_probability.json
  outputs/grpo_k16_analysis/fresh_holdout_probability.json
  outputs/grpo_v1_analysis/rollout_A_final200.jsonl      Epoch4,  K_eval=8
  outputs/grpo_v2_analysis/rollout_D_final200.jsonl      K8 (V2), K_eval=8
  outputs/grpo_k16_analysis/rollout_F_final200.jsonl     K16,     K_eval=8

Outputs:
  equal_step_comparison.csv
  equal_budget_comparison.csv
  stratified_low_support.csv
  common_k8_behavior.json
  bootstrap_results.json

Reminder (preregistered): training-time group statistics are process diagnostics only.
With K=16, P(all-wrong) falls mechanically and Pass@16 is not comparable to Pass@8,
so all behavioural conclusions use the common K_eval=8 protocol.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kk_sft.data import read_jsonl  # noqa: E402

OUT_DIR = ROOT / "outputs" / "grpo_k16_analysis"
ROLLOUT = {
    "epoch4": ROOT / "outputs/grpo_v1_analysis/rollout_A_final200.jsonl",
    "k8": ROOT / "outputs/grpo_v2_analysis/rollout_D_final200.jsonl",
    "k16": OUT_DIR / "rollout_F_final200.jsonl",
}

BINS = [(0.0, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.40), (0.40, 0.60), (0.60, 0.80), (0.80, 1.01)]
LOW_SUPPORT_CUT = 0.20
BOOTSTRAP = 10000
SEED = 20260903
K8_STEPS = [100, 200, 300, 400, 500, 600]
K16_STEPS = [100, 200, 300, 400, 500, 600]


def bin_label(lo: float, hi: float) -> str:
    return f"[{lo:.2f},{hi:.2f})" if hi <= 1.0 else f"[{lo:.2f},1.00]"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def arr(data: dict, key: str) -> np.ndarray:
    return np.array(data[key], dtype=float)


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def summarize(q: np.ndarray, rank: np.ndarray, gold: np.ndarray, top1: np.ndarray, q0: np.ndarray, entropy: np.ndarray) -> dict:
    low = q0 < LOW_SUPPORT_CUT
    return {
        "p10_gold_q": float(np.percentile(q, 10)),
        "median_gold_q": float(np.median(q)),
        "mean_gold_q": float(q.mean()),
        "frac_gold_q_lt_05": float((q < 0.05).mean()),
        "top1_accuracy": float((top1 == gold).mean()),
        "top3_coverage": float((rank <= 3).mean()),
        "normalized_entropy": float(entropy.mean()),
        "low_support_n": int(low.sum()),
        "low_support_median_delta": float(np.median(q[low] - q0[low])) if low.sum() else None,
        "low_support_frac_lt_05": float((q[low] < 0.05).mean()) if low.sum() else None,
        "low_support_top1": float((top1[low] == gold[low]).mean()) if low.sum() else None,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    diag = load(OUT_DIR / "diag200_probability.json")
    fresh = load(OUT_DIR / "fresh_holdout_probability.json")

    gold_diag = np.array(diag["gold"])
    q0_diag = arr(diag["models"]["e4_step0"], "gold_q")

    # ---------------------------------------------------------- Comparison A: equal optimizer steps
    equal_rows = []
    for step in K8_STEPS:
        for arm, prefix in (("k8", "k8"), ("k16", "k16")):
            data = diag["models"].get(f"{prefix}_{step}")
            if data is None:
                continue
            summary = summarize(
                arr(data, "gold_q"),
                arr(data, "gold_rank"),
                gold_diag,
                np.array(data["top1_pattern"]),
                q0_diag,
                arr(data, "normalized_entropy"),
            )
            equal_rows.append({"step": step, "arm": arm, **summary})
    write_csv(
        OUT_DIR / "equal_step_comparison.csv",
        ["step", "arm", "p10_gold_q", "median_gold_q", "frac_gold_q_lt_05", "top1_accuracy", "top3_coverage", "normalized_entropy", "low_support_median_delta", "low_support_frac_lt_05"],
        [
            [r["step"], r["arm"], f"{r['p10_gold_q']:.6f}", f"{r['median_gold_q']:.6f}", f"{r['frac_gold_q_lt_05']:.4f}",
             f"{r['top1_accuracy']:.4f}", f"{r['top3_coverage']:.4f}", f"{r['normalized_entropy']:.6f}",
             f"{r['low_support_median_delta']:.6f}" if r["low_support_median_delta"] is not None else "",
             f"{r['low_support_frac_lt_05']:.4f}" if r["low_support_frac_lt_05"] is not None else ""]
            for r in equal_rows
        ],
    )

    # ------------------------------------------- Comparison B: equal total rollout budget
    # K8 step s uses 8*8*s rollouts; K16 step s uses 8*16*s. Equal budget => K16 step s ~ K8 step 2s.
    budget_rows = []
    for k16_step, k8_step in ((100, 200), (200, 400), (300, 600)):
        for arm, label in (("k16", f"k16_{k16_step}"), ("k8", f"k8_{k8_step}")):
            data = diag["models"].get(label)
            if data is None:
                continue
            summary = summarize(
                arr(data, "gold_q"),
                arr(data, "gold_rank"),
                gold_diag,
                np.array(data["top1_pattern"]),
                q0_diag,
                arr(data, "normalized_entropy"),
            )
            budget_rows.append({"pair": f"{k16_step}_vs_{k8_step}", "rollouts": 8 * 16 * k16_step, "arm": arm, "step": k16_step if arm == "k16" else k8_step, **summary})
    write_csv(
        OUT_DIR / "equal_budget_comparison.csv",
        ["pair", "rollouts", "arm", "step", "p10_gold_q", "median_gold_q", "frac_gold_q_lt_05", "top1_accuracy", "top3_coverage", "normalized_entropy", "low_support_median_delta", "low_support_frac_lt_05"],
        [
            [r["pair"], r["rollouts"], r["arm"], r["step"], f"{r['p10_gold_q']:.6f}", f"{r['median_gold_q']:.6f}",
             f"{r['frac_gold_q_lt_05']:.4f}", f"{r['top1_accuracy']:.4f}", f"{r['top3_coverage']:.4f}",
             f"{r['normalized_entropy']:.6f}", f"{r['low_support_median_delta']:.6f}", f"{r['low_support_frac_lt_05']:.4f}"]
            for r in budget_rows
        ],
    )

    # ---------------------------------------------------------- Fresh holdout stratification
    gold_f = np.array(fresh["gold"])
    q_e4 = arr(fresh["models"]["epoch4"], "gold_q")
    top1_e4 = np.array(fresh["models"]["epoch4"]["top1_pattern"])
    models_f = {name: {k: arr(data, k) if k != "top1_pattern" else np.array(data[k]) for k in ("gold_q", "gold_rank", "gold_margin", "normalized_entropy", "effective_support")} for name, data in fresh["models"].items()}
    models_f_top1 = {name: np.array(data["top1_pattern"]) for name, data in fresh["models"].items()}

    strat_rows = []
    for lo, hi in BINS:
        mask = (q_e4 >= lo) & (q_e4 < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        row = {"bin": bin_label(lo, hi), "N": n, "initial_mean_gold_q": float(q_e4[mask].mean())}
        for name in ("k8_best", "k16_best"):
            if name not in models_f:
                continue
            q = models_f[name]["gold_q"]
            delta = q - q_e4
            row[f"{name}_median_delta"] = float(np.median(delta[mask]))
            row[f"{name}_frac_decreased"] = float((delta[mask] < 0).mean())
            row[f"{name}_frac_lt_05"] = float((q[mask] < 0.05).mean())
            row[f"{name}_top1"] = float((models_f_top1[name][mask] == gold_f[mask]).mean())
            c0 = top1_e4[mask] == gold_f[mask]
            c1 = models_f_top1[name][mask] == gold_f[mask]
            row[f"{name}_wrong_to_correct"] = int(((~c0) & c1).sum())
            row[f"{name}_correct_to_wrong"] = int((c0 & (~c1)).sum())
        strat_rows.append(row)
    write_csv(OUT_DIR / "stratified_low_support.csv", list(strat_rows[0].keys()), [[r.get(k, "") for k in strat_rows[0]] for r in strat_rows])

    # ---------------------------------------------------------- bootstrap on the fresh holdout
    n = len(gold_f)
    rng = np.random.default_rng(SEED)
    idx = rng.choice(n, size=(BOOTSTRAP, n), replace=True)
    low_idx = np.where(q_e4 < LOW_SUPPORT_CUT)[0]
    idx_low = rng.choice(len(low_idx), size=(BOOTSTRAP, len(low_idx)), replace=True)

    def ci(values: np.ndarray) -> dict:
        lo, hi = np.percentile(values, [2.5, 97.5])
        return {"mean": float(values.mean()), "ci_2_5": float(lo), "ci_97_5": float(hi), "ci_crosses_zero": bool(lo < 0 < hi)}

    q_k8, q_k16 = models_f["k8_best"]["gold_q"], models_f["k16_best"]["gold_q"]
    delta_k8, delta_k16 = q_k8 - q_e4, q_k16 - q_e4

    bootstrap = {
        "n": n,
        "k8_best_step": fresh.get("k8_best_step"),
        "k16_best_step": fresh.get("k16_best_step"),
        "equal_step_aligned": fresh.get("k8_best_step") == fresh.get("k16_best_step"),
        "primary": {
            "p10_gold_q_k16_minus_k8": ci(np.percentile(q_k16[idx], 10, axis=1) - np.percentile(q_k8[idx], 10, axis=1)),
            "frac_gold_q_lt_05_k16_minus_k8": ci((q_k16[idx] < 0.05).mean(axis=1) - (q_k8[idx] < 0.05).mean(axis=1)),
            "mean_gold_q_k16_minus_k8": ci(q_k16[idx].mean(axis=1) - q_k8[idx].mean(axis=1)),
            "normalized_entropy_k16_minus_k8": ci(models_f["k16_best"]["normalized_entropy"][idx].mean(axis=1) - models_f["k8_best"]["normalized_entropy"][idx].mean(axis=1)),
            "top1_accuracy_k16_minus_k8": ci(
                (models_f_top1["k16_best"][idx] == gold_f[idx]).mean(axis=1) - (models_f_top1["k8_best"][idx] == gold_f[idx]).mean(axis=1)
            ),
        },
        "low_support_region": {
            "cut": LOW_SUPPORT_CUT,
            "n": int(len(low_idx)),
            "median_delta_k8": float(np.median(delta_k8[low_idx])),
            "median_delta_k16": float(np.median(delta_k16[low_idx])),
            "median_delta_k16_minus_k8": ci(np.median(delta_k16[low_idx][idx_low], axis=1) - np.median(delta_k8[low_idx][idx_low], axis=1)),
            "frac_lt_05_k8": float((q_k8[low_idx] < 0.05).mean()),
            "frac_lt_05_k16": float((q_k16[low_idx] < 0.05).mean()),
            "frac_lt_05_k16_minus_k8": ci((q_k16[low_idx][idx_low] < 0.05).mean(axis=1) - (q_k8[low_idx][idx_low] < 0.05).mean(axis=1)),
            "top1_k8": float((models_f_top1["k8_best"][low_idx] == gold_f[low_idx]).mean()),
            "top1_k16": float((models_f_top1["k16_best"][low_idx] == gold_f[low_idx]).mean()),
        },
        "same_step_600": {
            "note": "K8 best (600) vs K16 checkpoint-600, i.e. identical optimizer steps and identical prompt exposure",
            "p10_gold_q_k16_600_minus_k8_600": ci(np.percentile(models_f["k16_600"]["gold_q"][idx], 10, axis=1) - np.percentile(q_k8[idx], 10, axis=1)),
            "frac_lt_05_k16_600_minus_k8_600": ci((models_f["k16_600"]["gold_q"][idx] < 0.05).mean(axis=1) - (q_k8[idx] < 0.05).mean(axis=1)),
            "top1_k16_600_minus_k8_600": ci(
                (models_f_top1["k16_600"][idx] == gold_f[idx]).mean(axis=1) - (models_f_top1["k8_best"][idx] == gold_f[idx]).mean(axis=1)
            ),
            "p10_k8_600": float(np.percentile(q_k8, 10)),
            "p10_k16_600": float(np.percentile(models_f["k16_600"]["gold_q"], 10)),
        },
        "high_support_region": {
            "cut": 0.60,
            "n": int((q_e4 >= 0.60).sum()),
            "median_delta_k8": float(np.median(delta_k8[q_e4 >= 0.60])) if (q_e4 >= 0.60).sum() else None,
            "median_delta_k16": float(np.median(delta_k16[q_e4 >= 0.60])) if (q_e4 >= 0.60).sum() else None,
        },
    }

    # McNemar for exact accuracy
    from scipy.stats import binomtest

    c_k8 = models_f_top1["k8_best"] == gold_f
    c_k16 = models_f_top1["k16_best"] == gold_f
    b = int((c_k8 & (~c_k16)).sum())
    c = int(((~c_k8) & c_k16).sum())
    bootstrap["exact_accuracy_mcnemar"] = {
        "acc_k8": float(c_k8.mean()),
        "acc_k16": float(c_k16.mean()),
        "b_k8_only": b,
        "c_k16_only": c,
        "exact_p": float(binomtest(min(b, c), b + c, 0.5, alternative="two-sided").pvalue) if b + c else None,
    }

    (OUT_DIR / "bootstrap_results.json").write_text(json.dumps(bootstrap, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # ---------------------------------------------------------- common K_eval=8 behaviour
    behavior: dict = {}
    if all(path.exists() for path in ROLLOUT.values()):
        for name, path in ROLLOUT.items():
            rows = {row["id"]: row for row in read_jsonl(path)}
            ids = list(rows)
            n_r = len(ids)
            behavior[name] = {
                "n": n_r,
                "pass_at_8": sum(rows[i]["correct_count"] > 0 for i in ids) / n_r,
                "all_correct": sum(rows[i]["all_correct"] for i in ids) / n_r,
                "all_wrong": sum(rows[i]["all_wrong"] for i in ids) / n_r,
                "mixed": sum(not rows[i]["all_correct"] and not rows[i]["all_wrong"] for i in ids) / n_r,
                "avg_correct_per_group": sum(rows[i]["correct_count"] for i in ids) / n_r,
                "avg_unique_answers": sum(rows[i]["unique_answer_count"] for i in ids) / n_r,
                "mean_reward_mean": sum(rows[i]["reward_mean"] for i in ids) / n_r,
            }
        a = {row["id"]: row for row in read_jsonl(ROLLOUT["epoch4"])}
        for name in ("k8", "k16"):
            t = {row["id"]: row for row in read_jsonl(ROLLOUT[name])}
            state_a = {i: ("all-correct" if a[i]["all_correct"] else ("all-wrong" if a[i]["all_wrong"] else "mixed")) for i in a}
            state_t = {i: ("all-correct" if t[i]["all_correct"] else ("all-wrong" if t[i]["all_wrong"] else "mixed")) for i in t}
            behavior[f"epoch4_to_{name}"] = dict(sorted(Counter(f"{state_a[i]} -> {state_t[i]}" for i in state_a).items()))
        (OUT_DIR / "common_k8_behavior.json").write_text(json.dumps(behavior, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("\n=== common K_eval=8 behaviour (200-prompt) ===")
        for name in ("epoch4", "k8", "k16"):
            if name in behavior:
                b_ = behavior[name]
                print(f"  {name:<8} Pass@8={b_['pass_at_8']:.4f} all-correct={b_['all_correct']:.4f} all-wrong={b_['all_wrong']:.4f} "
                      f"mixed={b_['mixed']:.4f} avg_correct={b_['avg_correct_per_group']:.3f} avg_unique={b_['avg_unique_answers']:.3f}")
        for key in ("epoch4_to_k8", "epoch4_to_k16"):
            if key in behavior:
                print(f"  {key}: {behavior[key]}")
    else:
        print(f"\n[warn] rollout_F_final200.jsonl missing; common K_eval=8 behaviour not written")

    # ---------------------------------------------------------- console summary
    print("\n=== Comparison A: equal optimizer steps (200-prompt) ===")
    print(f"{'step':>6}{'arm':>6}{'p10':>9}{'q<.05':>9}{'top1':>9}{'lowmedΔ':>10}{'low q<.05':>11}")
    for r in equal_rows:
        print(f"{r['step']:>6}{r['arm']:>6}{r['p10_gold_q']:>9.4f}{r['frac_gold_q_lt_05']:>9.4f}{r['top1_accuracy']:>9.4f}"
              f"{r['low_support_median_delta']:>+10.4f}{r['low_support_frac_lt_05']:>11.4f}")

    print("\n=== Comparison B: equal rollout budget ===")
    print(f"{'pair':>16}{'arm':>6}{'step':>6}{'p10':>9}{'q<.05':>9}{'top1':>9}{'lowmedΔ':>10}")
    for r in budget_rows:
        print(f"{r['pair']:>16}{r['arm']:>6}{r['step']:>6}{r['p10_gold_q']:>9.4f}{r['frac_gold_q_lt_05']:>9.4f}"
              f"{r['top1_accuracy']:>9.4f}{r['low_support_median_delta']:>+10.4f}")

    print("\n=== Fresh holdout stratified by Epoch4 initial gold_q ===")
    for r in strat_rows:
        print(f"  {r['bin']:<14} N={r['N']:<5} medΔ k8={r.get('k8_best_median_delta'):+.4f} k16={r.get('k16_best_median_delta'):+.4f} | "
              f"q<.05 k8={r.get('k8_best_frac_lt_05'):.3f} k16={r.get('k16_best_frac_lt_05'):.3f} | "
              f"top1 k8={r.get('k8_best_top1'):.3f} k16={r.get('k16_best_top1'):.3f}")

    print("\n=== bootstrap (fresh holdout, 10000) ===")
    for key, value in bootstrap["primary"].items():
        print(f"  {key:<44} {value['mean']:+.5f} CI=[{value['ci_2_5']:+.5f},{value['ci_97_5']:+.5f}] {'n.s.' if value['ci_crosses_zero'] else 'EXCL0'}")
    print(f"  low-support region n={bootstrap['low_support_region']['n']}")
    print(f"    median Δ k8={bootstrap['low_support_region']['median_delta_k8']:+.5f} k16={bootstrap['low_support_region']['median_delta_k16']:+.5f}")
    for key in ("median_delta_k16_minus_k8", "frac_lt_05_k16_minus_k8"):
        v = bootstrap["low_support_region"][key]
        print(f"    {key:<40} {v['mean']:+.5f} CI=[{v['ci_2_5']:+.5f},{v['ci_97_5']:+.5f}] {'n.s.' if v['ci_crosses_zero'] else 'EXCL0'}")
    print(f"  McNemar exact: {bootstrap['exact_accuracy_mcnemar']}")


if __name__ == "__main__":
    main()
