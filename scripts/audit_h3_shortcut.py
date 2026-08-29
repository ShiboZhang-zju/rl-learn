#!/usr/bin/env python3
"""H3 zero-training diagnostic: does GRPO amplify a structure/operator -> answer-pattern shortcut?

Pure diagnostic. No training, no generator change, no new inference: it reuses the
per-sample predictions that already exist for all five models on the GRPO-V3 fresh
holdout (N=2000, seed 20260902).

Structure features are taken from scripts/audit_dataset_features.py (row_features):
  same_count, different_count, and_count, or_count, not_count, person_is_count,
  expression_nodes, expression_depth, top_ops (per-speaker top-level operator
  sequence), op_signature (full operator multiset).

Two metrics matter and must be read together:

  MI(structure; prediction)          raw mutual information
  NMI = MI / H(prediction)           fraction of the model's *prediction uncertainty*
                                     explained by structure

Raw MI is bounded by H(prediction). GRPO sharpens the policy, which mechanically
lowers H(prediction) and therefore lowers raw MI even if shortcut reliance grows.
NMI removes that confound and is the metric used for the cross-model comparison.

The GT-controlled version (section 6 of the spec) is

  MI(structure; prediction | GT)     and   NMI_cond = MI(.|GT) / H(prediction | GT)

which is the closest available measure of "given the answer is fixed, does structure
still steer the model, and more so after GRPO?".

Writes to outputs/grpo_h3_shortcut_audit/.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_dataset_features import row_features  # noqa: E402
from kk_sft.data import read_jsonl  # noqa: E402

OUT_DIR = ROOT / "outputs" / "grpo_h3_shortcut_audit"
DATA_FILE = ROOT / "data/processed/grpo_v3_final_holdout.jsonl"
PRED_DIR = ROOT / "outputs" / "grpo_v3_final"

MODELS = {
    "epoch4": "sft_epoch4",
    "epoch5": "sft_epoch5",
    "v1": "grpo_v1_best",
    "v2": "grpo_v2_best",
    "v3": "grpo_v3_best",
}
ORDER = ["epoch4", "epoch5", "v1", "v2", "v3"]
LABELS = ["KKK", "KKN", "KNK", "KNN", "NKK", "NKN", "NNK", "NNN"]

COUNT_FEATURES = [
    "same_count",
    "different_count",
    "and_count",
    "or_count",
    "not_count",
    "person_is_count",
    "expression_nodes",
    "expression_depth",
]
# top_ops is the per-speaker (Alice/Bob/Carol) top-level operator sequence. With 3
# statements and 6 operators it has 211 realised levels over N=2000, i.e. ~9.5 samples
# per signature, so only 5 signatures reach N>=20. top_ops_multiset drops the
# speaker->operator pairing and keeps only the operator multiset (<=56 levels), which is
# coarse enough for per-signature accuracy tables while staying purely structural.
CATEGORICAL_FEATURES = ["top_ops", "top_ops_multiset", "op_signature"]
ALL_FEATURES = COUNT_FEATURES + CATEGORICAL_FEATURES
SIGNATURE_FEATURES = ["top_ops", "top_ops_multiset"]

MIN_SIGNATURE_N = 20
PERMUTATIONS = 1000
BOOTSTRAP = 1000
SEED = 20260829


# --------------------------------------------------------------------- information theory
def entropy_from_counts(counts: np.ndarray) -> float:
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log(p)).sum())


def mi_counts(joint: np.ndarray) -> float:
    """Mutual information (nats) from a joint count matrix."""
    total = joint.sum()
    if total == 0:
        return 0.0
    pxy = joint / total
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    outer = np.outer(px, py)
    mask = pxy > 0
    return float((pxy[mask] * np.log(pxy[mask] / outer[mask])).sum())


def make_codes(values: list) -> tuple[np.ndarray, int]:
    mapping: dict = {}
    codes = np.empty(len(values), dtype=np.int64)
    for i, value in enumerate(values):
        index = mapping.get(value)
        if index is None:
            index = len(mapping)
            mapping[value] = index
        codes[i] = index
    return codes, len(mapping)


def mi(x_codes: np.ndarray, y_codes: np.ndarray, kx: int, ky: int) -> float:
    joint = np.bincount(x_codes * ky + y_codes, minlength=kx * ky).reshape(kx, ky).astype(np.float64)
    return mi_counts(joint)


def conditional_mi(x_codes: np.ndarray, y_codes: np.ndarray, z_codes: np.ndarray, kx: int, ky: int, kz: int) -> float:
    """MI(X;Y|Z) = sum_z p(z) MI(X;Y | Z=z)."""
    total = 0.0
    n = len(x_codes)
    for z in range(kz):
        mask = z_codes == z
        count = int(mask.sum())
        if count == 0:
            continue
        total += (count / n) * mi(x_codes[mask], y_codes[mask], kx, ky)
    return total


def conditional_entropy(x_codes: np.ndarray, z_codes: np.ndarray, kx: int, kz: int) -> float:
    """H(X|Z)."""
    total = 0.0
    n = len(x_codes)
    for z in range(kz):
        mask = z_codes == z
        count = int(mask.sum())
        if count == 0:
            continue
        counts = np.bincount(x_codes[mask], minlength=kx).astype(np.float64)
        total += (count / n) * entropy_from_counts(counts)
    return total


# --------------------------------------------------------------------- data loading
def load(consistency_check: bool = True) -> tuple[list[dict], dict[str, list[str]], dict[str, list[bool]]]:
    rows = read_jsonl(DATA_FILE)
    features = [row_features(row) for row in rows]
    predictions: dict[str, list[str]] = {}
    correct: dict[str, list[bool]] = {}
    for short, prefix in MODELS.items():
        path = PRED_DIR / f"{prefix}_grpo_v3_holdout.jsonl"
        pred_rows = {row["id"]: row for row in read_jsonl(path)}
        predictions[short] = [pred_rows[row["id"]]["prediction_pattern"] for row in rows]
        correct[short] = [bool(pred_rows[row["id"]]["correct"]) for row in rows]

    if consistency_check:
        assert len(rows) == 2000, f"expected N=2000, got {len(rows)}"
        ids = [row["id"] for row in rows]
        assert len(set(ids)) == len(ids), "duplicate ids in the holdout"
        for short in MODELS:
            assert len(predictions[short]) == len(rows), f"{short}: prediction count mismatch"
        ground_truth = [feature["pattern"] for feature in features]
        for row, gt in zip(rows, ground_truth):
            own = "".join("K" if row["answer"][p] == "knight" else "N" for p in row["puzzle"]["people"])
            assert own == gt, "feature pattern disagrees with the answer dict"
    return features, predictions, correct


# --------------------------------------------------------------------- outputs helpers
def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--permutations", type=int, default=PERMUTATIONS)
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP)
    parser.add_argument("--min-signature-n", type=int, default=MIN_SIGNATURE_N)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    features, predictions, correct = load()

    gt = [f["pattern"] for f in features]
    ids = [f"row_{i:04d}" for i in range(len(features))]
    for index, feature in enumerate(features):
        feature["id"] = ids[index]

    for feature in features:
        feature["top_ops_multiset"] = "|".join(sorted(feature["top_ops"].split("+")))

    gt_codes, n_gt = make_codes(gt)
    feature_codes: dict[str, tuple[np.ndarray, int]] = {
        name: make_codes([f[name] for f in features]) for name in ALL_FEATURES
    }
    pred_codes = {model: make_codes(predictions[model]) for model in ORDER}
    n_pred = pred_codes["epoch4"][1]

    # ---------------------------------------------------------------- sample_features.csv
    header = ["id", "gt_pattern"]
    header += COUNT_FEATURES + CATEGORICAL_FEATURES + ["expression_nodes", "statement_chars"]
    header = ["id", "gt_pattern"] + COUNT_FEATURES + CATEGORICAL_FEATURES + ["statement_chars"]
    for model in ORDER:
        header += [f"{model}_pred", f"{model}_correct"]
    sample_rows = []
    for i, feature in enumerate(features):
        row = [feature["id"], feature["pattern"]] + [feature[name] for name in COUNT_FEATURES]
        row += [feature[name] for name in CATEGORICAL_FEATURES] + [feature["statement_chars"]]
        for model in ORDER:
            row += [predictions[model][i], int(correct[model][i])]
        sample_rows.append(row)
    write_csv(OUT_DIR / "sample_features.csv", header, sample_rows)

    # ------------------------------------------------------- Q1: structure -> GT (dataset shortcut)
    dataset_mi = {"n": len(features), "features": {}}
    for name in ALL_FEATURES:
        codes, kx = feature_codes[name]
        dataset_mi["features"][name] = {
            "cardinality": kx,
            "mi_vs_gt": mi(codes, gt_codes, kx, n_gt),
            "h_feature": entropy_from_counts(np.bincount(codes, minlength=kx).astype(np.float64)),
            "nmi_vs_gt": None,
        }
        h_gt = entropy_from_counts(np.bincount(gt_codes, minlength=n_gt).astype(np.float64))
        dataset_mi["features"][name]["h_gt"] = h_gt
        dataset_mi["features"][name]["nmi_vs_gt"] = dataset_mi["features"][name]["mi_vs_gt"] / h_gt
    dataset_mi["gt_distribution"] = dict(sorted(Counter(gt).items()))
    (OUT_DIR / "dataset_structure_mi.json").write_text(json.dumps(dataset_mi, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # --------------------------------------- Q2: structure -> prediction (per model) + entropies
    model_mi: dict[str, dict] = {}
    for model in ORDER:
        entry: dict[str, dict] = {}
        pred_h = entropy_from_counts(np.bincount(pred_codes[model][0], minlength=n_pred).astype(np.float64))
        pred_h_given_gt = conditional_entropy(pred_codes[model][0], gt_codes, n_pred, n_gt)
        for name in ALL_FEATURES:
            codes, kx = feature_codes[name]
            raw = mi(codes, pred_codes[model][0], kx, n_pred)
            cond = conditional_mi(codes, pred_codes[model][0], gt_codes, kx, n_pred, n_gt)
            entry[name] = {
                "mi": raw,
                "nmi_over_pred_entropy": raw / pred_h if pred_h > 0 else None,
                "conditional_mi_given_gt": cond,
                "conditional_nmi_given_gt": cond / pred_h_given_gt if pred_h_given_gt > 0 else None,
            }
        model_mi[model] = {
            "features": entry,
            "prediction_entropy": pred_h,
            "prediction_entropy_given_gt": pred_h_given_gt,
            "exact_accuracy": sum(correct[model]) / len(correct[model]),
            "prediction_distribution": dict(sorted(Counter(predictions[model]).items())),
        }
    model_mi["_meta"] = {
        "dataset": str(DATA_FILE.relative_to(ROOT)),
        "n": len(features),
        "models": MODELS,
        "note": "Raw MI is bounded by H(prediction); GRPO lowers H(prediction), so use nmi_over_pred_entropy and conditional_nmi_given_gt for cross-model comparison.",
    }
    (OUT_DIR / "model_prediction_mi.json").write_text(json.dumps(model_mi, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # --------------------------------------------- permutation baseline for MI(structure; prediction)
    rng = np.random.default_rng(SEED)
    permutation: dict[str, dict] = {}
    for model in ORDER:
        permutation[model] = {}
        for name in ALL_FEATURES:
            codes, kx = feature_codes[name]
            observed = model_mi[model]["features"][name]["mi"]
            null = np.empty(args.permutations, dtype=np.float64)
            shuffled = pred_codes[model][0].copy()
            for i in range(args.permutations):
                rng.shuffle(shuffled)
                null[i] = mi(codes, shuffled, kx, n_pred)
            p_value = float((np.sum(null >= observed) + 1) / (args.permutations + 1))
            permutation[model][name] = {
                "observed_mi": observed,
                "null_mean": float(null.mean()),
                "null_std": float(null.std()),
                "null_p95": float(np.percentile(null, 95)),
                "null_max": float(null.max()),
                "empirical_p": p_value,
                "permutations": args.permutations,
                "cardinality": kx,
            }
    (OUT_DIR / "permutation_results.json").write_text(json.dumps(permutation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # ------------------------- bootstrap CI for MI(GRPO) - MI(Epoch4) and for the conditional version
    # Two baselines matter:
    #   vs epoch4 -> total change from the GRPO starting point
    #   vs epoch5 -> isolates GRPO from "just train the SFT model longer", because
    #                Epoch4 -> Epoch5 is more SFT while Epoch4 -> GRPO is RL on top of Epoch4.
    bootstrap: dict[str, dict] = {}
    rng = np.random.default_rng(SEED + 1)
    indices = np.arange(len(features))
    for baseline in ("epoch4", "epoch5"):
        for model in ("v1", "v2", "v3", "epoch5"):
            if model == baseline:
                continue
            key = f"{model}_minus_{baseline}"
            bootstrap[key] = {}
            for name in ALL_FEATURES:
                codes, kx = feature_codes[name]
                deltas, cond_deltas = np.empty(args.bootstrap), np.empty(args.bootstrap)
                for i in range(args.bootstrap):
                    sample = rng.choice(indices, size=len(indices), replace=True)
                    x = codes[sample]
                    y_base = pred_codes[baseline][0][sample]
                    y_model = pred_codes[model][0][sample]
                    z = gt_codes[sample]
                    deltas[i] = mi(x, y_model, kx, n_pred) - mi(x, y_base, kx, n_pred)
                    cond_deltas[i] = conditional_mi(x, y_model, z, kx, n_pred, n_gt) - conditional_mi(x, y_base, z, kx, n_pred, n_gt)
                lo, hi = np.percentile(deltas, [2.5, 97.5])
                clo, chi = np.percentile(cond_deltas, [2.5, 97.5])
                bootstrap[key][name] = {
                    "baseline": baseline,
                    "model": model,
                    "mi_delta_mean": float(deltas.mean()),
                    "ci_2_5": float(lo),
                    "ci_97_5": float(hi),
                    "ci_crosses_zero": bool(lo < 0 < hi),
                    "conditional_mi_delta_mean": float(cond_deltas.mean()),
                    "conditional_ci_2_5": float(clo),
                    "conditional_ci_97_5": float(chi),
                    "conditional_ci_crosses_zero": bool(clo < 0 < chi),
                }
    (OUT_DIR / "bootstrap_results.json").write_text(json.dumps(bootstrap, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # --------------------------------- Q3/Q7: structure-conditioned accuracy (signatures with N >= min)
    signature_sets = {
        name: sorted(s for s, c in Counter(f[name] for f in features).items() if c >= args.min_signature_n)
        for name in SIGNATURE_FEATURES
    }
    structure_rows = []
    for sig_feature in SIGNATURE_FEATURES:
        signatures = signature_sets[sig_feature]
        for signature in signatures:
            mask = [i for i, f in enumerate(features) if f[sig_feature] == signature]
            n = len(mask)
            accuracies = {model: sum(correct[model][i] for i in mask) / n for model in ORDER}
            structure_rows.append(
                {
                    "signature_type": sig_feature,
                    "signature": signature,
                    "n": n,
                    "accuracy": accuracies,
                    "delta_vs_epoch4": {model: accuracies[model] - accuracies["epoch4"] for model in ORDER},
                    "gt_distribution": dict(sorted(Counter(gt[i] for i in mask).items())),
                    "prediction_distribution": {
                        model: dict(sorted(Counter(predictions[model][i] for i in mask).items())) for model in ORDER
                    },
                }
            )
    write_csv(
        OUT_DIR / "structure_accuracy.csv",
        ["signature_type", "signature", "n"] + [f"acc_{m}" for m in ORDER] + [f"delta_{m}" for m in ORDER if m != "epoch4"] + ["gt_distribution"],
        [
            [row["signature_type"], row["signature"], row["n"]]
            + [f"{row['accuracy'][m]:.4f}" for m in ORDER]
            + [f"{row['delta_vs_epoch4'][m]:+.4f}" for m in ORDER if m != "epoch4"]
            + [json.dumps(row["gt_distribution"], ensure_ascii=False)]
            for row in structure_rows
        ],
    )
    (OUT_DIR / "structure_accuracy.json").write_text(json.dumps(structure_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # --------------------------- Q3: per-GT-class conditional analysis (accuracy by (gt, signature))
    conditional_rows = []
    for gt_label in LABELS:
        gt_mask = [i for i, value in enumerate(gt) if value == gt_label]
        for sig_feature in SIGNATURE_FEATURES:
            sub_sig = Counter(features[i][sig_feature] for i in gt_mask)
            for signature, n in sorted(sub_sig.items()):
                if n < args.min_signature_n:
                    continue
                mask = [i for i in gt_mask if features[i][sig_feature] == signature]
                accuracies = {model: sum(correct[model][i] for i in mask) / len(mask) for model in ORDER}
                conditional_rows.append(
                    {
                        "gt": gt_label,
                        "signature_type": sig_feature,
                        "signature": signature,
                        "n": len(mask),
                        "accuracy": accuracies,
                        "delta_vs_epoch4": {model: accuracies[model] - accuracies["epoch4"] for model in ORDER if model != "epoch4"},
                    }
                )
    write_csv(
        OUT_DIR / "conditional_gt_analysis.csv",
        ["gt", "signature_type", "signature", "n"] + [f"acc_{m}" for m in ORDER] + [f"delta_{m}" for m in ORDER if m != "epoch4"],
        [
            [row["gt"], row["signature_type"], row["signature"], row["n"]]
            + [f"{row['accuracy'][m]:.4f}" for m in ORDER]
            + [f"{row['delta_vs_epoch4'][m]:+.4f}" for m in ORDER if m != "epoch4"]
            for row in conditional_rows
        ],
    )
    (OUT_DIR / "conditional_gt_analysis.json").write_text(json.dumps(conditional_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # ------------------ Q4: class bias by structure (P(pred = class | signature, model))
    class_bias_rows = []
    for sig_feature in SIGNATURE_FEATURES:
        for signature in signature_sets[sig_feature]:
            mask = [i for i, f in enumerate(features) if f[sig_feature] == signature]
            n = len(mask)
            for cls in LABELS:
                rates = {model: sum(predictions[model][i] == cls for i in mask) / n for model in ORDER}
                if max(rates.values()) < 0.02:
                    continue
                class_bias_rows.append(
                    {
                        "signature_type": sig_feature,
                        "signature": signature,
                        "n": n,
                        "target_class": cls,
                        **{f"p_{m}": rates[m] for m in ORDER},
                        **{f"delta_{m}": rates[m] - rates["epoch4"] for m in ORDER if m != "epoch4"},
                    }
                )
    write_csv(
        OUT_DIR / "class_bias_by_structure.csv",
        ["signature_type", "signature", "n", "target_class"] + [f"p_{m}" for m in ORDER] + [f"delta_{m}" for m in ORDER if m != "epoch4"],
        [
            [row["signature_type"], row["signature"], row["n"], row["target_class"]]
            + [f"{row[f'p_{m}']:.4f}" for m in ORDER]
            + [f"{row[f'delta_{m}']:+.4f}" for m in ORDER if m != "epoch4"]
            for row in class_bias_rows
        ],
    )

    # --------------------------- Q9: WC / CW transitions by structure (vs Epoch4)
    transition_rows = []
    for sig_feature in SIGNATURE_FEATURES:
        for signature in signature_sets[sig_feature]:
            mask = [i for i, f in enumerate(features) if f[sig_feature] == signature]
            entry = {"signature_type": sig_feature, "signature": signature, "n": len(mask)}
            for model in ("v1", "v2", "v3"):
                wc = sum(1 for i in mask if not correct["epoch4"][i] and correct[model][i])
                cw = sum(1 for i in mask if correct["epoch4"][i] and not correct[model][i])
                entry[f"{model}_wc"] = wc
                entry[f"{model}_cw"] = cw
                entry[f"{model}_wc_rate"] = wc / len(mask)
                entry[f"{model}_cw_rate"] = cw / len(mask)
            transition_rows.append(entry)
    write_csv(
        OUT_DIR / "transition_by_structure.csv",
        ["signature_type", "signature", "n"] + [f"{m}_{k}" for m in ("v1", "v2", "v3") for k in ("wc", "cw", "wc_rate", "cw_rate")],
        [
            [row["signature_type"], row["signature"], row["n"]]
            + [f"{row[f'{m}_{k}']:.4f}" if k.endswith("rate") else row[f"{m}_{k}"] for m in ("v1", "v2", "v3") for k in ("wc", "cw", "wc_rate", "cw_rate")]
            for row in transition_rows
        ],
    )

    # -------------------------------------------------------------- console summary
    print(f"dataset N = {len(features)}")
    for name, values in signature_sets.items():
        print(f"  signatures with n>={args.min_signature_n} [{name}]: {len(values)} "
              f"(coverage {sum(Counter(f[name] for f in features)[s] for s in values) / len(features) * 100:.1f}%)")
    print("\n=== Q1: structure -> ground truth (dataset shortcut) ===")
    print(f"{'feature':<20}{'card':>6}{'MI':>10}{'MI/H(GT)':>12}")
    for name in ALL_FEATURES:
        info = dataset_mi["features"][name]
        print(f"{name:<20}{info['cardinality']:>6}{info['mi_vs_gt']:>10.4f}{info['nmi_vs_gt']:>12.4f}")

    print("\n=== Q2: structure -> prediction (raw MI; bounded by H(pred)) ===")
    print(f"{'feature':<20}" + "".join(f"{m:>12}" for m in ORDER))
    for name in ALL_FEATURES:
        print(f"{name:<20}" + "".join(f"{model_mi[m]['features'][name]['mi']:>12.4f}" for m in ORDER))
    print(f"{'H(prediction)':<20}" + "".join(f"{model_mi[m]['prediction_entropy']:>12.4f}" for m in ORDER))

    print("\n=== Q2 (primary): NMI = MI / H(prediction) ===")
    print(f"{'feature':<20}" + "".join(f"{m:>12}" for m in ORDER))
    for name in ALL_FEATURES:
        print(f"{name:<20}" + "".join(f"{model_mi[m]['features'][name]['nmi_over_pred_entropy']:>12.5f}" for m in ORDER))

    print("\n=== Q3 (primary): conditional NMI given GT = MI(.|GT) / H(pred|GT) ===")
    print(f"{'feature':<20}" + "".join(f"{m:>12}" for m in ORDER))
    for name in ALL_FEATURES:
        print(f"{name:<20}" + "".join(f"{model_mi[m]['features'][name]['conditional_nmi_given_gt']:>12.5f}" for m in ORDER))
    print(f"{'H(pred|GT)':<20}" + "".join(f"{model_mi[m]['prediction_entropy_given_gt']:>12.4f}" for m in ORDER))

    print("\n=== permutation empirical p (MI vs null) ===")
    print(f"{'feature':<20}" + "".join(f"{m:>12}" for m in ORDER))
    for name in ALL_FEATURES:
        print(f"{name:<20}" + "".join(f"{permutation[m][name]['empirical_p']:>12.4f}" for m in ORDER))


if __name__ == "__main__":
    main()
