#!/usr/bin/env python3
"""H4 zero-training audit: the 8-way answer-space probability landscape.

Question: does GRPO primarily reshape/sharpen the probability distribution over the
8 legal answer patterns, rather than broadly increasing answer-space competence?

No temperature sampling. For every puzzle we enumerate all 8 canonical answer
completions, teacher-force them, and normalise:

    s(c)  = sum over completion tokens of log p(c_t | prompt, c_<t)
    q(c)  = softmax over the 8 candidates (logsumexp, completion tokens only)

q is the model's distribution **re-normalised over the 8 canonical legal answers**.
It is NOT the absolute probability inside the full language generation space.

Known tokenization artifact (documented, not hidden):
    "knight" -> 1 token, "knave" -> 2 tokens, so KKK is 18 completion tokens and
    NNN is 21. Sequence log-prob therefore carries a per-candidate length offset.
    The offset is identical for every model (same tokenizer, same 8 strings), so it
    cancels in cross-model deltas, but it does shift absolute q toward knight-heavy
    patterns. Both sequence_sum_logprob (main) and mean_token_logprob (sensitivity)
    are recorded.

Writes to outputs/grpo_h4_probability_audit/.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kk_sft.data import format_answer_only_completion, read_jsonl  # noqa: E402

OUT_DIR = ROOT / "outputs" / "grpo_h4_probability_audit"
MAIN_DATA = ROOT / "data/processed/grpo_v3_final_holdout.jsonl"
DIAG_DATA = ROOT / "outputs/grpo_v3_analysis/diagnostic_200.jsonl"
PRED_DIR = ROOT / "outputs" / "grpo_v3_final"

MODELS = {
    "epoch4": "outputs/sft_v2_5k_p800/checkpoint-1252",
    "epoch5": "outputs/sft_v2_5k_p800/checkpoint-1565",
    "v1": "outputs/grpo_v1/checkpoint-200",
    "v2": "outputs/grpo_v2_kl001/checkpoint-600",
    "v3": "outputs/grpo_v3_partial/checkpoint-400",
}
ORDER = ["epoch4", "epoch5", "v1", "v2", "v3"]
LABELS = ["KKK", "KKN", "KNK", "KNN", "NKK", "NKN", "NNK", "NNN"]
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

BOOTSTRAP = 10000
SEED = 20260830


def canonical_completion(people: list[str], assignment: dict[str, str]) -> str:
    """Byte-identical to kk_sft.data.format_answer_only_completion, for any assignment."""
    answer_lines = [f"{person}: {assignment[person]}" for person in people]
    return "<answer>\n" + "\n".join(answer_lines) + "\n</answer>"


def pattern_of(assignment: dict[str, str], people: list[str]) -> str:
    return "".join("K" if assignment[person] == "knight" else "N" for person in people)


def all_candidates(people: list[str]) -> list[tuple[str, dict[str, str], str]]:
    out = []
    for bits in itertools.product(["knight", "knave"], repeat=len(people)):
        assignment = dict(zip(people, bits))
        pattern = "".join("K" if value == "knight" else "N" for value in bits)
        out.append((pattern, assignment, canonical_completion(people, assignment)))
    return out


def verify_formatter(rows: list[dict]) -> None:
    """The candidate builder must reproduce the project's own completion format."""
    for row in rows:
        people = row["puzzle"]["people"]
        assert canonical_completion(people, row["answer"]) == format_answer_only_completion(row["puzzle"]), (
            "candidate formatter diverged from kk_sft.data.format_answer_only_completion"
        )


@torch.inference_mode()
def score_dataset(model, tokenizer, rows: list[dict], batch_size: int, device: str) -> dict[str, np.ndarray]:
    """Teacher-forced sequence log-prob of all 8 candidates for every puzzle.

    Returns {pattern: np.ndarray[N]} of sequence sum-logprob, plus token counts.
    """
    n = len(rows)
    people0 = rows[0]["puzzle"]["people"]
    candidates = all_candidates(people0)
    scores: dict[str, np.ndarray] = {}
    lengths: dict[str, int] = {}
    prompt_cache = [tokenizer.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True) for row in rows]

    for pattern, _assignment, completion in candidates:
        # completion text is identical for every puzzle (same 3 people), tokenize once
        cand_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
        lengths[pattern] = len(cand_ids)
        values = np.empty(n, dtype=np.float64)
        for start in range(0, n, batch_size):
            batch_rows = rows[start : start + batch_size]
            prompts = prompt_cache[start : start + batch_size]
            sequences = []
            for prompt_text in prompts:
                prompt_ids = tokenizer(prompt_text, add_special_tokens=True)["input_ids"]
                sequences.append(prompt_ids + cand_ids)
            max_len = max(len(seq) for seq in sequences)
            input_ids = torch.full((len(sequences), max_len), tokenizer.pad_token_id, dtype=torch.long)
            attention = torch.zeros((len(sequences), max_len), dtype=torch.long)
            for i, seq in enumerate(sequences):  # LEFT padding so completions end at the same offset
                input_ids[i, max_len - len(seq) :] = torch.tensor(seq, dtype=torch.long)
                attention[i, max_len - len(seq) :] = 1
            input_ids = input_ids.to(device)
            attention = attention.to(device)
            logits = model(input_ids=input_ids, attention_mask=attention).logits
            sliced = logits[:, -len(cand_ids) - 1 : -1].to(torch.float32)
            log_probs = torch.log_softmax(sliced, dim=-1)
            targets = input_ids[:, -len(cand_ids) :].unsqueeze(-1)
            token_logp = torch.gather(log_probs, 2, targets).squeeze(-1)
            values[start : start + len(sequences)] = token_logp.sum(dim=1).detach().cpu().numpy().astype(np.float64)
        scores[pattern] = values
    return scores, lengths


def landscape_from_scores(scores: dict[str, np.ndarray], lengths: dict[str, int], gold: list[str]) -> dict[str, np.ndarray]:
    """Derive q and all per-sample landscape metrics from the 8 score arrays."""
    n = len(gold)
    matrix = np.stack([scores[p] for p in LABELS], axis=1)  # [N, 8]
    length_matrix = np.array([lengths[p] for p in LABELS], dtype=np.float64)[None, :]

    shifted = matrix - matrix.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    q = exp / exp.sum(axis=1, keepdims=True)

    gold_index = np.array([LABELS.index(g) for g in gold])
    gold_score = matrix[np.arange(n), gold_index]
    gold_q = q[np.arange(n), gold_index]

    # rank: 1 = highest score. argsort descending, stable.
    order = np.argsort(-matrix, axis=1, kind="stable")
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.arange(1, 9)[None, :].repeat(n, axis=0), axis=1)
    gold_rank = ranks[np.arange(n), gold_index]

    sorted_scores = np.take_along_axis(matrix, order, axis=1)
    top1_index = order[:, 0]
    top1_q = q[np.arange(n), top1_index]
    top1_margin = sorted_scores[:, 0] - sorted_scores[:, 1]

    # gold margin: s(gold) - max over wrong candidates
    wrong = matrix.copy()
    wrong[np.arange(n), gold_index] = -np.inf
    gold_margin = gold_score - wrong.max(axis=1)

    entropy = -(q * np.log(np.clip(q, 1e-300, None))).sum(axis=1)
    norm_entropy = entropy / math.log(8)
    effective_support = np.exp(entropy)

    mean_token = matrix / length_matrix

    return {
        "scores": matrix,
        "q": q,
        "gold_index": gold_index,
        "gold_score": gold_score,
        "gold_q": gold_q,
        "gold_rank": gold_rank,
        "top1_index": top1_index,
        "top1_pattern": np.array(LABELS)[top1_index],
        "top1_q": top1_q,
        "top1_margin": top1_margin,
        "gold_margin": gold_margin,
        "entropy": entropy,
        "normalized_entropy": norm_entropy,
        "effective_support": effective_support,
        "mean_token_logprob": mean_token,
    }


def aggregate(land: dict[str, np.ndarray], gold: list[str]) -> dict:
    gold_rank = land["gold_rank"]
    gold_q = land["gold_q"]
    return {
        "top1_accuracy_8way": float((land["top1_pattern"] == np.array(gold)).mean()),
        "mean_gold_q": float(gold_q.mean()),
        "median_gold_q": float(np.median(gold_q)),
        "gold_rank_1": float((gold_rank == 1).mean()),
        "gold_rank_le_2": float((gold_rank <= 2).mean()),
        "gold_rank_le_3": float((gold_rank <= 3).mean()),
        "mean_gold_rank": float(gold_rank.mean()),
        "mean_top1_margin": float(land["top1_margin"].mean()),
        "mean_gold_margin": float(land["gold_margin"].mean()),
        "mean_normalized_entropy": float(land["normalized_entropy"].mean()),
        "mean_effective_support": float(land["effective_support"].mean()),
        "gold_q_percentiles": {f"p{p}": float(np.percentile(gold_q, p)) for p in (10, 25, 50, 75, 90)},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="score only the first N puzzles (smoke)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-diagnostic", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = read_jsonl(MAIN_DATA)
    verify_formatter(rows)
    if args.limit:
        rows = rows[: args.limit]
    gold = [pattern_of(row["answer"], row["puzzle"]["people"]) for row in rows]

    diag_rows = [] if args.skip_diagnostic else read_jsonl(DIAG_DATA)
    diag_gold = [pattern_of(row["answer"], row["puzzle"]["people"]) for row in diag_rows]

    # ------------------------------------------------------------------ candidate tokenization
    people = rows[0]["puzzle"]["people"]
    candidates = all_candidates(people)
    cand_ids = {pattern: tokenizer(text, add_special_tokens=False)["input_ids"] for pattern, _a, text in candidates}
    (OUT_DIR / "candidate_tokenization.json").write_text(
        json.dumps(
            {
                "people": people,
                "formatter": "kk_sft.data.format_answer_only_completion (verified byte-identical)",
                "candidates": [
                    {"pattern": pattern, "completion": text, "token_count": len(cand_ids[pattern])} for pattern, _a, text in candidates
                ],
                "length_spread": {
                    "min": min(len(v) for v in cand_ids.values()),
                    "max": max(len(v) for v in cand_ids.values()),
                    "note": "'knight' is 1 token, 'knave' is 2 tokens -> KKK is shorter than NNN",
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("candidate token counts:", {p: len(v) for p, v in cand_ids.items()})

    # ------------------------------------------------------------------ existing greedy predictions
    greedy: dict[str, list[str]] = {}
    for short, prefix in (("epoch4", "sft_epoch4"), ("epoch5", "sft_epoch5"), ("v1", "grpo_v1_best"), ("v2", "grpo_v2_best"), ("v3", "grpo_v3_best")):
        path = PRED_DIR / f"{prefix}_grpo_v3_holdout.jsonl"
        mapping = {row["id"]: row["prediction_pattern"] for row in read_jsonl(path)}
        greedy[short] = [mapping[row["id"]] for row in rows]

    # ------------------------------------------------------------------ scoring
    results: dict[str, dict] = {}
    diag_results: dict[str, dict] = {}
    for short, adapter in MODELS.items():
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16, trust_remote_code=True)
        model = PeftModel.from_pretrained(model, adapter)
        model.to(args.device)
        model.eval()
        scores, lengths = score_dataset(model, tokenizer, rows, args.batch_size, args.device)
        results[short] = landscape_from_scores(scores, lengths, gold)
        if diag_rows:
            dscores, dlengths = score_dataset(model, tokenizer, diag_rows, args.batch_size, args.device)
            diag_results[short] = landscape_from_scores(dscores, dlengths, diag_gold)
        del model
        torch.cuda.empty_cache()
        print(f"scored {short}", flush=True)

    # ------------------------------------------------------------------ correctness gate
    gate = {}
    for short in ORDER:
        agreement = float((results[short]["top1_pattern"] == np.array(greedy[short])).mean())
        gate[short] = {
            "agreement_with_existing_greedy": agreement,
            "disagreements": int((results[short]["top1_pattern"] != np.array(greedy[short])).sum()),
            "n": len(rows),
        }
    gate["_all_models_ge_95pct"] = all(v["agreement_with_existing_greedy"] >= 0.95 for k, v in gate.items() if not k.startswith("_"))
    gate["_verdict"] = "H4_SCORING_VALID" if gate["_all_models_ge_95pct"] else "H4_SCORING_INVALID_ANALYSE_FIRST"

    # ------------------------------------------------------------------ smoke sanity
    sanity = {}
    for short in ORDER:
        land = results[short]
        sanity[short] = {
            "no_nan": bool(np.isfinite(land["scores"]).all() and np.isfinite(land["q"]).all()),
            "q_sum_min": float(land["q"].sum(axis=1).min()),
            "q_sum_max": float(land["q"].sum(axis=1).max()),
            "gold_rank_min": int(land["gold_rank"].min()),
            "gold_rank_max": int(land["gold_rank"].max()),
            "entropy_min": float(land["entropy"].min()),
            "entropy_max": float(land["entropy"].max()),
            "argmax_parseable": bool(all(p in LABELS for p in land["top1_pattern"])),
        }
    sanity["_passed"] = all(
        v["no_nan"] and 0.999 < v["q_sum_min"] <= v["q_sum_max"] < 1.001 and v["gold_rank_min"] == 1 and v["gold_rank_max"] <= 8 and v["argmax_parseable"]
        for k, v in sanity.items()
        if not k.startswith("_")
    )

    if args.limit or not sanity["_passed"] or not gate["_all_models_ge_95pct"]:
        (OUT_DIR / "smoke_check.json").write_text(
            json.dumps({"limit": args.limit, "gate": gate, "sanity": sanity}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print("\n=== GATE ===", json.dumps(gate, ensure_ascii=False, indent=2))
        print("=== SANITY ===", json.dumps(sanity, ensure_ascii=False, indent=2))
        if args.limit:
            print(f"\nsmoke on N={len(rows)} finished; re-run without --limit for the full audit")
            return
        print("\nsanity/gate failed -- not writing interpretation outputs")
        return

    # ------------------------------------------------------------------ aggregate metrics
    aggregates = {short: aggregate(results[short], gold) for short in ORDER}
    (OUT_DIR / "aggregate_metrics.json").write_text(
        json.dumps({"n": len(rows), "gate": gate, "sanity": sanity, "models": aggregates}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # ------------------------------------------------------------------ rank transitions
    transitions = {}
    for target in ("epoch5", "v1", "v2", "v3"):
        base_rank = results["epoch4"]["gold_rank"]
        new_rank = results[target]["gold_rank"]
        matrix = Counter(f"r{int(a)}->r{int(b)}" for a, b in zip(base_rank, new_rank))
        promoted = (base_rank > 1) & (new_rank == 1)
        existing_support = promoted & (base_rank <= 3)
        new_support = promoted & (base_rank > 3)
        demoted = (base_rank == 1) & (new_rank > 1)
        transitions[f"epoch4_to_{target}"] = {
            "matrix": dict(sorted(matrix.items(), key=lambda kv: (int(kv[0].split("->")[0][1:]), int(kv[0].split("->")[1][1:])))),
            "n_promoted_to_rank1": int(promoted.sum()),
            "existing_support_promotion": int(existing_support.sum()),
            "new_support_acquisition": int(new_support.sum()),
            "share_existing_support": float(existing_support.sum() / promoted.sum()) if promoted.sum() else None,
            "n_demoted_from_rank1": int(demoted.sum()),
            "demoted_to_rank_le_3": int((demoted & (new_rank <= 3)).sum()),
            "demoted_to_rank_gt_3": int((demoted & (new_rank > 3)).sum()),
        }
    (OUT_DIR / "rank_transition.json").write_text(json.dumps(transitions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # ------------------------------------------------------------------ top-k coverage
    topk = {}
    for short in ORDER:
        gold_rank = results[short]["gold_rank"]
        topk[short] = {
            "top1_coverage": float((gold_rank <= 1).mean()),
            "top2_coverage": float((gold_rank <= 2).mean()),
            "top3_coverage": float((gold_rank <= 3).mean()),
        }
    (OUT_DIR / "topk_coverage.json").write_text(json.dumps(topk, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # ------------------------------------------------------------------ probability mass by class
    mass = {}
    for short in ORDER:
        q = results[short]["q"]
        by_gt = {}
        for cls in LABELS:
            mask = np.array([g == cls for g in gold])
            if mask.sum() == 0:
                continue
            by_gt[cls] = {"n": int(mask.sum()), "mean_q": {label: float(q[mask, i].mean()) for i, label in enumerate(LABELS)}}
        mass[short] = {"mean_q_by_candidate": {label: float(q[:, i].mean()) for i, label in enumerate(LABELS)}, "by_ground_truth": by_gt}
    (OUT_DIR / "probability_mass_by_class.json").write_text(json.dumps(mass, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # ------------------------------------------------------------------ rollout alignment (200-prompt)
    alignment = {}
    if diag_rows:
        rollout_files = {
            "epoch4": ROOT / "outputs/grpo_v1_analysis/rollout_A_final200.jsonl",
            "v1": ROOT / "outputs/grpo_v1_analysis/rollout_C_final200.jsonl",
            "v2": ROOT / "outputs/grpo_v2_analysis/rollout_D_final200.jsonl",
            "v3": OUT_DIR.parent / "grpo_v3_analysis/rollout_E_final200.jsonl",
        }
        from scipy.stats import spearmanr

        alignment["per_model"] = {}
        for short, path in rollout_files.items():
            rollout = {row["id"]: row for row in read_jsonl(path)}
            ids = [row["id"] for row in diag_rows]
            gold_q = diag_results[short]["gold_q"]
            correct_frac = np.array([rollout[i]["correct_count"] / 8 for i in ids])
            actual_pass = np.array([rollout[i]["correct_count"] > 0 for i in ids], dtype=float)
            implied = 1.0 - (1.0 - gold_q) ** 8
            rho_q, p_q = spearmanr(gold_q, correct_frac)
            rho_pass, p_pass = spearmanr(implied, actual_pass)
            alignment["per_model"][short] = {
                "spearman_gold_q_vs_correct_fraction": {"rho": float(rho_q), "p": float(p_q)},
                "spearman_implied_pass8_vs_actual_pass8": {"rho": float(rho_pass), "p": float(p_pass)},
                "mean_8way_implied_pass8": float(implied.mean()),
                "actual_pass8": float(actual_pass.mean()),
                "mean_correct_fraction": float(correct_frac.mean()),
                "mean_gold_q": float(gold_q.mean()),
            }

        # state-conditioned landscape: mixed -> all-correct vs mixed -> all-wrong
        a_state = {row["id"]: ("all-correct" if row["all_correct"] else ("all-wrong" if row["all_wrong"] else "mixed")) for row in read_jsonl(rollout_files["epoch4"])}
        ids = [row["id"] for row in diag_rows]
        index = {pid: i for i, pid in enumerate(ids)}
        alignment["state_conditioned"] = {}
        for target in ("v1", "v2", "v3"):
            t_state = {row["id"]: ("all-correct" if row["all_correct"] else ("all-wrong" if row["all_wrong"] else "mixed")) for row in read_jsonl(rollout_files[target])}
            groups = defaultdict(list)
            for pid in ids:
                groups[(a_state[pid], t_state[pid])].append(index[pid])
            out = {}
            for (before, after), idx in sorted(groups.items()):
                if len(idx) < 5:
                    continue
                sel = np.array(idx)
                out[f"{before}->{after}"] = {
                    "n": len(idx),
                    "epoch4": {
                        "gold_q": float(diag_results["epoch4"]["gold_q"][sel].mean()),
                        "gold_rank": float(diag_results["epoch4"]["gold_rank"][sel].mean()),
                        "gold_margin": float(diag_results["epoch4"]["gold_margin"][sel].mean()),
                        "normalized_entropy": float(diag_results["epoch4"]["normalized_entropy"][sel].mean()),
                        "effective_support": float(diag_results["epoch4"]["effective_support"][sel].mean()),
                    },
                    target: {
                        "gold_q": float(diag_results[target]["gold_q"][sel].mean()),
                        "gold_rank": float(diag_results[target]["gold_rank"][sel].mean()),
                        "gold_margin": float(diag_results[target]["gold_margin"][sel].mean()),
                        "normalized_entropy": float(diag_results[target]["normalized_entropy"][sel].mean()),
                        "effective_support": float(diag_results[target]["effective_support"][sel].mean()),
                    },
                }
            alignment["state_conditioned"][target] = out
    (OUT_DIR / "rollout_probability_alignment.json").write_text(json.dumps(alignment, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # ------------------------------------------------------------------ paired bootstrap
    boot_indices = np.random.default_rng(SEED).choice(len(rows), size=(args.bootstrap, len(rows)), replace=True)
    arrays = {
        "gold_q": {s: results[s]["gold_q"] for s in ORDER},
        "gold_rank": {s: results[s]["gold_rank"].astype(np.float64) for s in ORDER},
        "top1_margin": {s: results[s]["top1_margin"] for s in ORDER},
        "normalized_entropy": {s: results[s]["normalized_entropy"] for s in ORDER},
        "effective_support": {s: results[s]["effective_support"] for s in ORDER},
        "top1_coverage": {s: (results[s]["gold_rank"] <= 1).astype(np.float64) for s in ORDER},
        "top3_coverage": {s: (results[s]["gold_rank"] <= 3).astype(np.float64) for s in ORDER},
    }
    bootstrap = {}
    for baseline, target in (("epoch4", "v2"), ("epoch5", "v2"), ("epoch4", "v1"), ("epoch4", "epoch5"), ("epoch4", "v3")):
        bootstrap[f"{target}_minus_{baseline}"] = {}
        for metric, per_model in arrays.items():
            a = per_model[target][boot_indices].mean(axis=1)
            b = per_model[baseline][boot_indices].mean(axis=1)
            delta = a - b
            lo, hi = np.percentile(delta, [2.5, 97.5])
            bootstrap[f"{target}_minus_{baseline}"][metric] = {
                "mean_delta": float(delta.mean()),
                "ci_2_5": float(lo),
                "ci_97_5": float(hi),
                "ci_crosses_zero": bool(lo < 0 < hi),
            }
    (OUT_DIR / "bootstrap_results.json").write_text(json.dumps(bootstrap, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # ------------------------------------------------------------------ per-sample CSV / JSONL
    csv_header = ["id", "gold_pattern"]
    for short in ORDER:
        csv_header += [f"{short}_pred8way", f"{short}_gold_q", f"{short}_gold_rank", f"{short}_top1_margin", f"{short}_gold_margin", f"{short}_norm_entropy", f"{short}_eff_support"]
    with (OUT_DIR / "sample_probability_landscape.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(csv_header)
        for i, row in enumerate(rows):
            line = [row["id"], gold[i]]
            for short in ORDER:
                land = results[short]
                line += [
                    land["top1_pattern"][i],
                    f"{land['gold_q'][i]:.6f}",
                    int(land["gold_rank"][i]),
                    f"{land['top1_margin'][i]:.4f}",
                    f"{land['gold_margin'][i]:.4f}",
                    f"{land['normalized_entropy'][i]:.6f}",
                    f"{land['effective_support'][i]:.4f}",
                ]
            writer.writerow(line)
    with (OUT_DIR / "sample_probability_landscape.jsonl").open("w", encoding="utf-8") as handle:
        for i, row in enumerate(rows):
            record = {"id": row["id"], "gold_pattern": gold[i], "models": {}}
            for short in ORDER:
                land = results[short]
                record["models"][short] = {
                    "predicted_pattern_8way": str(land["top1_pattern"][i]),
                    "existing_greedy_pattern": greedy[short][i],
                    "gold_score": float(land["gold_score"][i]),
                    "gold_q": float(land["gold_q"][i]),
                    "gold_rank": int(land["gold_rank"][i]),
                    "top1_pattern": str(land["top1_pattern"][i]),
                    "top1_q": float(land["top1_q"][i]),
                    "top1_margin": float(land["top1_margin"][i]),
                    "gold_margin": float(land["gold_margin"][i]),
                    "entropy": float(land["entropy"][i]),
                    "normalized_entropy": float(land["normalized_entropy"][i]),
                    "effective_support": float(land["effective_support"][i]),
                    "scores": {label: float(land["scores"][i, j]) for j, label in enumerate(LABELS)},
                    "q": {label: float(land["q"][i, j]) for j, label in enumerate(LABELS)},
                }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------ console summary
    print("\n=== GATE (8-way argmax vs existing greedy) ===")
    for short in ORDER:
        print(f"  {short:<8} agreement={gate[short]['agreement_with_existing_greedy']:.4f}  disagreements={gate[short]['disagreements']}")
    print(f"  verdict = {gate['_verdict']}")

    print("\n=== aggregate ===")
    metrics = ["top1_accuracy_8way", "mean_gold_q", "median_gold_q", "gold_rank_1", "gold_rank_le_2", "gold_rank_le_3", "mean_top1_margin", "mean_gold_margin", "mean_normalized_entropy", "mean_effective_support"]
    print(f"{'metric':<26}" + "".join(f"{m:>11}" for m in ORDER))
    for metric in metrics:
        print(f"{metric:<26}" + "".join(f"{aggregates[m][metric]:>11.4f}" for m in ORDER))
    print(f"{'gold_q p10':<26}" + "".join(f"{aggregates[m]['gold_q_percentiles']['p10']:>11.4f}" for m in ORDER))

    print("\n=== top-k gold coverage ===")
    for k in ("top1_coverage", "top2_coverage", "top3_coverage"):
        print(f"  {k:<16}" + "".join(f"{topk[m][k]:>11.4f}" for m in ORDER))

    print("\n=== rank transition (Epoch4 -> X) ===")
    for key, value in transitions.items():
        print(f"  {key}: promoted={value['n_promoted_to_rank1']} "
              f"existing_support(rank<=3->1)={value['existing_support_promotion']} ({value['share_existing_support']:.3f}) "
              f"new_support(rank>3->1)={value['new_support_acquisition']} demoted_from_rank1={value['n_demoted_from_rank1']}")


if __name__ == "__main__":
    main()
