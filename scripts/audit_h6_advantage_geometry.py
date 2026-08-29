#!/usr/bin/env python3
"""H6: group-normalized advantage geometry audit.

H6:
    With binary reward and group-wise sample-std normalization, group size changes
    the effective weighting of successful and failed rollouts even when the
    completions and rewards themselves are held fixed. This altered advantage
    geometry may contribute to the stronger sharpening observed under K=16.

Scope (strict):
    This round can only test whether K changes advantage / task-gradient geometry.
    It can NOT establish that the geometry caused the final polarization.

Phase A (no model): enumerate every binary group composition for K=8 and K=16,
    and validate the closed-form advantage against torch.std(correction=1).

Phase B (one fixed generation): draw 200 prompts (seed 20260904) from the GRPO
    train file, generate 16 completions each ONCE with SFT Epoch4, then compare
    K16 normalization against a synthetic-K8 counterfactual built from the very
    same 16 completions (20 random 8+8 partitions + deterministic first8/last8).

ZERO TRAINING. No optimizer step, no new checkpoint.

Outputs to outputs/grpo_h6_advantage_audit/.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kk_sft.data import read_jsonl  # noqa: E402
from kk_sft.evaluation import parse_answer  # noqa: E402

OUT_DIR = ROOT / "outputs" / "grpo_h6_advantage_audit"
TRAIN_FILE = ROOT / "data" / "processed" / "grpo_v1_train.jsonl"
POLICY = "outputs/sft_v2_5k_p800/checkpoint-1252"
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

K8, K16 = 8, 16
N_PROMPTS = 200
SAMPLE_SEED = 20260904
PARTITION_SEED = 20260905
PARTITIONS = 20
EPS = 1e-4  # TRL 0.23: advantages = (r - mean) / (sample_std + 1e-4)
ZERO_TOL = 1e-6


# --------------------------------------------------------------------------- utilities
def normalize(rewards: list[float]) -> np.ndarray:
    """TRL 0.23 scale_rewards='group': (r - mean) / (sample_std_ddof1 + 1e-4)."""
    r = np.asarray(rewards, dtype=np.float64)
    std = r.std(ddof=1) if len(r) > 1 else 0.0
    return (r - r.mean()) / (std + EPS)


def normalize_no_eps(rewards: list[float]) -> np.ndarray:
    """Closed-form counterpart: (r - mean) / sample_std_ddof1, without the 1e-4 guard.

    The spec's analytic formulas describe this quantity. TRL additionally adds 1e-4 to
    the denominator, which is a ~2.8e-4 relative correction (up to ~1.5e-3 absolute on
    A+). The gate validates the formulas against this no-eps version and reports the
    epsilon correction separately.
    """
    r = np.asarray(rewards, dtype=np.float64)
    std = r.std(ddof=1) if len(r) > 1 else 0.0
    if std <= 0:
        return np.zeros_like(r)
    return (r - r.mean()) / std


def analytic_std(k: int, m: int) -> float:
    return math.sqrt(m * (k - m) / (k * (k - 1)))


def analytic_pos(k: int, m: int) -> float:
    return math.sqrt((k - 1) * (k - m) / (k * m))


def analytic_neg(k: int, m: int) -> float:
    return -math.sqrt((k - 1) * m / (k * (k - m)))


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


# --------------------------------------------------------------------------- Phase A
def phase_a() -> bool:
    rows = []
    max_err = 0.0
    for k in (K8, K16):
        for m in range(k + 1):
            rewards = [1.0] * m + [0.0] * (k - m)
            adv = normalize(rewards)
            adv_no_eps = normalize_no_eps(rewards)
            r = np.asarray(rewards, dtype=np.float64)
            std_torch = float(torch.tensor(rewards, dtype=torch.float32).std(correction=1))
            mean = m / k
            zero_var = m == 0 or m == k

            pos = adv[m - 1] if m > 0 else 0.0
            neg = adv[-1] if m < k else 0.0
            pos_no_eps = adv_no_eps[m - 1] if m > 0 else 0.0
            neg_no_eps = adv_no_eps[-1] if m < k else 0.0
            a_pos = analytic_pos(k, m) if 0 < m < k else 0.0
            a_neg = analytic_neg(k, m) if 0 < m < k else 0.0
            a_std = analytic_std(k, m)

            err = 0.0
            eps_correction = 0.0
            if 0 < m < k:
                err = max(abs(pos_no_eps - a_pos), abs(neg_no_eps - a_neg), abs(std_torch - a_std))
                max_err = max(max_err, err)
                eps_correction = max(abs(pos - pos_no_eps), abs(neg - neg_no_eps))

            rows.append(
                {
                    "K": k,
                    "m_correct": m,
                    "reward_mean": mean,
                    "sample_std": std_torch,
                    "analytic_std": a_std,
                    "zero_variance": zero_var,
                    "pos_advantage_trl": pos,
                    "neg_advantage_trl": neg,
                    "pos_advantage_no_eps": pos_no_eps,
                    "neg_advantage_no_eps": neg_no_eps,
                    "analytic_pos": a_pos,
                    "analytic_neg": a_neg,
                    "formula_max_abs_error_vs_no_eps": err,
                    "eps_correction_magnitude": eps_correction,
                    "sum_positive_advantage": float(adv[adv > 0].sum()),
                    "sum_abs_negative_advantage": float(np.abs(adv[adv < 0]).sum()),
                    "mean_abs_advantage": float(np.abs(adv).mean()),
                    "mean_squared_advantage": float((adv**2).mean()),
                    "n_positive": int((adv > ZERO_TOL).sum()),
                    "n_negative": int((adv < -ZERO_TOL).sum()),
                    "n_zero_advantage": int((np.abs(adv) <= ZERO_TOL).sum()),
                }
            )

    write_csv(
        OUT_DIR / "analytic_advantage_table.csv",
        list(rows[0].keys()),
        [[row[k] for k in rows[0]] for row in rows],
    )
    valid = max_err < 1e-6
    (OUT_DIR / "analytic_gate.json").write_text(
        json.dumps(
            {
                "max_abs_error_formula_vs_no_eps": max_err,
                "tolerance": 1e-6,
                "max_eps_correction": max(row["eps_correction_magnitude"] for row in rows),
                "note": "formulas validated against (r-mean)/std without the 1e-4 guard; TRL's +1e-4 is reported separately",
                "verdict": "H6_ANALYTIC_FORMULA_VALID" if valid else "H6_ANALYTIC_FORMULA_INVALID",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Phase A: max formula error (vs no-eps) = {max_err:.3e} -> {'H6_ANALYTIC_FORMULA_VALID' if valid else 'H6_ANALYTIC_FORMULA_INVALID'}")
    print(f"         max TRL epsilon correction    = {max(row['eps_correction_magnitude'] for row in rows):.3e}")

    print("\n=== rare-success groups (m=1,2,3) ===")
    print(f"{'K':>4}{'m':>4}{'std':>10}{'A+':>10}{'A-':>10}{'sum+':>10}{'sum|−|':>10}{'mean|A|':>10}")
    for row in rows:
        if row["m_correct"] in (1, 2, 3):
            print(f"{row['K']:>4}{row['m_correct']:>4}{row['sample_std']:>10.4f}{row['pos_advantage_trl']:>10.4f}{row['neg_advantage_trl']:>10.4f}"
                  f"{row['sum_positive_advantage']:>10.4f}{row['sum_abs_negative_advantage']:>10.4f}{row['mean_abs_advantage']:>10.4f}")
    return valid


# --------------------------------------------------------------------------- Phase B
def generate_fixed_rollouts(batch_size: int, device: str) -> None:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = read_jsonl(TRAIN_FILE)
    rng = random.Random(SAMPLE_SEED)
    prompts_rows = [rows[i] for i in sorted(rng.sample(range(len(rows)), N_PROMPTS))]

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16, trust_remote_code=True)
    model = PeftModel.from_pretrained(model, POLICY)
    model.to(device)
    model.eval()

    out_path = OUT_DIR / "fixed_rollouts_200x16.jsonl"
    with out_path.open("w", encoding="utf-8") as handle:
        for start in range(0, len(prompts_rows), batch_size):
            batch = prompts_rows[start : start + batch_size]
            texts = [tokenizer.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True) for row in batch]
            expanded, owner = [], []
            for i, text in enumerate(texts):
                expanded.extend([text] * K16)
                owner.extend([i] * K16)
            inputs = tokenizer(expanded, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=64,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.95,
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=True,
                )
            texts_out = tokenizer.batch_decode(generated[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
            grouped: dict[int, list[str]] = defaultdict(list)
            for owner_index, text in zip(owner, texts_out):
                grouped[owner_index].append(text)
            for i, row in enumerate(batch):
                people = row["puzzle"]["people"]
                completions = []
                for text in grouped[i]:
                    parsed = parse_answer(text, people)
                    completions.append(
                        {
                            "text": text,
                            "parsed": parsed.parsed,
                            "format_valid": parsed.format_valid,
                            "parse_success": parsed.parsed is not None,
                            "reward": float(parsed.parsed == row["answer"]),
                            "pattern": "".join("K" if parsed.parsed and parsed.parsed.get(p) == "knight" else "N" for p in people) if parsed.parsed else "INVALID",
                        }
                    )
                record = {
                    "id": row["id"],
                    "gold_pattern": "".join("K" if row["answer"][p] == "knight" else "N" for p in people),
                    "people": people,
                    "completions": completions,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"  generated {start + len(batch)}/{len(prompts_rows)}", flush=True)
    print(f"wrote {out_path}")


def phase_b() -> None:
    path = OUT_DIR / "fixed_rollouts_200x16.jsonl"
    if not path.exists():
        raise SystemExit("fixed_rollouts_200x16.jsonl missing; run with --generate first")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"loaded {len(records)} prompts x {len(records[0]['completions'])} completions")

    # ---- group composition summary
    comp_rows = []
    comp_counter = Counter()
    for record in records:
        rewards = [c["reward"] for c in record["completions"]]
        m = int(sum(rewards))
        comp_counter[m] += 1
        adv16 = normalize(rewards)
        comp_rows.append(
            {
                "id": record["id"],
                "gold_pattern": record["gold_pattern"],
                "m_correct_16": m,
                "reward_mean_16": sum(rewards) / K16,
                "k16_zero_variance": bool(m == 0 or m == K16),
                "k16_sum_positive": float(adv16[adv16 > ZERO_TOL].sum()),
                "k16_sum_abs_negative": float(np.abs(adv16[adv16 < -ZERO_TOL]).sum()),
                "k16_mean_abs_advantage": float(np.abs(adv16).mean()),
                "k16_n_negative": int((adv16 < -ZERO_TOL).sum()),
            }
        )
    write_csv(OUT_DIR / "group_composition_summary.csv", list(comp_rows[0].keys()), [[r[k] for k in comp_rows[0]] for r in comp_rows])

    print("\n=== group composition (m_correct out of 16, n=200 prompts) ===")
    for m in sorted(comp_counter):
        print(f"  m={m:<3} n={comp_counter[m]}")

    # ---- partitions
    rng = random.Random(PARTITION_SEED)

    def make_partitions():
        parts = [list(range(8)), list(range(8, 16))]  # deterministic first8 / last8
        for _ in range(PARTITIONS):
            order = list(range(K16))
            rng.shuffle(order)
            parts.append(sorted(order[:8]))
        return parts  # first entry is first8; we track group2 implicitly as complement

    partition_sets_per_prompt = []
    for _ in records:
        parts = [("first8_last8", list(range(8)))]
        for i in range(PARTITIONS):
            order = list(range(K16))
            rng.shuffle(order)
            parts.append((f"rand{i:02d}", sorted(order[:8])))
        partition_sets_per_prompt.append(parts)

    part_rows = []
    for record, parts in zip(records, partition_sets_per_prompt):
        rewards = [c["reward"] for c in record["completions"]]
        adv16 = normalize(rewards)
        m16 = int(sum(rewards))
        for label, group1 in parts:
            group2 = [i for i in range(K16) if i not in group1]
            adv8 = np.empty(K16, dtype=np.float64)
            for group in (group1, group2):
                adv8[group] = normalize([rewards[i] for i in group])

            diff = adv16 - adv8
            norm16 = np.linalg.norm(adv16)
            norm8 = np.linalg.norm(adv8)
            cosine = float(adv16 @ adv8 / (norm16 * norm8)) if norm16 > 0 and norm8 > 0 else float("nan")

            zero8 = np.abs(adv8) <= ZERO_TOL
            newly_nonzero_wrong = int(np.sum(zero8 & (adv16 < -ZERO_TOL) & (np.array(rewards) == 0.0)))
            newly_nonzero_correct = int(np.sum(zero8 & (adv16 > ZERO_TOL) & (np.array(rewards) == 1.0)))

            part_rows.append(
                {
                    "id": record["id"],
                    "partition": label,
                    "m_correct_16": m16,
                    "mean_abs_delta": float(np.abs(diff).mean()),
                    "max_abs_delta": float(np.abs(diff).max()),
                    "l1_diff": float(np.abs(diff).sum()),
                    "l2_diff": float(np.linalg.norm(diff)),
                    "cosine": cosine,
                    "k16_positive_total_weight": float(adv16[adv16 > ZERO_TOL].sum()),
                    "k8_positive_total_weight": float(adv8[adv8 > ZERO_TOL].sum()),
                    "k16_negative_total_weight": float(np.abs(adv16[adv16 < -ZERO_TOL]).sum()),
                    "k8_negative_total_weight": float(np.abs(adv8[adv8 < -ZERO_TOL]).sum()),
                    "k16_nonzero": int((np.abs(adv16) > ZERO_TOL).sum()),
                    "k8_nonzero": int((np.abs(adv8) > ZERO_TOL).sum()),
                    "newly_nonzero_wrong": newly_nonzero_wrong,
                    "newly_nonzero_correct": newly_nonzero_correct,
                }
            )
    write_csv(OUT_DIR / "partition_advantage_comparison.csv", list(part_rows[0].keys()), [[r[k] for k in part_rows[0]] for r in part_rows])

    # ---- stratified by m_correct (using the 20 random partitions only)
    def bucket(m: int) -> str:
        return {0: "m=0", 1: "m=1", 2: "m=2", 3: "m=3"}.get(m, "m=4-7" if m <= 7 else ("m=8" if m == 8 else ("m=9-12" if m <= 12 else ("m=13-15" if m <= 15 else "m=16"))))

    rand_rows = [r for r in part_rows if r["partition"] != "first8_last8"]
    by_bucket: dict[str, list] = defaultdict(list)
    for row in rand_rows:
        by_bucket[bucket(row["m_correct_16"])].append(row)

    print("\n=== K16 vs synthetic-K8 advantage geometry, by group composition (20 random partitions) ===")
    print(f"{'bucket':<10}{'n_prompts':>10}{'mean|ΔA|':>12}{'cosine':>10}{'new_nonzero_wrong':>20}{'new_nonzero_correct':>22}")
    summary_rows = []
    for name in ["m=0", "m=1", "m=2", "m=3", "m=4-7", "m=8", "m=9-12", "m=13-15", "m=16"]:
        if name not in by_bucket:
            continue
        rows_b = by_bucket[name]
        n_prompts = len(rows_b) // PARTITIONS
        mean_abs = float(np.mean([r["mean_abs_delta"] for r in rows_b]))
        cos = float(np.nanmean([r["cosine"] for r in rows_b]))
        nw = float(np.mean([r["newly_nonzero_wrong"] for r in rows_b]))
        nc = float(np.mean([r["newly_nonzero_correct"] for r in rows_b]))
        print(f"{name:<10}{n_prompts:>10}{mean_abs:>12.4f}{cos:>10.4f}{nw:>20.3f}{nc:>22.3f}")
        summary_rows.append({"bucket": name, "n_prompts": n_prompts, "mean_abs_delta": mean_abs, "cosine": cos, "mean_newly_nonzero_wrong": nw, "mean_newly_nonzero_correct": nc})
    (OUT_DIR / "partition_summary.json").write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # ---- the rare-success mechanism (§9): how often does K16 convert
    #      zero-advantage wrong completions into negative-advantage ones?
    total_newly_nonzero_wrong = sum(r["newly_nonzero_wrong"] for r in rand_rows)
    total_newly_nonzero_correct = sum(r["newly_nonzero_correct"] for r in rand_rows)
    print(f"\n=== rare-success mechanism (§9), summed over {PARTITIONS} partitions x {len(records)} prompts ===")
    print(f"  wrong completions going 0 -> negative : {total_newly_nonzero_wrong}")
    print(f"  correct completions going 0 -> positive: {total_newly_nonzero_correct}")
    json.dump(
        {
            "partitions": PARTITIONS,
            "n_prompts": len(records),
            "total_newly_nonzero_wrong": total_newly_nonzero_wrong,
            "total_newly_nonzero_correct": total_newly_nonzero_correct,
            "per_prompt_per_partition_newly_nonzero_wrong": total_newly_nonzero_wrong / (PARTITIONS * len(records)),
        },
        (OUT_DIR / "newly_nonzero_mechanism.json").open("w", encoding="utf-8"),
        indent=2,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["a", "b", "ab"], default="ab")
    parser.add_argument("--generate", action="store_true", help="generate the fixed 200x16 rollouts (once)")
    parser.add_argument("--batch-size", type=int, default=4, help="prompts per generation batch (x16 sequences)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.phase in ("a", "ab"):
        ok = phase_a()
        if not ok and args.phase == "ab":
            print("analytic gate failed; stopping")
            return

    if args.phase in ("b", "ab"):
        if args.generate:
            generate_fixed_rollouts(args.batch_size, args.device)
        phase_b()


if __name__ == "__main__":
    main()
