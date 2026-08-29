#!/usr/bin/env python3
"""H6 Phase C: does group normalization change the LoRA task-gradient geometry?

Zero training: no optimizer step, no new checkpoint. Everything is computed at
SFT Epoch4 with the SAME frozen parameters, SAME input_ids, SAME attention mask,
SAME completion mask, SAME log-probabilities and SAME rewards. Only the advantage
vector is swapped:

    A16      : normalize all 16 rewards together (K=16)
    A8-split : split the SAME 16 completions 8+8 and normalize each half (synthetic K=8)

Loss mirrors TRL 0.23 (`loss_type="dapo"`, ratio = 1 because the policy is frozen
and on-policy here):

    per_token_loss = -min(coef_1, coef_2) * A  ->  -A      (coef_1 = coef_2 = 1)
    loss           = sum(per_token_loss * completion_mask) / num_items_in_batch

KL is deliberately excluded: H6 isolates the reward-relative gradient component.
(In real training beta = 0.01 exists, but at SFT initialization reference ~= policy
so KL is near zero anyway.)

Outputs outputs/grpo_h6_advantage_audit/gradient_geometry.json and
rare_success_gradient_geometry.json
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

from audit_h6_advantage_geometry import (  # noqa: E402
    BASE_MODEL,
    K16,
    OUT_DIR,
    POLICY,
    ZERO_TOL,
    normalize,
)
from kk_sft.data import read_jsonl  # noqa: E402

N_PROMPTS = 64
PARTITION_SEED = 20260905
PARTITIONS = 20
BOOTSTRAP = 10000
BOOT_SEED = 20260906
MAX_NEW_TOKENS = 64


def build_partitions(rng) -> list[list[int]]:
    parts = []
    for _ in range(PARTITIONS):
        order = list(range(K16))
        rng.shuffle(order)
        parts.append(sorted(order[:8]))
    return parts


def select_prompts(records: list[dict], n_target: int) -> list[int]:
    """Prioritise rare success (m=1..3), then mixed medium (4..12), then high (13..15)."""
    ms = [int(sum(c["reward"] for c in r["completions"])) for r in records]
    rare = [i for i, m in enumerate(ms) if 1 <= m <= 3]
    mid = [i for i, m in enumerate(ms) if 4 <= m <= 12]
    high = [i for i, m in enumerate(ms) if 13 <= m <= 15]
    ordered = list(rare)
    pool: list[int] = []
    while mid or high:
        if mid:
            pool.append(mid.pop(0))
        if high:
            pool.append(high.pop(0))
    ordered += pool
    return ordered[:n_target]


def completion_logps_with_grad(model, tokenizer, prompt_texts: list[str], completion_texts: list[str], device: str):
    """Per-token log p of each completion token, retaining the autograd graph."""
    sequences = []
    for prompt_text, completion_text in zip(prompt_texts, completion_texts):
        prompt_ids = tokenizer(prompt_text, add_special_tokens=True)["input_ids"]
        completion_ids = tokenizer(completion_text, add_special_tokens=False)["input_ids"]
        sequences.append(prompt_ids + completion_ids)
    max_len = max(len(seq) for seq in sequences)
    input_ids = torch.full((len(sequences), max_len), tokenizer.pad_token_id, dtype=torch.long)
    attention = torch.zeros((len(sequences), max_len), dtype=torch.long)
    completion_len = [len(tokenizer(c, add_special_tokens=False)["input_ids"]) for c in completion_texts]
    keep = max(completion_len) + 1
    for i, seq in enumerate(sequences):  # LEFT padding so completions end at the same offset
        input_ids[i, max_len - len(seq) :] = torch.tensor(seq, dtype=torch.long)
        attention[i, max_len - len(seq) :] = 1
    input_ids = input_ids.to(device)
    attention = attention.to(device)
    logits = model(input_ids=input_ids, attention_mask=attention, logits_to_keep=keep).logits
    log_probs = torch.log_softmax(logits[:, :-1].float(), dim=-1)  # [B, keep-1, V]
    per_seq = []
    for i, length in enumerate(completion_len):
        targets = input_ids[i, -length:].unsqueeze(-1)
        token_logp = torch.gather(log_probs[i, -length:], 1, targets).squeeze(-1)
        per_seq.append(token_logp)
    return per_seq, completion_len


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-prompts", type=int, default=N_PROMPTS)
    parser.add_argument("--partitions", type=int, default=PARTITIONS)
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rollout_path = OUT_DIR / "fixed_rollouts_200x16.jsonl"
    records = [json.loads(line) for line in rollout_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = select_prompts(records, args.n_prompts)
    print(f"selected {len(selected)} prompts "
          f"(rare success m=1..3: {sum(1 for i in selected if 1 <= int(sum(c['reward'] for c in records[i]['completions'])) <= 3)})")

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16, trust_remote_code=True)
    # is_trainable=True matters: PeftModel.from_pretrained defaults to False, which leaves
    # zero trainable tensors and makes the gradient audit impossible.
    model = PeftModel.from_pretrained(model, POLICY, is_trainable=True)
    model.to(args.device)
    model.eval()  # dropout off; the audit needs a deterministic frozen forward

    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    params = [p for _name, p in trainable]
    print(f"trainable tensors: {len(params)}  params: {sum(p.numel() for p in params)}")

    # prompt texts come from the train file (the rollout file stores completions + rewards)
    train_rows = {row["id"]: row for row in read_jsonl(ROOT / "data/processed/grpo_v1_train.jsonl")}

    rng = np.random.default_rng(PARTITION_SEED)
    partitions = []
    for _ in range(args.partitions):
        partitions.append(sorted(rng.permutation(K16)[:8].tolist()))

    results = []
    for index in selected:
        record = records[index]
        row = train_rows[record["id"]]
        prompt_text = tokenizer.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True)
        completions = [c["text"] for c in record["completions"]]
        rewards = [c["reward"] for c in record["completions"]]
        adv16 = torch.tensor(normalize(rewards), dtype=torch.float32, device=args.device)

        per_seq, lengths = completion_logps_with_grad(model, tokenizer, [prompt_text] * K16, completions, args.device)
        total_tokens = float(sum(lengths))

        def grad_for(adv: torch.Tensor) -> torch.Tensor:
            loss = -(sum((seq * adv[i]).sum() for i, seq in enumerate(per_seq))) / total_tokens
            model.zero_grad(set_to_none=True)
            grads = torch.autograd.grad(loss, params, retain_graph=True)
            return torch.cat([g.detach().reshape(-1).float() for g in grads])

        g16 = grad_for(adv16)
        norm16 = float(g16.norm())
        entry = {
            "id": record["id"],
            "m_correct": int(sum(rewards)),
            "norm16": norm16,
            "partitions": [],
        }
        for part in partitions:
            group2 = [i for i in range(K16) if i not in part]
            adv8 = np.empty(K16, dtype=np.float64)
            for group in (part, group2):
                adv8[group] = normalize([rewards[i] for i in group])
            adv8_t = torch.tensor(adv8, dtype=torch.float32, device=args.device)
            g8 = grad_for(adv8_t)
            norm8 = float(g8.norm())
            cosine = float(torch.nn.functional.cosine_similarity(g16, g8, dim=0))
            entry["partitions"].append(
                {
                    "norm8": norm8,
                    "norm_ratio": (norm16 / norm8) if norm8 > 0 else float("nan"),
                    "cosine": cosine,
                    "relative_diff": float((g16 - g8).norm() / g8.norm()) if norm8 > 0 else float("nan"),
                }
            )
        # free the graph
        del per_seq
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        results.append(entry)
        print(f"  {record['id']} m={entry['m_correct']:<3} ||g16||={norm16:.5f} "
              f"ratio={np.median([p['norm_ratio'] for p in entry['partitions']]):.4f} "
              f"cos={np.median([p['cosine'] for p in entry['partitions']]):.4f}", flush=True)

    payload = {"n_prompts": len(results), "partitions": args.partitions, "policy": POLICY, "loss": "TRL 0.23 dapo-style, ratio=1, KL excluded", "per_prompt": results}
    (OUT_DIR / "gradient_geometry.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    rare = [e for e in results if 1 <= e["m_correct"] <= 3]
    (OUT_DIR / "rare_success_gradient_geometry.json").write_text(
        json.dumps({"n_prompts": len(rare), "partitions": args.partitions, "per_prompt": rare}, indent=2) + "\n", encoding="utf-8"
    )

    def summarize(entries: list[dict], label: str) -> dict:
        ratios = np.array([[p["norm_ratio"] for p in e["partitions"]] for e in entries], dtype=float)
        cosines = np.array([[p["cosine"] for p in e["partitions"]] for e in entries], dtype=float)
        rel = np.array([[p["relative_diff"] for p in e["partitions"]] for e in entries], dtype=float)
        flat = {"norm_ratio": ratios.ravel(), "cosine": cosines.ravel(), "relative_diff": rel.ravel()}
        n_prompts = len(entries)
        boot = np.random.default_rng(BOOT_SEED).choice(n_prompts, size=(args.bootstrap, n_prompts), replace=True)

        def ci(values: np.ndarray) -> dict:
            lo, hi = np.percentile(values, [2.5, 97.5])
            return {"mean": float(values.mean()), "median": float(np.median(values)), "p10": float(np.percentile(values, 10)),
                    "p90": float(np.percentile(values, 90)), "ci_2_5": float(lo), "ci_97_5": float(hi)}

        out = {key: ci(val) for key, val in flat.items()}
        # paired bootstrap over prompts: resample prompts (with replacement), keeping all
        # partitions of a resampled prompt together, then average over prompts x partitions.
        for key, mat in (("norm_ratio", ratios), ("cosine", cosines), ("relative_diff", rel)):
            means = mat[boot, :].mean(axis=(1, 2))  # [bootstrap]
            lo, hi = np.percentile(means, [2.5, 97.5])
            out[key]["prompt_bootstrap_ci"] = [float(lo), float(hi)]
            out[key]["prompt_bootstrap_means_std"] = float(means.std())
        print(f"\n=== {label} (n={n_prompts} prompts x {args.partitions} partitions) ===")
        for key in ("norm_ratio", "cosine", "relative_diff"):
            o = out[key]
            print(f"  {key:<16} mean={o['mean']:.4f} median={o['median']:.4f} p10={o['p10']:.4f} p90={o['p90']:.4f} "
                  f"prompt-CI=[{o['prompt_bootstrap_ci'][0]:.4f},{o['prompt_bootstrap_ci'][1]:.4f}]")
        return out

    summary = {"all": summarize(results, "all selected prompts")}
    if rare:
        summary["rare_success_m1_3"] = summarize(rare, f"rare-success (m=1..3), n={len(rare)}")
    (OUT_DIR / "bootstrap_results.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
