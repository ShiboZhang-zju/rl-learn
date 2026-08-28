#!/usr/bin/env python3
"""Single-GPU custom GRPO trainer for K&K v2 (no TRL GRPOTrainer).

Design (matches the formal GRPO-v1 spec):
- LoRA adapter from an SFT checkpoint is used as the RL policy.
- Rollout: 8 sampled completions per prompt (temperature 0.8, top_p 0.95).
- Reward: +1 iff parsed answer == ground truth, else 0 (no format/parse bonus).
- Advantage: per-group (8 rollouts) normalization with population std;
  if group std == 0 the advantages are forced to 0 (no NaN).
- old_log_probs: rollout-policy logprobs of the actual completions, detached.
- new_log_probs: current-policy logprobs of the same completions, with grad.
- Loss: PPO-style clipped surrogate objective (clip_range 0.2), no KL (beta=0).
- Loss mask: only assistant completion tokens; excludes system/user prompt,
  padding, and any token at/after EOS.
- No vLLM / DeepSpeed / FSDP / reference model / KL / extra rewards.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kk_sft.data import read_jsonl  # noqa: E402
from kk_sft.evaluation import aggregate_metrics, assignment_pattern, parse_answer  # noqa: E402


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    cfg["training"] = {**cfg.get("training", {})}
    cfg["generation"] = {**cfg.get("generation", {})}
    cfg["reward"] = {**cfg.get("reward", {})}
    return cfg


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_prompt_texts(tokenizer, rows: list[dict]) -> list[str]:
    return [tokenizer.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True) for row in rows]


def parse_reward(completion_text: str, row: dict, reward_cfg: dict) -> dict:
    parsed = parse_answer(completion_text, row["puzzle"]["people"])
    correct = parsed.parsed == row["answer"]
    if correct:
        reward = float(reward_cfg.get("correct", 1.0))
    elif parsed.parsed is None:
        reward = float(reward_cfg.get("parse_failure", 0.0))
    elif not parsed.format_valid:
        reward = float(reward_cfg.get("format_invalid", 0.0))
    else:
        reward = float(reward_cfg.get("incorrect", 0.0))
    return {
        "reward": reward,
        "correct": correct,
        "format_valid": parsed.format_valid,
        "parse_success": parsed.parsed is not None,
        "parsed_answer": parsed.parsed,
        "pattern": assignment_pattern(parsed.parsed, row["puzzle"]["people"]),
    }


def group_advantages(rewards_2d: torch.Tensor, eps: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[bool]]:
    """Per-group normalization with population std. Returns (adv, mean, std, zero_var)."""
    mean = rewards_2d.mean(dim=1, keepdim=True)
    std = rewards_2d.std(dim=1, unbiased=False, keepdim=True)
    zero_var = (std < eps).squeeze(1).tolist()
    adv = torch.where(std < eps, torch.zeros_like(rewards_2d), (rewards_2d - mean) / (std + eps))
    return adv, mean, std, zero_var


def compute_log_probs(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_mask: torch.Tensor, use_grad: bool):
    """Return per-position token logprobs (aligned with input_ids[:, 1:]) and the loss mask."""
    if use_grad:
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    else:
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    log_probs = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    targets = input_ids[:, 1:]
    gathered = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
    loss_mask = token_mask[:, 1:].bool()
    return gathered, loss_mask


def greedy_eval(model, tokenizer, rows: list[dict], device: str, batch_size: int, max_new_tokens: int, max_prompt_length: int) -> dict:
    records: list[dict] = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        prompt_texts = build_prompt_texts(tokenizer, batch)
        inputs = tokenizer(
            prompt_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_prompt_length
        ).to(device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
        completions = tokenizer.batch_decode(generated[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        for row, text in zip(batch, completions):
            parsed = parse_answer(text, row["puzzle"]["people"])
            records.append(
                {
                    "id": row["id"],
                    "prediction": text,
                    "parsed_answer": parsed.parsed,
                    "format_valid": parsed.format_valid,
                    "correct": parsed.parsed == row["answer"],
                    "ground_truth_pattern": assignment_pattern(row["answer"], row["puzzle"]["people"]),
                    "prediction_pattern": assignment_pattern(parsed.parsed, row["puzzle"]["people"]),
                }
            )
    return aggregate_metrics(records)


def canonical_answer(answer) -> str:
    if answer is None:
        return None
    return json.dumps(answer, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/grpo_v1_full.yaml"))
    parser.add_argument("--train-file", type=Path)
    parser.add_argument("--val-file", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--no-val", action="store_true")
    parser.add_argument("--no-probe", action="store_true")
    parser.add_argument("--no-checkpoint", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_cfg = cfg["training"]
    gen_cfg = cfg["generation"]
    reward_cfg = cfg["reward"]
    device = choose_device(args.device)
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_file = args.train_file or Path(cfg["train_file"])
    val_file = args.val_file or Path(cfg["val_file"])
    output_dir = args.output_dir or Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_dir = output_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    train_rows = read_jsonl(train_file)
    if args.train_limit:
        train_rows = train_rows[: args.train_limit]
    val_rows = [] if args.no_val else read_jsonl(val_file)
    n_prompts = len(train_rows)
    prompt_batch_size = int(train_cfg["prompt_batch_size"])
    num_gen = int(gen_cfg["num_generations"])
    total_batches = math.ceil(n_prompts / prompt_batch_size)
    max_steps = args.max_steps if args.max_steps is not None else int(train_cfg.get("max_steps", -1))
    if max_steps <= 0:
        max_steps = int(total_batches * float(train_cfg.get("epochs", 1)))
    max_steps = min(max_steps, total_batches)

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = cfg["model_name_or_path"]
    init_checkpoint = Path(cfg["init_checkpoint"])
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = torch.bfloat16 if (bool(train_cfg.get("bf16", False)) and device != "cpu") else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, trust_remote_code=True)
    model = PeftModel.from_pretrained(base_model, init_checkpoint)
    model.to(device)
    model.config.use_cache = False

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    n_trainable = sum(param.numel() for param in trainable_params)
    n_total = sum(param.numel() for param in model.parameters())
    print(f"device={device} dtype={dtype} trainable={n_trainable}/{n_total} ({100 * n_trainable / n_total:.4f}%)", flush=True)
    print(f"prompts={n_prompts} prompt_batch={prompt_batch_size} num_gen={num_gen} batches={total_batches} max_steps={max_steps}", flush=True)
    print(f"advantage_std=population(unbiased=False) beta=0.0", flush=True)

    optimizer = torch.optim.AdamW(trainable_params, lr=float(train_cfg["learning_rate"]), weight_decay=float(train_cfg.get("weight_decay", 0.0)))
    max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
    clip_range = float(train_cfg.get("clip_range", 0.2))
    max_comp_len = int(gen_cfg["max_completion_length"])
    max_prompt_len = int(gen_cfg["max_prompt_length"])
    temperature = float(gen_cfg["temperature"])
    top_p = float(gen_cfg["top_p"])
    eval_every = int(train_cfg.get("eval_every", 100))
    log_every = int(train_cfg.get("log_every", 1))
    max_new_tokens = int(gen_cfg.get("max_completion_length", 64))
    eval_batch = 32
    pad_id = tokenizer.pad_token_id

    train_metrics_path = output_dir / "grpo_v1_train_metrics.jsonl"
    val_metrics_path = output_dir / "grpo_v1_val_metrics.json"
    probe_path = output_dir / "grpo_v1_probe_rollouts.json"

    val_metrics: list[dict] = []
    probe_records: list[dict] = []
    best = {"step": -1, "exact": -1.0, "checkpoint": None}

    # Fixed 20-prompt probe from V2 Val.
    probe_rows: list[dict] = []
    probe_seed_meta = {}
    if not args.no_probe and val_rows:
        probe_rng = random.Random(20260830)
        probe_indices = probe_rng.sample(range(len(val_rows)), 20)
        probe_rows = [val_rows[i] for i in probe_indices]
        probe_seed_meta = {"seed": 20260830, "indices": probe_indices, "count": len(probe_rows)}

    def run_probe(step: int) -> None:
        if not probe_rows:
            return
        model.eval()
        prompts = build_prompt_texts(tokenizer, probe_rows)
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_prompt_len).to(device)
        expanded = {k: v.repeat_interleave(num_gen, dim=0) for k, v in inputs.items()}
        with torch.inference_mode():
            generated = model.generate(
                **expanded,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=pad_id,
                use_cache=True,
            )
        prompt_len = expanded["input_ids"].shape[1]
        texts = tokenizer.batch_decode(generated[:, prompt_len:], skip_special_tokens=True)
        group = {"probe_step": step}
        for index, row in enumerate(probe_rows):
            samples = texts[index * num_gen : (index + 1) * num_gen]
            rollouts = [parse_reward(text, row, reward_cfg) for text in samples]
            group.setdefault("prompts", []).append(
                {
                    "id": row["id"],
                    "ground_truth_pattern": assignment_pattern(row["answer"], row["puzzle"]["people"]),
                    "rollouts": [
                        {
                            "text": samples[i],
                            "reward": rollout["reward"],
                            "pattern": rollout["pattern"],
                            "format_valid": rollout["format_valid"],
                            "parse_success": rollout["parse_success"],
                        }
                        for i, rollout in enumerate(rollouts)
                    ],
                }
            )
        probe_records.append(group)
        probe_path.write_text(json.dumps({"probe_seed_meta": probe_seed_meta, "groups": probe_records}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        model.train()

    def save_checkpoint(step: int) -> None:
        if args.no_checkpoint:
            return
        ckpt_dir = output_dir / f"checkpoint-{step}"
        model.save_pretrained(str(ckpt_dir))
        print(f"saved checkpoint {ckpt_dir}", flush=True)

    def run_val(step: int) -> None:
        if not val_rows:
            return
        model.eval()
        metrics = greedy_eval(model, tokenizer, val_rows, device, eval_batch, max_new_tokens, max_prompt_len)
        model.train()
        entry = {"step": step, **metrics}
        val_metrics.append(entry)
        val_metrics_path.write_text(json.dumps(val_metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[VAL step={step}] exact={metrics['exact_accuracy']:.4f} format={metrics['format_accuracy']:.4f} parse={metrics['parse_success_rate']:.4f}", flush=True)
        if metrics["exact_accuracy"] > best["exact"]:
            best.update(step=step, exact=metrics["exact_accuracy"])
            best["checkpoint"] = str(output_dir / f"checkpoint-{step}")
            print(f"[BEST] new best val exact {metrics['exact_accuracy']:.4f} at step {step}", flush=True)

    # Step 0 baseline
    save_checkpoint(0)
    run_val(0)
    if not args.no_probe:
        run_probe(0)

    # Debug/audit dump for step 0 (logprob mask inspection).
    audit_dumped = False

    def dump_audit_sample(row, prompt_text, completion_text, input_ids, token_mask, row_index: int) -> None:
        seq_len = input_ids.shape[1]
        # loss position p corresponds to predicting token at position p+1.
        loss_positions = [p for p in range(seq_len - 1) if bool(token_mask[row_index, p + 1])]
        token_strings = tokenizer.convert_ids_to_tokens(input_ids[row_index].tolist())
        audit = {
            "step": 0,
            "row_index": row_index,
            "id": row["id"],
            "prompt_text": prompt_text,
            "completion_text": completion_text,
            "input_ids": input_ids[row_index].tolist(),
            "token_strings": token_strings,
            "completion_mask": [bool(m) for m in token_mask[row_index].tolist()],
            "loss_token_positions": loss_positions,
            "note": (
                "loss_token_positions p means token p+1 is an assistant completion token. "
                "Excludes system/user prompt tokens, padding, and tokens at/after EOS."
            ),
        }
        (audit_dir / "debug_step0.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[AUDIT] saved debug_step0.json (seq_len={seq_len}, loss_positions={len(loss_positions)})", flush=True)

    start_time = time.time()
    for step in range(1, max_steps + 1):
        step_start = time.time()
        start_idx = (step - 1) * prompt_batch_size
        batch_rows = train_rows[start_idx : start_idx + prompt_batch_size]
        current_batch = len(batch_rows)
        prompt_texts = build_prompt_texts(tokenizer, batch_rows)
        inputs = tokenizer(prompt_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_prompt_len).to(device)
        prompt_len = inputs["input_ids"].shape[1]
        expanded = {k: v.repeat_interleave(num_gen, dim=0) for k, v in inputs.items()}

        # --- Rollout -----------------------------------------------------------------
        model.eval()
        with torch.inference_mode():
            generated = model.generate(
                **expanded,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=pad_id,
                use_cache=True,
            )
        model.train()
        completion_ids = generated[:, prompt_len:]
        completion_texts = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        # Per-row completion length: tokens before the first pad token.
        non_pad = completion_ids != pad_id
        comp_len = non_pad.sum(dim=1)  # [B]
        if comp_len.min().item() == 0:
            # Safety: keep at least the first token so the loss mask is non-empty.
            comp_len = comp_len.clamp(min=1)

        # --- Old log probs (rollout policy, detached) -------------------------------
        full_ids = torch.cat([inputs["input_ids"], completion_ids], dim=1)
        prompt_attn = inputs["attention_mask"]
        comp_attn = non_pad.float()
        full_attn = torch.cat([prompt_attn, comp_attn], dim=1)
        seq_len = full_ids.shape[1]
        token_mask = torch.zeros_like(full_ids, dtype=torch.bool)
        batch_size_flat = full_ids.shape[0]
        for row_idx in range(batch_size_flat):
            length = int(comp_len[row_idx])
            token_mask[row_idx, prompt_len : prompt_len + length] = True
        old_lp, loss_mask = compute_log_probs(model, full_ids, full_attn, token_mask, use_grad=False)
        old_lp = old_lp.detach()

        # --- Rewards + group advantages ----------------------------------------------
        rollout_meta = []
        for index, row in enumerate(batch_rows):
            for g in range(num_gen):
                flat = index * num_gen + g
                meta = parse_reward(completion_texts[flat], row, reward_cfg)
                meta["completion_text"] = completion_texts[flat]
                meta["completion_length"] = int(comp_len[flat])
                meta["row_index"] = index
                rollout_meta.append(meta)
        rewards_flat = torch.tensor([m["reward"] for m in rollout_meta], dtype=torch.float32, device=device)
        rewards_2d = rewards_flat.view(current_batch, num_gen)
        adv, group_mean, group_std, zero_var = group_advantages(rewards_2d)
        adv_flat = adv.reshape(-1)

        # --- New log probs (current policy, with grad) + loss -------------------------
        new_lp, loss_mask = compute_log_probs(model, full_ids, full_attn, token_mask, use_grad=True)
        ratio = torch.exp(new_lp - old_lp)
        advantage_tokens = adv_flat.unsqueeze(1).expand_as(ratio)
        clipped = ratio.clamp(1.0 - clip_range, 1.0 + clip_range)
        surrogate = -torch.minimum(ratio * advantage_tokens, clipped * advantage_tokens)
        masked = surrogate[loss_mask]
        policy_loss = masked.mean()

        optimizer.zero_grad()
        policy_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
        optimizer.step()

        # --- Step metrics --------------------------------------------------------------
        with torch.no_grad():
            ratio_all = ratio[loss_mask]
            clip_fraction = float((ratio_all != ratio_all.clamp(1.0 - clip_range, 1.0 + clip_range)).float().mean())
            approx_kl = float(((old_lp - new_lp) * loss_mask).sum() / loss_mask.sum())
            ratio_mean = float(ratio_all.mean())
        group_flags = {
            "mixed": sum(any(r > 0.5 for r in row_rs) and any(r < 0.5 for r in row_rs) for row_rs in rewards_2d.tolist()),
            "all_correct": sum(all(r > 0.5 for r in row_rs) for row_rs in rewards_2d.tolist()),
            "all_wrong": sum(all(r < 0.5 for r in row_rs) for row_rs in rewards_2d.tolist()),
            "zero_var": sum(zero_var),
        }
        n_groups = current_batch
        unique_counts = []
        for index in range(current_batch):
            parsed = {canonical_answer(m["parsed_answer"]) for m in rollout_meta if m["row_index"] == index and m["parsed_answer"] is not None}
            unique_counts.append(len(parsed))
        avg_completion_length = sum(m["completion_length"] for m in rollout_meta) / len(rollout_meta)
        format_valid_ratio = sum(m["format_valid"] for m in rollout_meta) / len(rollout_meta)
        parse_success_ratio = sum(m["parse_success"] for m in rollout_meta) / len(rollout_meta)
        metric = {
            "step": step,
            "reward_mean": float(rewards_flat.mean()),
            "reward_std": float(rewards_flat.std()),
            "group_reward_mean_mean": float(group_mean.mean()),
            "mixed_group_ratio": group_flags["mixed"] / n_groups,
            "zero_variance_group_ratio": group_flags["zero_var"] / n_groups,
            "all_wrong_ratio": group_flags["all_wrong"] / n_groups,
            "all_correct_ratio": group_flags["all_correct"] / n_groups,
            "avg_correct_per_group": float(rewards_2d.sum(dim=1).float().mean()),
            "avg_unique_answers": sum(unique_counts) / n_groups,
            "policy_loss": float(policy_loss.detach()),
            "grad_norm": float(grad_norm),
            "completion_length": avg_completion_length,
            "format_valid_ratio": format_valid_ratio,
            "parse_success_ratio": parse_success_ratio,
            "peak_memory_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0,
            "ratio_mean": ratio_mean,
            "clip_fraction": clip_fraction,
            "sampled_approx_kl": approx_kl,
            "seconds_per_step": time.time() - step_start,
        }
        with train_metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metric, ensure_ascii=False) + "\n")
        if step % log_every == 0 or step == max_steps:
            print(
                f"[step {step}/{max_steps}] reward={metric['reward_mean']:.3f}±{metric['reward_std']:.3f} "
                f"mixed={metric['mixed_group_ratio']:.3f} all-correct={metric['all_correct_ratio']:.3f} "
                f"all-wrong={metric['all_wrong_ratio']:.3f} loss={metric['policy_loss']:.4f} "
                f"grad={metric['grad_norm']:.3f} ratio={metric['ratio_mean']:.3f} clip={metric['clip_fraction']:.4f} "
                f"kl={metric['sampled_approx_kl']:.4f} len={metric['completion_length']:.1f} "
                f"mem={metric['peak_memory_allocated_gb']:.1f}GB ({time.time() - start_time:.0f}s)",
                flush=True,
            )

        if not audit_dumped:
            dump_audit_sample(batch_rows[0], prompt_texts[0], completion_texts[0], full_ids, token_mask, 0)
            audit_dumped = True

        if step % eval_every == 0:
            save_checkpoint(step)
            run_val(step)
            if not args.no_probe and step in (100, 300, 500):
                run_probe(step)

    # --- Final step ------------------------------------------------------------------
    save_checkpoint(max_steps)
    run_val(max_steps)
    if not args.no_probe:
        run_probe(max_steps)

    print(f"total_time_s={time.time() - start_time:.0f}", flush=True)
    print(f"best_val_exact={best['exact']:.4f} at step {best['step']} checkpoint={best['checkpoint']}", flush=True)


if __name__ == "__main__":
    main()
