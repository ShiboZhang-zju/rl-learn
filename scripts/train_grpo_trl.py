#!/usr/bin/env python3
"""GRPO-v1 formal run using TRL GRPOTrainer (trl==0.23.0) on a single GPU.

Matches the GRPO-v1 spec:
- Initializes from SFT checkpoint-1252 (Epoch4) via an existing LoRA adapter.
- Rollout: 8 generations/prompt, temperature 0.8, top_p 0.95, max 64 completion tokens.
- Reward: exact-answer verifier, +1 iff parsed answer == ground truth, else 0.
- scale_rewards="group": per-group normalization with population std (TRL zeroes
  the advantage when group std == 0 internally, so no NaN).
- beta=0.0: no reference model, no KL penalty.
- No vLLM / DeepSpeed / FSDP / format / length / entropy rewards.

Per-step logs come from TRL's log_history plus group statistics collected inside
the reward function. Validation (greedy, V2 Val, 500 prompts) and 20-prompt probes
run inside a callback every `eval_every` optimizer steps.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kk_sft.data import read_jsonl  # noqa: E402
from kk_sft.evaluation import aggregate_metrics, assignment_pattern, parse_answer  # noqa: E402


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def build_prompt_texts(tokenizer, rows: list[dict]) -> list[str]:
    return [tokenizer.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True) for row in rows]


def greedy_eval(model, tokenizer, rows: list[dict], device: str, batch_size: int, max_new_tokens: int, max_prompt_length: int) -> dict:
    records: list[dict] = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        prompt_texts = build_prompt_texts(tokenizer, batch)
        inputs = tokenizer(prompt_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_prompt_length).to(device)
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
    parser.add_argument("--no-val", action="store_true")
    parser.add_argument("--no-probe", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_cfg = {**cfg.get("training", {})}
    gen_cfg = {**cfg.get("generation", {})}
    reward_cfg = {**cfg.get("reward", {})}
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 42))
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

    train_file = args.train_file or Path(cfg["train_file"])
    val_file = args.val_file or Path(cfg["val_file"])
    output_dir = args.output_dir or Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = read_jsonl(train_file)
    if args.train_limit:
        train_rows = train_rows[: args.train_limit]
    val_rows = [] if args.no_val else read_jsonl(val_file)

    n_prompts = len(train_rows)
    prompt_batch_size = int(train_cfg.get("prompt_batch_size", 8))
    num_gen = int(gen_cfg.get("num_generations", 8))
    # TRL 0.23 semantics: RepeatSampler produces batches of
    #   batch_size = generation_batch_size // num_generations  unique prompts,
    # each repeated num_generations times. generation_batch_size =
    # per_device_train_batch_size * steps_per_generation. So to get
    # `prompt_batch_size` unique prompts per optimizer step we set
    # per_device_train_batch_size = prompt_batch_size * num_generations.
    per_device_train_batch_size = prompt_batch_size * num_gen
    total_batches = math.ceil(n_prompts / prompt_batch_size)
    epochs = float(train_cfg.get("epochs", 1))
    max_steps = args.max_steps if args.max_steps is not None else int(train_cfg.get("max_steps", -1))
    if max_steps <= 0:
        max_steps = int(total_batches * epochs)
    max_steps = min(max_steps, total_batches)
    eval_every = int(train_cfg.get("eval_every", 100))

    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    model_name = cfg["model_name_or_path"]
    init_checkpoint = Path(cfg["init_checkpoint"])
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = torch.bfloat16 if (bool(train_cfg.get("bf16", False)) and device != "cpu") else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, trust_remote_code=True)
    from peft import PeftModel

    model = PeftModel.from_pretrained(base_model, init_checkpoint, is_trainable=True)
    model.to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"device={device} dtype={dtype} trainable={n_trainable}/{n_total} ({100 * n_trainable / n_total:.4f}%)", flush=True)
    print(f"prompts={n_prompts} prompt_batch={prompt_batch_size} num_gen={num_gen} batches={total_batches} max_steps={max_steps} per_device_train_batch_size={per_device_train_batch_size}", flush=True)
    print("advantage=group normalization (population std) via TRL scale_rewards='group'; beta=0.0", flush=True)

    # ------------------------------------------------------------------ reward + stats
    stats_buffer: list[dict] = []
    call_counter = [0]

    def to_text(completion) -> str:
        if isinstance(completion, str):
            return completion
        if isinstance(completion, (list, tuple)) and completion and isinstance(completion[0], dict):
            return "".join(m.get("content", "") for m in completion if isinstance(m, dict))
        return str(completion)

    def reward_fn(prompts, completions, completion_ids=None, answer=None, puzzle_json=None, **kwargs):
        G = num_gen
        # TRL 0.23 RepeatSampler repeats each unique prompt num_generations times
        # consecutively, so completions are ordered prompt-major: group j//G is the
        # unique prompt index. Number of unique prompts in this call:
        n_groups = len(prompts) // G
        rewards: list[float] = []
        group_parsed: list[list[dict]] = [[] for _ in range(n_groups)]
        for j, comp in enumerate(completions):
            pi = j // G
            pz = json.loads(puzzle_json[j])
            parsed = parse_answer(to_text(comp), pz["people"])
            correct = parsed.parsed == answer[j]
            reward = float(reward_cfg.get("correct", 1.0)) if correct else 0.0
            rewards.append(reward)
            group_parsed[pi].append(
                {
                    "reward": reward,
                    "correct": correct,
                    "format_valid": parsed.format_valid,
                    "parse_success": parsed.parsed is not None,
                    "parsed_answer": parsed.parsed,
                    "pattern": assignment_pattern(parsed.parsed, pz["people"]),
                }
            )
        if n_groups == 0:
            stats_buffer.append({"n_prompts": 0, "n_completions": len(completions)})
            return rewards
        stats = {
            "n_prompts": n_groups,
            "n_completions": len(completions),
            "reward_mean": sum(rewards) / len(rewards),
            "mixed_group_ratio": sum(any(r["reward"] > 0.5 for r in group) and any(r["reward"] < 0.5 for r in group) for group in group_parsed) / n_groups,
            "all_wrong_ratio": sum(all(r["reward"] < 0.5 for r in group) for group in group_parsed) / n_groups,
            "all_correct_ratio": sum(all(r["reward"] > 0.5 for r in group) for group in group_parsed) / n_groups,
            "avg_correct_per_group": sum(sum(r["reward"] for r in group) for group in group_parsed) / n_groups,
            "avg_unique_answers": sum(len({canonical_answer(r["parsed_answer"]) for r in group if r["parsed_answer"] is not None}) for group in group_parsed) / n_groups,
            "format_valid_ratio": sum(r["format_valid"] for group in group_parsed for r in group) / len(completions),
            "parse_success_ratio": sum(r["parse_success"] for group in group_parsed for r in group) / len(completions),
        }
        stats_buffer.append(stats)
        call_counter[0] += 1
        return rewards

    # ------------------------------------------------------------------ dataset + config
    # "puzzle" contains nested expressions with heterogeneous shapes that Arrow cannot
    # infer as one struct, so it is passed to the reward function as a JSON string.
    train_dataset = Dataset.from_list(
        [
            {"id": row["id"], "prompt": row["prompt"], "answer": row["answer"], "puzzle_json": json.dumps(row["puzzle"], ensure_ascii=False)}
            for row in train_rows
        ]
    )

    grpo_config = GRPOConfig(
        output_dir=str(output_dir),
        seed=seed,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=1,
        learning_rate=float(train_cfg.get("learning_rate", 1e-5)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        max_grad_norm=float(train_cfg.get("max_grad_norm", 1.0)),
        num_train_epochs=epochs,
        max_steps=max_steps,
        lr_scheduler_type="constant",
        warmup_ratio=0.0,
        logging_steps=1,
        save_strategy="steps",
        save_steps=eval_every,
        save_total_limit=None,
        bf16=bool(train_cfg.get("bf16", False)),
        gradient_checkpointing=False,
        beta=0.0,
        num_generations=num_gen,
        max_prompt_length=int(gen_cfg.get("max_prompt_length", 512)),
        max_completion_length=int(gen_cfg.get("max_completion_length", 64)),
        temperature=float(gen_cfg.get("temperature", 0.8)),
        top_p=float(gen_cfg.get("top_p", 0.95)),
        use_vllm=False,
        scale_rewards="group",
        disable_dropout=True,
        report_to="none",
        remove_unused_columns=False,
        optim="adamw_torch",
        dataloader_num_workers=0,
        dataloader_drop_last=False,
        disable_tqdm=False,
        log_on_each_node=False,
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        reward_funcs=[reward_fn],
    )

    # ------------------------------------------------------------------ val / probe / logging
    train_metrics_path = output_dir / "grpo_v1_train_metrics.jsonl"
    val_metrics_path = output_dir / "grpo_v1_val_metrics.json"
    probe_path = output_dir / "grpo_v1_probe_rollouts.json"
    val_metrics: list[dict] = []
    probe_records: list[dict] = []
    best = {"step": -1, "exact": -1.0, "checkpoint": None}
    probe_seed_meta = {}
    probe_rows: list[dict] = []
    if not args.no_probe and val_rows:
        import random

        probe_rng = random.Random(20260830)
        probe_indices = probe_rng.sample(range(len(val_rows)), 20)
        probe_rows = [val_rows[i] for i in probe_indices]
        probe_seed_meta = {"seed": 20260830, "indices": probe_indices, "count": len(probe_rows)}

    max_new_tokens = int(gen_cfg.get("max_completion_length", 64))
    max_prompt_len = int(gen_cfg.get("max_prompt_length", 512))
    temperature = float(gen_cfg.get("temperature", 0.8))
    top_p = float(gen_cfg.get("top_p", 0.95))

    def run_probe(step: int) -> None:
        if not probe_rows:
            return
        trl_model = trainer.model
        was_training = trl_model.training
        trl_model.eval()
        prompts = build_prompt_texts(tokenizer, probe_rows)
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_prompt_len).to(device)
        expanded = {k: v.repeat_interleave(num_gen, dim=0) for k, v in inputs.items()}
        with torch.inference_mode():
            generated = trl_model.generate(
                **expanded,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
        prompt_len = expanded["input_ids"].shape[1]
        texts = tokenizer.batch_decode(generated[:, prompt_len:], skip_special_tokens=True)
        group = {"probe_step": step}
        for index, row in enumerate(probe_rows):
            samples = texts[index * num_gen : (index + 1) * num_gen]
            parsed_list = [parse_answer(text, row["puzzle"]["people"]) for text in samples]
            group.setdefault("prompts", []).append(
                {
                    "id": row["id"],
                    "ground_truth_pattern": assignment_pattern(row["answer"], row["puzzle"]["people"]),
                    "rollouts": [
                        {
                            "text": text,
                            "reward": 1.0 if parsed.parsed == row["answer"] else 0.0,
                            "pattern": assignment_pattern(parsed.parsed, row["puzzle"]["people"]),
                            "format_valid": parsed.format_valid,
                            "parse_success": parsed.parsed is not None,
                        }
                        for text, parsed in zip(samples, parsed_list)
                    ],
                }
            )
        probe_records.append(group)
        probe_path.write_text(json.dumps({"probe_seed_meta": probe_seed_meta, "groups": probe_records}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if was_training:
            trl_model.train()
        print(f"[PROBE step={step}] saved {len(probe_rows)} prompts x {num_gen} rollouts", flush=True)

    def run_val(step: int) -> None:
        if not val_rows:
            return
        trl_model = trainer.model
        was_training = trl_model.training
        trl_model.eval()
        metrics = greedy_eval(trl_model, tokenizer, val_rows, device, 32, max_new_tokens, max_prompt_len)
        if was_training:
            trl_model.train()
        entry = {"step": step, **metrics}
        val_metrics.append(entry)
        val_metrics_path.write_text(json.dumps(val_metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[VAL step={step}] exact={metrics['exact_accuracy']:.4f} format={metrics['format_accuracy']:.4f} parse={metrics['parse_success_rate']:.4f}", flush=True)
        if metrics["exact_accuracy"] > best["exact"]:
            best.update(step=step, exact=metrics["exact_accuracy"])
            best["checkpoint"] = str(output_dir / f"checkpoint-{step}")
            print(f"[BEST] new best val exact {metrics['exact_accuracy']:.4f} at step {step}", flush=True)

    from transformers import TrainerCallback

    do_probe = not args.no_probe

    class GrpoV1Callback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            step = state.global_step
            entry = {}
            if state.log_history:
                entry = dict(state.log_history[-1])
            stats = stats_buffer.pop(0) if stats_buffer else {}
            entry.update(
                {
                    "step": step,
                    "reward_mean": stats.get("reward_mean"),
                    "mixed_group_ratio": stats.get("mixed_group_ratio"),
                    "all_wrong_ratio": stats.get("all_wrong_ratio"),
                    "all_correct_ratio": stats.get("all_correct_ratio"),
                    "avg_correct_per_group": stats.get("avg_correct_per_group"),
                    "avg_unique_answers": stats.get("avg_unique_answers"),
                    "format_valid_ratio": stats.get("format_valid_ratio"),
                    "parse_success_ratio": stats.get("parse_success_ratio"),
                    "zero_variance_group_ratio": entry.get("train/frac_reward_zero_std"),
                    "peak_memory_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0,
                }
            )
            with train_metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            if step % eval_every == 0:
                run_val(step)
                if do_probe and step in (100, 300, 500):
                    run_probe(step)

    # Step-0 baseline (before any update): greedy val + probe.
    if val_rows:
        run_val(0)
    if probe_rows:
        run_probe(0)
    print(f"[baseline] val exact={best['exact']:.4f} (step 0 = SFT Epoch4 ckpt-1252)", flush=True)

    trainer.add_callback(GrpoV1Callback())
    trainer.train()

    # Final metrics: val at last step, final probe, best checkpoint bookkeeping.
    run_val(max_steps)
    run_probe(max_steps)
    best_path = output_dir / "best_checkpoint.json"
    best_path.write_text(json.dumps(best, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"best_val_exact={best['exact']:.4f} at step {best['step']} checkpoint={best['checkpoint']}", flush=True)


if __name__ == "__main__":
    main()
