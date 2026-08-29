#!/usr/bin/env python3
"""GRPO formal run using TRL GRPOTrainer (trl==0.23.0) on a single GPU.

Shared by GRPO-V1 (beta=0.0) and GRPO-V2 (beta=0.01). Everything algorithmic is
driven by the YAML config; `training.beta` is the only variable between the two.

- Initializes from an SFT LoRA adapter (`init_checkpoint`), e.g. SFT Epoch4
  checkpoint-1252.
- Rollout: 8 generations/prompt, temperature 0.8, top_p 0.95, max 64 completion tokens.
- Reward: `reward.mode` selects the training signal.
    exact   (default, GRPO-V1/V2): +1 iff parsed answer == ground truth, else 0.
    partial (GRPO-V3, H2):        correct_person_count / len(people), a Hamming-style
      dense reward. It exists to test whether the binary reward's zero-variance
      all-wrong groups are what blocks correction of hard prompts.
  Regardless of mode, the exact-reward group statistics (mixed / all-correct /
  all-wrong / zero-variance, and how many all-wrong groups the shaped reward
  re-energises) are logged every step, so "all-wrong" always keeps its exact
  meaning and a rising shaped reward can never be mistaken for task improvement.
- scale_rewards="group": advantage = (r - group_mean) / (group_std + 1e-4). TRL
  computes group_std with `torch.std` over each group of `num_generations`
  rollouts, i.e. the *unbiased sample* std (ddof=1), not the population std.
  TRL does not special-case std == 0; it adds 1e-4 to the denominator, so
  zero-variance groups get advantage ~0 instead of NaN (`frac_reward_zero_std`
  reports their frequency).
- beta is read from the config, never hard-coded:
    beta == 0.0 -> no reference model, no KL penalty (GRPO-V1).
    beta  > 0.0 -> an explicit frozen reference model is loaded from
      `reference_checkpoint` and attached to the trainer, so that
          policy init = SFT adapter,  reference = frozen SFT adapter.
      Without this override TRL 0.23 would take the PEFT branch
      (`grpo_trainer.py`: `elif is_peft_model(model): self.ref_model = None`) and
      compute reference log-probs under `disable_adapter()`, i.e. against the raw
      Qwen base model instead of SFT Epoch4, which is not the intended constraint.
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
from kk_sft.reward import EXACT, PARTIAL, compute_reward  # noqa: E402


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


def snapshot_adapter_params(model: torch.nn.Module, limit: int = 4) -> dict[str, list[float]]:
    """Full-tensor copy of the first `limit` LoRA params, used to prove a model moved / did not move."""
    snapshot: dict[str, list[float]] = {}
    for name, param in model.named_parameters():
        if "lora_" in name:
            snapshot[name] = param.detach().to(torch.float32).cpu().flatten().tolist()
            if len(snapshot) >= limit:
                break
    return snapshot


def max_param_delta(before: dict[str, list[float]], after: dict[str, list[float]]) -> float:
    if not before or set(before) != set(after):
        return float("nan")
    worst = 0.0
    for key, values in before.items():
        other = after[key]
        if len(other) != len(values):
            return float("nan")
        worst = max(worst, max((abs(a - b) for a, b in zip(values, other)), default=0.0))
    return worst


def completion_logps(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, completion_len: int) -> torch.Tensor:
    """Per-token log p of each completion token, aligned with TRL's `logits_to_keep` slicing."""
    logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
    sliced = logits[:, -completion_len - 1 : -1].to(torch.float32)
    log_probs = torch.log_softmax(sliced, dim=-1)
    targets = input_ids[:, -completion_len:].unsqueeze(-1)
    return torch.gather(log_probs, 2, targets).squeeze(-1)


def run_reference_audit(
    policy,
    reference,
    tokenizer,
    rows: list[dict],
    device: str,
    beta: float,
    init_checkpoint: str,
    reference_checkpoint: str,
    max_new_tokens: int,
    max_prompt_len: int,
    output_path: Path,
) -> dict:
    """Compare policy-at-initialization against the frozen reference on identical token sequences.

    Both models are `Qwen base + the same SFT adapter`, so the expected result is
    logprob_diff ~ 0 and initial KL ~ 0. Anything else means the reference is not the
    policy's initialization and training must not proceed.
    """
    was_training = policy.training
    policy.eval()
    reference.eval()

    per_prompt = []
    all_diffs: list[torch.Tensor] = []
    all_kl: list[torch.Tensor] = []
    for row in rows:
        prompt_text = tokenizer.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=max_prompt_len).to(device)
        with torch.inference_mode():
            generated = policy.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
        completion_len = generated.shape[1] - inputs["input_ids"].shape[1]
        if completion_len <= 0:
            continue
        attention_mask = torch.ones_like(generated)
        with torch.inference_mode():
            policy_lp = completion_logps(policy, generated, attention_mask, completion_len)[0]
            reference_lp = completion_logps(reference, generated, attention_mask, completion_len)[0]
        diff = (reference_lp - policy_lp).abs()
        # k3 estimator, identical to the one TRL uses for the KL penalty term.
        delta = reference_lp - policy_lp
        kl = torch.exp(delta) - delta - 1.0
        all_diffs.append(diff.detach().float().cpu())
        all_kl.append(kl.detach().float().cpu())
        per_prompt.append(
            {
                "id": row["id"],
                "completion_tokens": int(completion_len),
                "mean_abs_logprob_diff": float(diff.mean()),
                "max_abs_logprob_diff": float(diff.max()),
                "initial_kl": float(kl.mean()),
            }
        )

    all_diffs_t = torch.cat(all_diffs) if all_diffs else torch.zeros(1)
    all_kl_t = torch.cat(all_kl) if all_kl else torch.zeros(1)
    audit = {
        "REFERENCE_MODE": "explicit_sft_epoch4",
        "policy_checkpoint": str(init_checkpoint),
        "reference_checkpoint": str(reference_checkpoint),
        "beta": beta,
        "policy_trainable_params": int(sum(p.numel() for p in policy.parameters() if p.requires_grad)),
        "reference_trainable_params": int(sum(p.numel() for p in reference.parameters() if p.requires_grad)),
        "n_prompts": len(per_prompt),
        "mean_abs_logprob_diff": float(all_diffs_t.mean()),
        "max_abs_logprob_diff": float(all_diffs_t.max()),
        "initial_kl": float(all_kl_t.mean()),
        "prompts": per_prompt,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if was_training:
        policy.train()
    return audit


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
    parser.add_argument("--audit-only", action="store_true", help="Write the reference audit and exit without training.")
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
    beta = float(train_cfg.get("beta", 0.0))
    # reward.mode: "exact" (V1/V2, default) or "partial" (V3). Only the reward changes;
    # the exact-reward group statistics below are always logged so that "all-wrong",
    # "all-correct" and "mixed" keep their exact-reward meaning across rounds.
    reward_mode = str(reward_cfg.get("mode", EXACT))
    if reward_mode not in (EXACT, PARTIAL):
        raise ValueError(f"unknown reward.mode: {reward_mode!r} (expected {EXACT} or {PARTIAL})")

    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer
    from peft import PeftModel

    model_name = cfg["model_name_or_path"]
    init_checkpoint = Path(cfg["init_checkpoint"])
    reference_checkpoint = Path(cfg.get("reference_checkpoint") or cfg["init_checkpoint"])
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = torch.bfloat16 if (bool(train_cfg.get("bf16", False)) and device != "cpu") else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, trust_remote_code=True)

    model = PeftModel.from_pretrained(base_model, init_checkpoint, is_trainable=True)
    model.to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"device={device} dtype={dtype} trainable={n_trainable}/{n_total} ({100 * n_trainable / n_total:.4f}%)", flush=True)
    print(f"prompts={n_prompts} prompt_batch={prompt_batch_size} num_gen={num_gen} batches={total_batches} max_steps={max_steps} per_device_train_batch_size={per_device_train_batch_size}", flush=True)
    print("advantage=group normalization (unbiased sample std, ddof=1) via TRL scale_rewards='group'", flush=True)
    print(f"beta={beta}", flush=True)
    print(f"reward_mode={reward_mode} ({'Hamming-style dense: correct_people/3' if reward_mode == PARTIAL else 'binary: 1 iff exact match'})", flush=True)

    # ------------------------------------------------------------------ reference model
    # TRL 0.23 sets ref_model = None for PEFT models and falls back to disable_adapter(),
    # i.e. the raw base model. We want `reference = frozen SFT init`, so load it explicitly.
    reference_model = None
    if beta != 0.0:
        ref_base = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, trust_remote_code=True)
        reference_model = PeftModel.from_pretrained(ref_base, reference_checkpoint, is_trainable=False)
        reference_model.eval()
        for p in reference_model.parameters():
            p.requires_grad_(False)
        reference_model.to(device)
        print(f"reference=explicit_sft_epoch4 checkpoint={reference_checkpoint} trainable_params={sum(p.numel() for p in reference_model.parameters() if p.requires_grad)}", flush=True)
    else:
        print("reference=None (beta=0.0, no KL penalty)", flush=True)

    # ------------------------------------------------------------------ reward + stats
    stats_buffer: list[dict] = []
    call_counter = [0]
    rescued_examples_seen: list[dict] = []
    rescue_counters = {"all_wrong_groups": 0, "rescued_groups": 0}
    # See the rescue_mechanism_observed check: below this many exact-all-wrong groups the
    # mechanism check is not statistically meaningful and is treated as informational.
    MIN_ALL_WRONG_FOR_MECHANISM_CHECK = 20

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
        # `rewards` is the TRAINING signal: partial (V3) or exact (V1/V2).
        rewards: list[float] = []
        group_parsed: list[list[dict]] = [[] for _ in range(n_groups)]
        group_shaped: list[list[float]] = [[] for _ in range(n_groups)]
        group_exact: list[list[float]] = [[] for _ in range(n_groups)]
        for j, comp in enumerate(completions):
            pi = j // G
            pz = json.loads(puzzle_json[j])
            parsed = parse_answer(to_text(comp), pz["people"])
            reward = compute_reward(parsed.parsed, answer[j], pz["people"], reward_mode)
            rewards.append(reward)
            group_shaped[pi].append(reward)
            group_exact[pi].append(1.0 if parsed.parsed == answer[j] else 0.0)
            group_parsed[pi].append(
                {
                    "reward": reward,
                    "correct": parsed.parsed == answer[j],
                    "format_valid": parsed.format_valid,
                    "parse_success": parsed.parsed is not None,
                    "parsed_answer": parsed.parsed,
                    "pattern": assignment_pattern(parsed.parsed, pz["people"]),
                }
            )
        if n_groups == 0:
            stats_buffer.append({"n_prompts": 0, "n_completions": len(completions)})
            return rewards

        # Group-level state under the EXACT reward. These keep their historical names so
        # V1 / V2 / V3 stay comparable; the shaping must never redefine what "all-wrong"
        # means.
        def all_wrong(group: list[float]) -> bool:
            return all(value < 0.5 for value in group)

        def all_correct(group: list[float]) -> bool:
            return all(value > 0.5 for value in group)

        def mixed(group: list[float]) -> bool:
            return any(value < 0.5 for value in group) and any(value > 0.5 for value in group)

        def zero_var(group: list[float]) -> bool:
            # Rewards live on {0, 1/3, 2/3, 1}; anything below 1e-6 is float noise.
            mean = sum(group) / len(group)
            return sum((value - mean) ** 2 for value in group) / (len(group) - 1) < 1e-12 if len(group) > 1 else True

        exact_all_wrong_groups = [group for group in group_exact if all_wrong(group)]
        rescued = [
            index
            for index, group in enumerate(group_exact)
            if all_wrong(group) and not zero_var(group_shaped[index])
        ]
        # Keep a couple of concrete rescued groups per step as human-checkable evidence
        # that the intervention really fires on real rollouts (H2 mechanism check).
        rescued_examples = []
        for index in rescued[:2]:
            first = index * G
            rescued_examples.append(
                {
                    "group_index": index,
                    "ground_truth": answer[first] if first < len(answer) else None,
                    "predicted_patterns": [entry["pattern"] for entry in group_parsed[index]],
                    "exact_rewards": group_exact[index],
                    "shaped_rewards": [round(value, 6) for value in group_shaped[index]],
                }
            )
        all_shaped = [value for group in group_shaped for value in group]
        shaped_mean = sum(all_shaped) / len(all_shaped)

        stats = {
            "n_prompts": n_groups,
            "n_completions": len(completions),
            "reward_mean": shaped_mean,
            "mixed_group_ratio": sum(mixed(group) for group in group_exact) / n_groups,
            "all_wrong_ratio": sum(all_wrong(group) for group in group_exact) / n_groups,
            "all_correct_ratio": sum(all_correct(group) for group in group_exact) / n_groups,
            "avg_correct_per_group": sum(sum(value for value in group) for group in group_exact) / n_groups,
            "avg_unique_answers": sum(len({canonical_answer(r["parsed_answer"]) for r in group if r["parsed_answer"] is not None}) for group in group_parsed) / n_groups,
            "format_valid_ratio": sum(r["format_valid"] for group in group_parsed for r in group) / len(completions),
            "parse_success_ratio": sum(r["parse_success"] for group in group_parsed for r in group) / len(completions),
            # ---- dual-track: training signal (shaped) vs task metric (exact) ----
            "reward_mode": reward_mode,
            "shaped_reward_mean": shaped_mean,
            "shaped_reward_std": (sum((value - shaped_mean) ** 2 for value in all_shaped) / (len(all_shaped) - 1)) ** 0.5,
            "exact_reward_mean": sum(value for group in group_exact for value in group) / len(completions),
            "exact_mixed_ratio": sum(mixed(group) for group in group_exact) / n_groups,
            "exact_all_correct_ratio": sum(all_correct(group) for group in group_exact) / n_groups,
            "exact_all_wrong_ratio": sum(all_wrong(group) for group in group_exact) / n_groups,
            "shaped_zero_variance_ratio": sum(zero_var(group) for group in group_shaped) / n_groups,
            "exact_zero_variance_ratio": sum(zero_var(group) for group in group_exact) / n_groups,
            "exact_all_wrong_but_shaped_nonzero_variance_ratio": (len(rescued) / len(exact_all_wrong_groups)) if exact_all_wrong_groups else 0.0,
            "n_exact_all_wrong_groups": len(exact_all_wrong_groups),
            "n_rescued_groups": len(rescued),
            "rescued_examples": rescued_examples,
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
        beta=beta,
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

    # Attach the explicit frozen reference. TRL only builds one itself for non-PEFT
    # models, and only consults `self.ref_model` at loss time, so overriding it here is
    # enough to make `reference = frozen SFT init` instead of `disable_adapter() -> base`.
    if reference_model is not None:
        trainer.ref_model = trainer.accelerator.prepare_model(reference_model, evaluation_mode=True)
        print(f"trainer.beta={trainer.beta} trainer.ref_model is not None={trainer.ref_model is not None}", flush=True)
        print(f"reference_trainable_params={sum(p.numel() for p in trainer.ref_model.parameters() if p.requires_grad)}", flush=True)

    # ------------------------------------------------------------------ val / probe / logging
    train_metrics_path = output_dir / "train_metrics.jsonl"
    val_metrics_path = output_dir / "val_metrics.json"
    probe_path = output_dir / "probe_rollouts.json"
    audit_path = output_dir / "audit" / "reference_audit.json"
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

    # ------------------------------------------------------------------ reference audit
    # Must run before any optimizer step: policy-at-init vs frozen reference, same tokens.
    audit: dict = {}
    if beta != 0.0 and trainer.ref_model is not None:
        audit_rows = train_rows[:2]
        audit = run_reference_audit(
            policy=trainer.model,
            reference=trainer.ref_model,
            tokenizer=tokenizer,
            rows=audit_rows,
            device=device,
            beta=beta,
            init_checkpoint=str(init_checkpoint),
            reference_checkpoint=str(reference_checkpoint),
            max_new_tokens=max_new_tokens,
            max_prompt_len=max_prompt_len,
            output_path=audit_path,
        )
        print(
            f"[REFERENCE AUDIT] mode={audit['REFERENCE_MODE']} ref={audit['reference_checkpoint']} "
            f"ref_trainable={audit['reference_trainable_params']} "
            f"mean_abs_logprob_diff={audit['mean_abs_logprob_diff']:.3e} "
            f"max_abs_logprob_diff={audit['max_abs_logprob_diff']:.3e} "
            f"initial_kl={audit['initial_kl']:.3e}",
            flush=True,
        )
        if audit["reference_trainable_params"] != 0:
            print("REFERENCE_AUDIT_FAIL: reference has trainable parameters", flush=True)
            return
        if not (audit["initial_kl"] < 1e-3):
            print(f"REFERENCE_AUDIT_FAIL: initial_kl={audit['initial_kl']:.3e} is not ~0; STOP before training", flush=True)
            return
        print("REFERENCE_AUDIT_PASS", flush=True)
        if args.audit_only:
            return

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

    class GrpoCallback(TrainerCallback):
        # Metrics are recorded in `on_log` rather than `on_step_end`: Trainer flushes the
        # step log *after* `on_step_end`, so reading `state.log_history[-1]` there yields
        # the previous step's metrics (off by one). `on_log` receives this step's dict.
        def on_log(self, args, state, control, logs=None, **kwargs):
            logs = logs or {}
            if "loss" not in logs:  # eval / final summary logs, not an optimizer step
                return
            step = state.global_step
            entry = dict(logs)
            stats = stats_buffer.pop(0) if stats_buffer else {}
            rescue_counters["all_wrong_groups"] += int(stats.get("n_exact_all_wrong_groups") or 0)
            rescue_counters["rescued_groups"] += int(stats.get("n_rescued_groups") or 0)
            for example in stats.get("rescued_examples", []):
                if len(rescued_examples_seen) < 5:
                    rescued_examples_seen.append({"step": step, **example})
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
                    "zero_variance_group_ratio": logs.get("frac_reward_zero_std"),
                    # Dual track: the training reward may be shaped, but the task metric
                    # is always the exact reward. Both are logged every step.
                    "reward_mode": stats.get("reward_mode"),
                    "shaped_reward_mean": stats.get("shaped_reward_mean"),
                    "shaped_reward_std": stats.get("shaped_reward_std"),
                    "exact_reward_mean": stats.get("exact_reward_mean"),
                    "exact_mixed_ratio": stats.get("exact_mixed_ratio"),
                    "exact_all_correct_ratio": stats.get("exact_all_correct_ratio"),
                    "exact_all_wrong_ratio": stats.get("exact_all_wrong_ratio"),
                    "shaped_zero_variance_ratio": stats.get("shaped_zero_variance_ratio"),
                    "exact_zero_variance_ratio": stats.get("exact_zero_variance_ratio"),
                    "exact_all_wrong_but_shaped_nonzero_variance_ratio": stats.get("exact_all_wrong_but_shaped_nonzero_variance_ratio"),
                    "n_exact_all_wrong_groups": stats.get("n_exact_all_wrong_groups"),
                    "n_rescued_groups": stats.get("n_rescued_groups"),
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

    policy_before = snapshot_adapter_params(trainer.model)
    reference_before = snapshot_adapter_params(trainer.ref_model) if trainer.ref_model is not None else {}

    trainer.add_callback(GrpoCallback())
    trainer.train()

    # Final metrics: val at last step, final probe, best checkpoint bookkeeping.
    run_val(max_steps)
    run_probe(max_steps)
    best_path = output_dir / "best_checkpoint.json"
    best_path.write_text(json.dumps(best, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"best_val_exact={best['exact']:.4f} at step {best['step']} checkpoint={best['checkpoint']}", flush=True)

    # ------------------------------------------------------------------ post-run audit
    policy_after = snapshot_adapter_params(trainer.model)
    reference_after = snapshot_adapter_params(trainer.ref_model) if trainer.ref_model is not None else {}
    policy_delta = max_param_delta(policy_before, policy_after)
    reference_delta = max_param_delta(reference_before, reference_after) if reference_before else 0.0

    step_entries = []
    if train_metrics_path.exists():
        step_entries = [json.loads(line) for line in train_metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    kl_values = [float(entry["kl"]) for entry in step_entries if entry.get("kl") is not None]
    loss_values = [float(entry["loss"]) for entry in step_entries if entry.get("loss") is not None]
    grad_values = [float(entry["grad_norm"]) for entry in step_entries if entry.get("grad_norm") is not None]

    def all_finite(values: list[float]) -> bool:
        return all(math.isfinite(value) for value in values)

    expect_kl = beta != 0.0
    ref_trainable = int(sum(p.numel() for p in trainer.ref_model.parameters() if p.requires_grad)) if trainer.ref_model is not None else 0
    audit_record = {
        "REFERENCE_MODE": "explicit_sft_epoch4" if trainer.ref_model is not None else "none",
        "reward_mode": reward_mode,
        "beta": beta,
        "trainer_beta": trainer.beta,
        "reference_checkpoint": str(reference_checkpoint) if trainer.ref_model is not None else None,
        "reference_attached": trainer.ref_model is not None,
        "reference_trainable_params": ref_trainable,
        "policy_trainable_params": int(sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)),
        "steps": len(step_entries),
        "policy_param_delta_max": policy_delta,
        "reference_param_delta_max": reference_delta,
        "kl_logged": bool(kl_values),
        "kl_first": kl_values[0] if kl_values else None,
        "kl_last": kl_values[-1] if kl_values else None,
        "kl_max": max(kl_values) if kl_values else None,
        "kl_all_finite": all_finite(kl_values) if kl_values else None,
        "loss_all_finite": all_finite(loss_values),
        "grad_norm_all_finite": all_finite(grad_values),
        "n_loss_values": len(loss_values),
        # H2 mechanism evidence: real rollout groups that are exact-all-wrong yet regain
        # non-zero variance under the partial reward. Note that a short smoke sees far
        # too few groups to expect one (~16% all-wrong x ~31% rescue ~= 0.8 per 16 groups),
        # so distinguish "mechanism never fired" from "no opportunity to fire".
        "n_exact_all_wrong_groups_seen": rescue_counters["all_wrong_groups"],
        "n_rescued_groups_seen": rescue_counters["rescued_groups"],
        "rescue_mechanism_had_opportunity": rescue_counters["all_wrong_groups"] > 0,
        "rescue_mechanism_checkable": rescue_counters["all_wrong_groups"] >= MIN_ALL_WRONG_FOR_MECHANISM_CHECK,
        "n_rescued_examples_observed": len(rescued_examples_seen),
        "rescued_examples": rescued_examples_seen[:5],
    }
    checks = {
        "beta_matches_config": float(trainer.beta) == float(beta),
        "reference_attached_when_beta_nonzero": (not expect_kl) or trainer.ref_model is not None,
        "reference_frozen": (not expect_kl) or (ref_trainable == 0 and reference_delta == 0.0),
        "policy_updated": policy_delta > 0,
        "kl_logged_when_beta_nonzero": (not expect_kl) or bool(kl_values),
        "kl_nonnegative_and_finite": (not expect_kl) or (all_finite(kl_values) and min(kl_values) >= 0.0),
        "loss_finite": all_finite(loss_values),
        "grad_finite": all_finite(grad_values),
        # Partial mode must demonstrably fire on real rollouts, otherwise the run does
        # not test H2 at all. Only enforced once enough exact-all-wrong groups were seen
        # for the check to mean something: at ~31% rescue rate, 20 all-wrong groups give
        # ~6 expected rescues. A 2-step smoke over 16 prompts sees 0-1 all-wrong groups,
        # so it can only verify plumbing, not the mechanism.
        "rescue_mechanism_observed": (
            reward_mode != PARTIAL
            or rescue_counters["all_wrong_groups"] < MIN_ALL_WRONG_FOR_MECHANISM_CHECK
            or bool(rescued_examples_seen)
        ),
    }
    audit_record["checks"] = checks
    audit_record["AUDIT_PASS"] = all(checks.values())
    (output_dir / "audit.json").write_text(json.dumps(audit_record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"[AUDIT] policy_delta={policy_delta:.3e} reference_delta={reference_delta:.3e} "
        f"kl_logged={audit_record['kl_logged']} kl_first={audit_record['kl_first']} kl_last={audit_record['kl_last']} "
        f"kl_max={audit_record['kl_max']}",
        flush=True,
    )
    for name, ok in checks.items():
        if not ok:
            print(f"[AUDIT] FAILED CHECK: {name}", flush=True)
    if audit_record["AUDIT_PASS"]:
        # GRPO-V1/V2 named this gate KL_SMOKE_PASS; the H2 round names it H2_SMOKE_PASS.
        print("H2_SMOKE_PASS", flush=True)
    else:
        print("AUDIT_FAIL", flush=True)


if __name__ == "__main__":
    main()
