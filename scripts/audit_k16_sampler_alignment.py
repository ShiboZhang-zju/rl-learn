#!/usr/bin/env python3
"""Sampler alignment gate for the K=8 -> K=16 intervention.

Before training, verify (without any model inference) that changing `num_generations`
from 8 to 16 keeps:

  - the same unique prompt order for at least the first 100 optimizer steps
    (= 800 unique prompts),
  - the same number of unique prompts per optimizer step (8),
  - each unique prompt repeated K times consecutively inside a step.

If the unique prompt order differs, the K comparison is confounded by prompt
exposure and must not be run.

Writes outputs/grpo_k16_intervention/audit/sampler_alignment.json
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from kk_sft.data import read_jsonl  # noqa: E402
from trl.trainer.utils import RepeatSampler  # noqa: E402

OUT = ROOT / "outputs" / "grpo_k16_intervention" / "audit"
CONFIG = ROOT / "configs" / "grpo_v2_kl001.yaml"
K8, K16 = 8, 16
STEPS = 100


def load_config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def build(dataset_size: int, k: int, batch_size: int, seed: int, repeat_count: int):
    sampler = RepeatSampler(
        data_source=list(range(dataset_size)),
        mini_repeat_count=k,
        batch_size=batch_size,
        repeat_count=repeat_count,
        shuffle=True,
        seed=seed,
    )
    order = list(sampler)
    # group into optimizer steps: per_device_train_batch_size = batch_size * k rows
    rows_per_step = batch_size * k
    return [order[i : i + rows_per_step] for i in range(0, len(order), rows_per_step)]


def uniques_in_order(step_rows: list[int]) -> list[int]:
    """Unique prompt indices in first-appearance order (should be 8 per step)."""
    seen: list[int] = []
    for index in step_rows:
        if index not in seen:
            seen.append(index)
    return seen


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    seed = int(cfg["seed"])
    prompt_batch_size = int(cfg["training"]["prompt_batch_size"])
    rows = read_jsonl(ROOT / cfg["train_file"])
    dataset_size = len(rows)

    # generation_batch_size = per_device_train_batch_size * steps_per_generation;
    # the script sets gradient_accumulation_steps = 1 and does not set
    # steps_per_generation, so steps_per_generation defaults to
    # gradient_accumulation_steps = 1, and num_iterations defaults to 1.
    steps_per_generation = 1
    num_iterations = 1
    repeat_count = num_iterations * steps_per_generation
    batch_size = prompt_batch_size  # = generation_batch_size // num_generations

    steps8 = build(dataset_size, K8, batch_size, seed, repeat_count)
    steps16 = build(dataset_size, K16, batch_size, seed, repeat_count)

    uniq8 = [uniques_in_order(step) for step in steps8[:STEPS]]
    uniq16 = [uniques_in_order(step) for step in steps16[:STEPS]]

    same_order = uniq8 == uniq16
    first_mismatch = None
    if not same_order:
        for i, (a, b) in enumerate(zip(uniq8, uniq16)):
            if a != b:
                first_mismatch = {"step": i + 1, "k8": a, "k16": b}
                break

    # structural checks
    def structure_ok(steps, k) -> bool:
        for step in steps[:STEPS]:
            if len(step) != batch_size * k:
                return False
            for start in range(0, len(step), k):
                if len(set(step[start : start + k])) != 1:
                    return False
        return True

    result = {
        "config": str(CONFIG.relative_to(ROOT)),
        "train_file": cfg["train_file"],
        "dataset_size": dataset_size,
        "seed": seed,
        "prompt_batch_size": prompt_batch_size,
        "steps_per_generation": steps_per_generation,
        "num_iterations": num_iterations,
        "repeat_count": repeat_count,
        "unique_prompts_per_step_k8": len(uniq8[0]),
        "unique_prompts_per_step_k16": len(uniq16[0]),
        "completions_per_step_k8": len(steps8[0]),
        "completions_per_step_k16": len(steps16[0]),
        "steps_compared": STEPS,
        "unique_prompts_compared": STEPS * batch_size,
        "same_unique_prompt_order": bool(same_order),
        "first_mismatch": first_mismatch,
        "k8_group_structure_ok": bool(structure_ok(steps8, K8)),
        "k16_group_structure_ok": bool(structure_ok(steps16, K16)),
        "first_20_unique_k8": uniq8[0][:8] + uniq8[1][:8] + uniq8[2][:4],
        "first_20_unique_k16": uniq16[0][:8] + uniq16[1][:8] + uniq16[2][:4],
        "unique_prompts_100_steps_k8": len({i for step in uniq8 for i in step}),
        "unique_prompts_100_steps_k16": len({i for step in uniq16 for i in step}),
    }
    result["verdict"] = (
        "SAMPLER_ALIGNED" if (same_order and result["k8_group_structure_ok"] and result["k16_group_structure_ok"]) else "K_INTERVENTION_BLOCKED_BY_SAMPLER_ALIGNMENT"
    )
    (OUT / "sampler_alignment.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nverdict = {result['verdict']}")


if __name__ == "__main__":
    main()
