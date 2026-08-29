#!/usr/bin/env python3
"""Phase A of H2: zero-training reward-rescue audit.

Re-uses existing rollout dumps (no new sampling, no GPU, no training) and asks one
question:

    For a group where all 8 rollouts are *exact-wrong*, does the 3-person partial
    reward restore non-zero reward variance — and therefore a non-zero
    group-relative advantage?

Sources (all pre-existing):
  outputs/grpo_v1_analysis/rollout_A_final200.jsonl   SFT Epoch4, 200 prompts x 8
  outputs/grpo_v1_analysis/rollout_C_final200.jsonl   GRPO-V1,    200 prompts x 8
  outputs/grpo_v2_analysis/rollout_D_final200.jsonl   GRPO-V2,    200 prompts x 8  <- primary
  outputs/grpo_v2_kl001/probe_rollouts.json           GRPO-V2 mid-training probes (supplementary)

Writes:
  outputs/grpo_v3_h2_audit/reward_rescue_audit.json
  outputs/grpo_v3_h2_audit/reward_rescue_audit.md
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from kk_sft.reward import EXACT, PARTIAL, compute_reward, trl_group_advantages  # noqa: E402

OUT_DIR = ROOT / "outputs" / "grpo_v3_h2_audit"
PEOPLE = ["Alice", "Bob", "Carol"]

# Variance / advantage must be *meaningfully* non-zero. Rewards live on {0, 1/3, 2/3, 1},
# so any group whose rollouts genuinely disagree has std >= ~0.12. Float32 noise on an
# all-identical group leaves std ~1e-8, which after TRL's `/(std + 1e-4)` yields an
# advantage of ~1e-4 -- numerically non-zero but no usable training signal.
ZERO = 1e-6
MIN_ADVANTAGE = 1e-3


def pattern_to_assignment(pattern: str) -> dict[str, str]:
    return {person: ("knight" if flag == "K" else "knave") for person, flag in zip(PEOPLE, pattern)}


def load_rollout_audit_file(path: Path) -> list[dict]:
    """Format written by scripts/rollout_audit.py (A / C / D)."""
    groups = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        people = list(row["ground_truth"].keys())
        groups.append(
            {
                "id": row["id"],
                "source_prompt": row["id"],
                "ground_truth": row["ground_truth"],
                "ground_truth_pattern": row["ground_truth_pattern"],
                "people": people,
                "parsed": [gen["parsed_answer"] for gen in row["generations"]],
            }
        )
    return groups


def load_probe_file(path: Path) -> list[dict]:
    """Format written by the training callback (probe_rollouts.json)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    groups = []
    for block in data["groups"]:
        step = block["probe_step"]
        for prompt in block["prompts"]:
            groups.append(
                {
                    "id": f"{prompt['id']}@step{step}",
                    "source_prompt": prompt["id"],
                    "ground_truth": pattern_to_assignment(prompt["ground_truth_pattern"]),
                    "ground_truth_pattern": prompt["ground_truth_pattern"],
                    "people": list(PEOPLE),
                    "parsed": [
                        pattern_to_assignment(rollout["pattern"]) if rollout["parse_success"] else None
                        for rollout in prompt["rollouts"]
                    ],
                }
            )
    return groups


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if n == 0:
        return (None, None)
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def analyse(groups: list[dict]) -> dict:
    per_group = []
    for group in groups:
        people = group["people"]
        gt = group["ground_truth"]
        exact = [compute_reward(p, gt, people, EXACT) for p in group["parsed"]]
        partial = [compute_reward(p, gt, people, PARTIAL) for p in group["parsed"]]

        exact_t = torch.tensor(exact, dtype=torch.float32)
        partial_t = torch.tensor(partial, dtype=torch.float32)
        exact_std = float(exact_t.std(correction=1))
        partial_std = float(partial_t.std(correction=1))

        exact_all_wrong = all(r == 0.0 for r in exact)
        exact_all_correct = all(r == 1.0 for r in exact)
        # "Rescued" = the partial reward actually separates the rollouts. Count distinct
        # values on rounded floats (exact and robust), then require the resulting std and
        # advantage to be meaningfully non-zero.
        distinct_partial = len({round(r, 6) for r in partial})
        rescued = bool(
            exact_all_wrong and distinct_partial > 1 and partial_std > ZERO and float(trl_group_advantages(partial)[0].abs().max()) > MIN_ADVANTAGE
        )

        adv_exact, _ = trl_group_advantages(exact)
        adv_partial, _ = trl_group_advantages(partial)

        per_group.append(
            {
                "id": group["id"],
                "ground_truth_pattern": group["ground_truth_pattern"],
                "exact_rewards": exact,
                "partial_rewards": [round(r, 6) for r in partial],
                "exact_std": exact_std,
                "partial_std": partial_std,
                "exact_all_wrong": exact_all_wrong,
                "exact_all_correct": exact_all_correct,
                "exact_mixed": bool(any(r == 0.0 for r in exact) and any(r == 1.0 for r in exact)),
                "rescued": rescued,
                "unique_partial_values": distinct_partial,
                "max_abs_advantage_exact": float(adv_exact.abs().max()),
                "max_abs_advantage_partial": float(adv_partial.abs().max()),
                "predicted_patterns": [
                    ("".join("K" if p.get(person) == "knight" else "N" for person in people) if p else "INVALID")
                    for p in group["parsed"]
                ],
            }
        )

    all_wrong = [g for g in per_group if g["exact_all_wrong"]]
    rescued = [g for g in all_wrong if g["rescued"]]
    still_zero = [g for g in all_wrong if not g["rescued"]]

    all_partial = [r for g in per_group for r in g["partial_rewards"]]
    n = len(all_partial)
    partial_mean = sum(all_partial) / n if n else 0.0
    partial_var = sum((r - partial_mean) ** 2 for r in all_partial) / (n - 1) if n > 1 else 0.0
    lo, hi = wilson_ci(len(rescued), len(all_wrong))

    return {
        "n_groups": len(per_group),
        "n_rollouts": n,
        "exact_all_wrong_groups": len(all_wrong),
        "exact_all_correct_groups": sum(g["exact_all_correct"] for g in per_group),
        "exact_mixed_groups": sum(g["exact_mixed"] for g in per_group),
        "exact_zero_variance_groups": sum(g["exact_std"] <= ZERO for g in per_group),
        "partial_zero_variance_groups": sum(g["partial_std"] <= ZERO for g in per_group),
        "rescue": {
            "exact_all_wrong": len(all_wrong),
            "partial_still_zero_variance": len(still_zero),
            "partial_gains_variance": len(rescued),
            "rescue_rate": (len(rescued) / len(all_wrong)) if all_wrong else None,
            "rescue_rate_wilson_95ci": [lo, hi],
        },
        "partial_reward": {
            "mean": partial_mean,
            "std": math.sqrt(partial_var),
            "unique_values": sorted({round(r, 6) for r in all_partial}),
            "value_distribution": dict(sorted(Counter(round(r, 6) for r in all_partial).items(), key=lambda kv: str(kv[0]))),
        },
        "among_exact_all_wrong": {
            "unique_partial_value_distribution": dict(
                sorted(Counter(g["unique_partial_values"] for g in all_wrong).items(), key=lambda kv: str(kv[0]))
            ),
            "rescued_max_abs_advantage_min": min((g["max_abs_advantage_partial"] for g in rescued), default=None),
            "rescued_max_abs_advantage_mean": (
                sum(g["max_abs_advantage_partial"] for g in rescued) / len(rescued) if rescued else None
            ),
        },
        "examples": {
            "rescued": [
                {
                    "id": g["id"],
                    "ground_truth_pattern": g["ground_truth_pattern"],
                    "predicted_patterns": g["predicted_patterns"],
                    "exact_rewards": g["exact_rewards"],
                    "partial_rewards": g["partial_rewards"],
                    "partial_std": g["partial_std"],
                    "max_abs_advantage": g["max_abs_advantage_partial"],
                }
                for g in rescued[:3]
            ],
            "not_rescued": [
                {
                    "id": g["id"],
                    "ground_truth_pattern": g["ground_truth_pattern"],
                    "predicted_patterns": g["predicted_patterns"],
                    "exact_rewards": g["exact_rewards"],
                    "partial_rewards": g["partial_rewards"],
                    "partial_std": g["partial_std"],
                }
                for g in still_zero[:3]
            ],
        },
        "per_group": per_group,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sources = {
        "grpo_v2_200prompt": (load_rollout_audit_file(ROOT / "outputs/grpo_v2_analysis/rollout_D_final200.jsonl"), True),
        "sft_epoch4_200prompt": (load_rollout_audit_file(ROOT / "outputs/grpo_v1_analysis/rollout_A_final200.jsonl"), False),
        "grpo_v1_200prompt": (load_rollout_audit_file(ROOT / "outputs/grpo_v1_analysis/rollout_C_final200.jsonl"), False),
        "grpo_v2_probe_20prompt": (load_probe_file(ROOT / "outputs/grpo_v2_kl001/probe_rollouts.json"), False),
    }

    result = {"sources": {}}
    for name, (groups, _) in sources.items():
        result["sources"][name] = analyse(groups)

    # Gate is read off the primary source: GRPO-V2 on the fixed 200-prompt subset.
    primary = result["sources"]["grpo_v2_200prompt"]
    gate_rate = primary["rescue"]["rescue_rate"]
    result["gate"] = {
        "primary_source": "grpo_v2_200prompt",
        "rescue_rate": gate_rate,
        "threshold_stop_below": 0.10,
        "threshold_continue_above": 0.25,
        "decision": (
            "H2_INTERVENTION_WEAK_STOP"
            if gate_rate is not None and gate_rate < 0.10
            else ("CONTINUE_TO_PHASE_B" if gate_rate is not None and gate_rate > 0.25 else "BORDERLINE_REVIEW")
        ),
    }

    (OUT_DIR / "reward_rescue_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Phase A — reward-rescue audit (zero training)",
        "",
        "Question: for a group whose 8 rollouts are all **exact-wrong**, does the",
        "3-person **partial** reward restore non-zero variance (hence non-zero advantage)?",
        "",
        "Rewards: `exact = 1[parsed == gt]`; `partial = correct_person_count / 3`;",
        "parse failure / invalid -> 0.0 in both modes.",
        "Advantages use TRL 0.23 semantics: `(r - mean) / (sample_std_ddof1 + 1e-4)`.",
        "",
        "## Summary",
        "",
        "| Source | groups | exact all-wrong | partial gains variance | still zero-var | rescue rate | 95% CI |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for name, data in result["sources"].items():
        r = data["rescue"]
        ci = r["rescue_rate_wilson_95ci"]
        ci_text = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if r["rescue_rate"] is not None else "n/a"
        rate = f"{r['rescue_rate'] * 100:.1f}%" if r["rescue_rate"] is not None else "n/a"
        lines.append(
            f"| {name} | {data['n_groups']} | {r['exact_all_wrong']} | {r['partial_gains_variance']} "
            f"| {r['partial_still_zero_variance']} | {rate} | {ci_text} |"
        )

    lines += [
        "",
        "## Primary source detail (GRPO-V2, 200-prompt subset)",
        "",
        f"- Exact all-wrong groups: **{primary['rescue']['exact_all_wrong']}**",
        f"- Among them, partial reward still zero-variance: **{primary['rescue']['partial_still_zero_variance']}**",
        f"- Among them, partial reward gains variance: **{primary['rescue']['partial_gains_variance']}**",
        f"- **reward rescue rate = {primary['rescue']['rescue_rate']:.4f}**",
        "",
        "Zero-variance groups overall (all 8 rollouts identical reward):",
        "",
        f"- exact mode:   {primary['exact_zero_variance_groups']} / {primary['n_groups']}",
        f"- partial mode: {primary['partial_zero_variance_groups']} / {primary['n_groups']}",
        "",
        "Partial reward distribution over all rollouts:",
        "",
        f"- mean = {primary['partial_reward']['mean']:.4f}, std = {primary['partial_reward']['std']:.4f}",
        f"- unique values = {primary['partial_reward']['unique_values']}",
        f"- counts = {primary['partial_reward']['value_distribution']}",
        "",
        "Among the exact all-wrong groups, how many *distinct* partial reward values:",
        "",
        f"- {primary['among_exact_all_wrong']['unique_partial_value_distribution']}",
        "",
        f"- rescued groups: mean max|advantage| = {primary['among_exact_all_wrong']['rescued_max_abs_advantage_mean']}",
        "",
        "## Examples",
        "",
    ]
    for label in ("rescued", "not_rescued"):
        lines.append(f"### {label}")
        lines.append("")
        examples = primary["examples"][label]
        if not examples:
            lines.append("(none)")
            lines.append("")
            continue
        for ex in examples:
            lines.append(f"- `{ex['id']}` gt={ex['ground_truth_pattern']}")
            lines.append(f"  - predictions: {ex['predicted_patterns']}")
            lines.append(f"  - exact:   {ex['exact_rewards']}")
            lines.append(f"  - partial: {[round(v, 4) for v in ex['partial_rewards']]}")
            lines.append(f"  - partial_std = {ex['partial_std']:.6f}")
            if "max_abs_advantage" in ex:
                lines.append(f"  - max|advantage| = {ex['max_abs_advantage']:.4f}")
        lines.append("")

    lines += [
        "## Gate",
        "",
        f"- primary rescue rate = **{gate_rate:.4f}**",
        f"- STOP if < 0.10, continue to Phase B if > 0.25 (engineering gate, not a significance test)",
        f"- decision = **{result['gate']['decision']}**",
        "",
    ]
    (OUT_DIR / "reward_rescue_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
