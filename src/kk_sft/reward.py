"""Reward modes for GRPO rollouts.

Two modes:

``exact`` (default, GRPO-V1 / GRPO-V2)
    1.0 iff the parsed assignment equals the ground truth, else 0.0.
    Binary and sparse: if all 8 rollouts of a prompt are wrong the group has
    **zero reward variance**, so the group-relative advantage is zero and the
    prompt contributes no gradient at all.

``partial`` (GRPO-V3, H2 intervention)
    ``correct_person_count / len(people)``, i.e. a Hamming-style dense reward.
    Inside an all-wrong group it still separates rollouts by *how many* people
    they got right, so a group whose 8 samples disagree recovers non-zero
    variance — and therefore a non-zero group-relative advantage.

Parse failure / invalid format -> 0.0 in both modes. No format bonus, no length
reward, no reasoning reward, no alpha scaling.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

EXACT = "exact"
PARTIAL = "partial"
MODES = (EXACT, PARTIAL)


def compute_reward(
    parsed_answer: dict[str, str] | None,
    ground_truth: dict[str, str],
    people: Sequence[str],
    mode: str = EXACT,
) -> float:
    """Reward for a single rollout."""
    if mode not in MODES:
        raise ValueError(f"unknown reward mode: {mode!r} (expected one of {MODES})")
    if parsed_answer is None:
        return 0.0
    people = list(people)
    if not people:
        raise ValueError("people must be non-empty")
    if mode == EXACT:
        return float(parsed_answer == ground_truth)
    correct = sum(parsed_answer.get(person) == ground_truth.get(person) for person in people)
    return correct / len(people)


def trl_group_advantages(rewards: Sequence[float], eps: float = 1e-4) -> tuple[torch.Tensor, float]:
    """Group-relative advantages exactly as TRL 0.23 computes them.

    TRL (`grpo_trainer.py`, `scale_rewards="group"`):

        std_rewards = rewards.view(-1, num_generations).std(dim=1)   # torch.std -> unbiased, ddof=1
        advantages  = advantages / (std_rewards + 1e-4)

    TRL does not special-case std == 0; it adds `1e-4` to the denominator, so a
    zero-variance group yields advantage ~0 instead of NaN.

    Returns (advantages, sample_std).
    """
    values = torch.tensor(list(rewards), dtype=torch.float32)
    std = values.std(dim=0, correction=1) if values.numel() > 1 else torch.tensor(0.0)
    advantages = (values - values.mean()) / (std + eps)
    return advantages, float(std)
