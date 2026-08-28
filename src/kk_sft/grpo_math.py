"""Small, framework-independent GRPO calculations for debugging."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class GroupStats:
    mean: float
    std: float
    zero_variance: bool


def normalize_group_rewards(rewards: torch.Tensor, eps: float = 1e-8) -> tuple[torch.Tensor, GroupStats]:
    rewards = rewards.float()
    mean = rewards.mean()
    # std(unbiased=False) is the direct population std for one sampled group.
    std = rewards.std(unbiased=False)
    normalized = (rewards - mean) / (std + eps)
    stats = GroupStats(float(mean), float(std), bool(float(std) < eps))
    return normalized, stats


def policy_ratio(new_log_probs: torch.Tensor, old_log_probs: torch.Tensor) -> torch.Tensor:
    return torch.exp(new_log_probs - old_log_probs)


def clipped_surrogate_loss(
    advantages: torch.Tensor,
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    clip_range: float = 0.2,
) -> torch.Tensor:
    ratio = policy_ratio(new_log_probs, old_log_probs)
    unclipped = ratio * advantages
    clipped = ratio.clamp(1.0 - clip_range, 1.0 + clip_range) * advantages
    return -torch.minimum(unclipped, clipped).mean()


def approximate_kl(old_log_probs: torch.Tensor, new_log_probs: torch.Tensor) -> torch.Tensor:
    return (old_log_probs - new_log_probs).mean()

