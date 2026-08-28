#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kk_sft.grpo_math import (  # noqa: E402
    approximate_kl,
    clipped_surrogate_loss,
    normalize_group_rewards,
    policy_ratio,
)


def main() -> None:
    rewards = torch.tensor([0.0, 1.0, 0.0, 1.0])
    advantages, stats = normalize_group_rewards(rewards)
    old_log_probs = torch.tensor([-1.0, -1.0, -1.0, -1.0])
    new_log_probs = torch.tensor([-1.1, -0.8, -1.2, -0.9])
    print("rewards:", rewards.tolist())
    print("mean:", stats.mean, "std:", stats.std, "zero_variance:", stats.zero_variance)
    print("advantages:", advantages.tolist())
    print("ratios:", policy_ratio(new_log_probs, old_log_probs).tolist())
    print("approx_kl:", float(approximate_kl(old_log_probs, new_log_probs)))
    print("clipped_surrogate_loss:", float(clipped_surrogate_loss(advantages, new_log_probs, old_log_probs)))


if __name__ == "__main__":
    main()

