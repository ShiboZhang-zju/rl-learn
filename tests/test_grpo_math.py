import torch

from kk_sft.grpo_math import normalize_group_rewards, policy_ratio


def test_group_advantage_matches_expected_shape():
    advantages, stats = normalize_group_rewards(torch.tensor([0.0, 1.0, 0.0, 1.0]))
    assert stats.mean == 0.5
    assert round(stats.std, 6) == 0.5
    assert torch.allclose(advantages, torch.tensor([-1.0, 1.0, -1.0, 1.0]), atol=1e-5)


def test_policy_ratio():
    ratio = policy_ratio(torch.tensor([-0.8]), torch.tensor([-1.0]))
    assert torch.allclose(ratio, torch.tensor([torch.exp(torch.tensor(0.2))]))

