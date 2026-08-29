import pytest
import torch

from kk_sft.logic import KNAVE, KNIGHT
from kk_sft.reward import EXACT, PARTIAL, compute_reward, trl_group_advantages

PEOPLE = ["Alice", "Bob", "Carol"]

# Ground truth: Alice = knight, Bob = knave, Carol = knight  ->  pattern "KNK"
GROUND_TRUTH = {"Alice": KNIGHT, "Bob": KNAVE, "Carol": KNIGHT}


def from_pattern(pattern: str) -> dict[str, str]:
    return {person: (KNIGHT if flag == "K" else KNAVE) for person, flag in zip(PEOPLE, pattern)}


def test_partial_reward_matches_spec():
    """KNK = 1.0, KNN = 2/3, NNK = 2/3, NNN = 1/3, NKN = 0."""
    expected = {"KNK": 1.0, "KNN": 2 / 3, "NNK": 2 / 3, "NNN": 1 / 3, "NKN": 0.0}
    for pattern, value in expected.items():
        assert compute_reward(from_pattern(pattern), GROUND_TRUTH, PEOPLE, PARTIAL) == pytest.approx(value)


def test_exact_reward_is_binary():
    assert compute_reward(from_pattern("KNK"), GROUND_TRUTH, PEOPLE, EXACT) == 1.0
    for pattern in ("KNN", "NNK", "NNN", "NKN"):
        assert compute_reward(from_pattern(pattern), GROUND_TRUTH, PEOPLE, EXACT) == 0.0


def test_default_mode_is_exact():
    assert compute_reward(from_pattern("KNK"), GROUND_TRUTH, PEOPLE) == 1.0
    assert compute_reward(from_pattern("NNN"), GROUND_TRUTH, PEOPLE) == 0.0


def test_parse_failure_is_zero_in_both_modes():
    assert compute_reward(None, GROUND_TRUTH, PEOPLE, EXACT) == 0.0
    assert compute_reward(None, GROUND_TRUTH, PEOPLE, PARTIAL) == 0.0


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        compute_reward(from_pattern("KNK"), GROUND_TRUTH, PEOPLE, "shaped")


def test_all_wrong_group_is_rescued_by_partial_reward():
    """Core H2 mechanism test.

    The group below is exact-all-wrong (std == 0 -> zero advantage), but under the
    partial reward the 8 rollouts differ in how many people they got right, so the
    group regains non-zero variance and a non-zero group-relative advantage.
    """
    patterns = ["KNN", "NNK", "NNN", "NKN", "KNN", "NNN", "NNK", "NKN"]

    exact = [compute_reward(from_pattern(p), GROUND_TRUTH, PEOPLE, EXACT) for p in patterns]
    partial = [compute_reward(from_pattern(p), GROUND_TRUTH, PEOPLE, PARTIAL) for p in patterns]

    assert exact == [0.0] * 8, "group must be exact-all-wrong"

    exact_adv, exact_std = trl_group_advantages(exact)
    partial_adv, partial_std = trl_group_advantages(partial)

    assert exact_std == pytest.approx(0.0), "exact reward gives the group zero variance"
    assert partial_std > 0.0, "partial reward must restore non-zero variance"
    assert torch.allclose(exact_adv, torch.zeros(8), atol=1e-6), "zero-variance group has no training signal"
    assert not torch.allclose(partial_adv, torch.zeros(8), atol=1e-6), "partial reward must produce non-zero advantage"
    assert torch.isfinite(partial_adv).all()

    # best rollouts (2/3) get positive advantage, worst (0) get negative
    assert partial_adv[0] > 0, partial_adv
    assert partial_adv[3] < 0, partial_adv


def test_trl_normalization_matches_unbiased_sample_std():
    rewards = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    _, std = trl_group_advantages(rewards)
    assert std == pytest.approx(torch.tensor(rewards).std(correction=1).item(), rel=1e-6)


def test_zero_variance_group_advantage_is_zero_not_nan():
    adv, std = trl_group_advantages([0.0] * 8)
    assert std == pytest.approx(0.0)
    assert torch.isfinite(adv).all()
    assert torch.allclose(adv, torch.zeros(8), atol=1e-6)
