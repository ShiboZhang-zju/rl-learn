from kk_sft.data import build_sft_example, format_completion
from kk_sft.evaluation import aggregate_metrics, assignment_pattern, parse_answer, score_completion
from kk_sft.logic import generate_puzzle


def test_sft_example_has_prompt_completion_and_answer():
    puzzle = generate_puzzle(99)
    example = build_sft_example(puzzle, "kk_test")
    assert example["id"] == "kk_test"
    assert example["prompt"][-1]["role"] == "user"
    assert example["completion"][0]["role"] == "assistant"
    assert "<reasoning>" in format_completion(puzzle)
    assert "<answer>" in format_completion(puzzle)


def test_parser_and_reward():
    text = "<reasoning>check</reasoning>\n<answer>\nAlice: knight\nBob: knave\nCarol: knight\n</answer>"
    answer = {"Alice": "knight", "Bob": "knave", "Carol": "knight"}
    parsed = parse_answer(text, answer)
    assert parsed.format_valid
    assert parsed.parsed == answer
    reward = score_completion(text, answer, answer)
    assert reward.total == 1.1
    assert reward.exact_correct


def test_metrics_empty_and_nonempty():
    assert aggregate_metrics([])["count"] == 0
    metrics = aggregate_metrics(
        [{"correct": True, "format_valid": True, "parsed_answer": {"A": "knight"}, "prediction": "ok"}]
    )
    assert metrics["exact_accuracy"] == 1.0


def test_assignment_pattern_and_distribution():
    assert assignment_pattern({"Alice": "knight", "Bob": "knight", "Carol": "knave"}, ["Alice", "Bob", "Carol"]) == "KKN"
    metrics = aggregate_metrics(
        [
            {
                "correct": True,
                "format_valid": True,
                "parsed_answer": {"A": "knight"},
                "prediction": "ok",
                "ground_truth_pattern": "KKN",
                "prediction_pattern": "KKN",
            },
            {
                "correct": False,
                "format_valid": False,
                "parsed_answer": None,
                "prediction": "bad",
                "ground_truth_pattern": "KNN",
                "prediction_pattern": "invalid",
            },
        ]
    )
    assert metrics["ground_truth_pattern_distribution"] == {"KKN": 1, "KNN": 1}
    assert metrics["prediction_pattern_distribution"] == {"KKN": 1, "invalid": 1}
