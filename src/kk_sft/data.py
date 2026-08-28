"""Dataset serialization and deterministic SFT formatting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .logic import KNIGHT, all_assignments, eval_expr, is_consistent, solve_puzzle

SYSTEM_PROMPT = (
    "You solve Knights and Knaves logic puzzles. "
    "A knight always tells the truth and a knave always lies. "
    "Show a concise verification trace, then output exactly one answer block."
)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def format_puzzle_prompt(puzzle: dict[str, Any]) -> str:
    lines = ["Determine whether each person is a knight or a knave.", ""]
    for statement in puzzle["statements"]:
        lines.append(f"{statement['speaker']} says: {statement['text']}")
    lines.extend(
        [
            "",
            "Return your reasoning inside <reasoning>...</reasoning>.",
            "Then return exactly this format inside <answer>...</answer>:",
            "Alice: knight|knave",
            "Bob: knight|knave",
            "Carol: knight|knave",
        ]
    )
    return "\n".join(lines)


def _assignment_text(assignment: dict[str, str], people: list[str]) -> str:
    return ", ".join(f"{person}={assignment[person]}" for person in people)


def build_reasoning_trace(puzzle: dict[str, Any]) -> str:
    people = list(puzzle["people"])
    lines = ["I check all possible assignments."]
    for assignment in all_assignments(people):
        if is_consistent(puzzle, assignment):
            lines.append(f"- {_assignment_text(assignment, people)}: valid; every statement matches its speaker's type.")
        else:
            mismatches = []
            for statement in puzzle["statements"]:
                actual_truth = eval_expr(statement["expr"], assignment)
                expected_truth = assignment[statement["speaker"]] == KNIGHT
                if actual_truth != expected_truth:
                    mismatches.append(statement["speaker"])
            lines.append(
                f"- {_assignment_text(assignment, people)}: invalid because "
                f"the statement by {', '.join(mismatches)} does not match the speaker's type."
            )
    lines.append("Only one assignment is valid, so it is the answer.")
    return "\n".join(lines)


def format_completion(puzzle: dict[str, Any]) -> str:
    solutions = solve_puzzle(puzzle)
    if len(solutions) != 1:
        raise ValueError(f"Expected one solution, found {len(solutions)}")
    solution = solutions[0]
    reasoning = build_reasoning_trace(puzzle)
    answer_lines = [f"{person}: {solution[person]}" for person in puzzle["people"]]
    return "<reasoning>\n" + reasoning + "\n</reasoning>\n<answer>\n" + "\n".join(answer_lines) + "\n</answer>"


def format_answer_only_prompt(puzzle: dict[str, Any]) -> str:
    lines = ["Determine whether each person is a knight or a knave.", ""]
    for statement in puzzle["statements"]:
        lines.append(f"{statement['speaker']} says: {statement['text']}")
    lines.extend(
        [
            "",
            "Return exactly one answer block inside <answer>...</answer>:",
            "Alice: knight",
            "Bob: knave",
            "Carol: knight",
        ]
    )
    return "\n".join(lines)


def format_answer_only_completion(puzzle: dict[str, Any]) -> str:
    solutions = solve_puzzle(puzzle)
    if len(solutions) != 1:
        raise ValueError(f"Expected one solution, found {len(solutions)}")
    solution = solutions[0]
    answer_lines = [f"{person}: {solution[person]}" for person in puzzle["people"]]
    return "<answer>\n" + "\n".join(answer_lines) + "\n</answer>"


def build_answer_only_sft_example(puzzle: dict[str, Any], example_id: str) -> dict[str, Any]:
    solutions = solve_puzzle(puzzle)
    if len(solutions) != 1:
        raise ValueError("Answer-only SFT examples must contain uniquely solvable puzzles")
    prompt = [
        {"role": "system", "content": "You solve Knights and Knaves logic puzzles. Output only one answer block."},
        {"role": "user", "content": format_answer_only_prompt(puzzle)},
    ]
    completion = [{"role": "assistant", "content": format_answer_only_completion(puzzle)}]
    return {
        "id": example_id,
        "prompt": prompt,
        "completion": completion,
        "messages": prompt + completion,
        "answer": solutions[0],
        "puzzle": puzzle,
    }


def build_sft_example(puzzle: dict[str, Any], example_id: str) -> dict[str, Any]:
    solutions = solve_puzzle(puzzle)
    if len(solutions) != 1:
        raise ValueError("SFT examples must contain uniquely solvable puzzles")
    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": format_puzzle_prompt(puzzle)},
    ]
    completion = [{"role": "assistant", "content": format_completion(puzzle)}]
    return {
        "id": example_id,
        "prompt": prompt,
        "completion": completion,
        "messages": prompt + completion,
        "answer": solutions[0],
        "puzzle": puzzle,
    }

