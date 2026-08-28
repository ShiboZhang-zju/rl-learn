"""Shared answer parser, metrics, and verifier-backed reward."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from .logic import KNAVE, KNIGHT

ANSWER_BLOCK_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
ANSWER_LINE_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_-]*)\s*:\s*(knight|knave)\s*$", re.IGNORECASE)


@dataclass
class ParseResult:
    parsed: dict[str, str] | None
    format_valid: bool
    reason: str


@dataclass
class RewardResult:
    total: float
    answer_reward: float
    format_reward: float
    exact_correct: bool
    format_valid: bool
    parsed_answer: dict[str, str] | None
    parse_reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_answer(text: str, people: Iterable[str]) -> ParseResult:
    people = list(people)
    match = ANSWER_BLOCK_RE.search(text)
    if not match:
        return ParseResult(None, False, "missing <answer> block")
    parsed: dict[str, str] = {}
    for raw_line in match.group(1).splitlines():
        if not raw_line.strip():
            continue
        line_match = ANSWER_LINE_RE.match(raw_line)
        if not line_match:
            return ParseResult(None, False, f"invalid answer line: {raw_line.strip()!r}")
        person, value = line_match.groups()
        person = next((known for known in people if known.lower() == person.lower()), person)
        value = value.lower()
        if person not in people:
            return ParseResult(None, False, f"unknown person: {person}")
        if person in parsed:
            return ParseResult(None, False, f"duplicate person: {person}")
        parsed[person] = value
    if set(parsed) != set(people):
        missing = sorted(set(people) - set(parsed))
        return ParseResult(None, False, f"missing people: {missing}")
    if any(value not in (KNIGHT, KNAVE) for value in parsed.values()):
        return ParseResult(None, False, "invalid type")
    return ParseResult(parsed, True, "ok")


def score_completion(
    completion: str,
    people: Iterable[str],
    answer: dict[str, str],
    answer_reward: float = 1.0,
    format_reward: float = 0.1,
) -> RewardResult:
    parsed = parse_answer(completion, people)
    exact_correct = parsed.parsed == answer if parsed.parsed is not None else False
    return RewardResult(
        total=(answer_reward if exact_correct else 0.0) + (format_reward if parsed.format_valid else 0.0),
        answer_reward=answer_reward if exact_correct else 0.0,
        format_reward=format_reward if parsed.format_valid else 0.0,
        exact_correct=exact_correct,
        format_valid=parsed.format_valid,
        parsed_answer=parsed.parsed,
        parse_reason=parsed.reason,
    )


def assignment_pattern(assignment: dict[str, str] | None, people: Iterable[str]) -> str:
    """Return a compact K/N pattern in the stable people order, or invalid."""

    if assignment is None:
        return "invalid"
    return "".join("K" if assignment[person] == KNIGHT else "N" for person in people)


def _has_repeated_line(text: str, repeat_count: int = 3) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for index in range(len(lines) - repeat_count + 1):
        if len(set(lines[index : index + repeat_count])) == 1:
            return True
    return False


def aggregate_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    if total == 0:
        return {
            "count": 0,
            "exact_accuracy": 0.0,
            "format_accuracy": 0.0,
            "parse_success_rate": 0.0,
            "ground_truth_pattern_distribution": {},
            "prediction_pattern_distribution": {},
        }
    ground_truth_patterns = [row["ground_truth_pattern"] for row in records if row.get("ground_truth_pattern")]
    prediction_patterns = [row.get("prediction_pattern", "invalid") for row in records]
    return {
        "count": total,
        "exact_accuracy": sum(bool(row.get("correct")) for row in records) / total,
        "format_accuracy": sum(bool(row.get("format_valid")) for row in records) / total,
        "parse_success_rate": sum(row.get("parsed_answer") is not None for row in records) / total,
        "average_response_chars": sum(len(row.get("prediction", "")) for row in records) / total,
        "repetition_rate": sum(_has_repeated_line(row.get("prediction", "")) for row in records) / total,
        "ground_truth_pattern_distribution": dict(Counter(ground_truth_patterns)),
        "prediction_pattern_distribution": dict(Counter(prediction_patterns)),
    }
