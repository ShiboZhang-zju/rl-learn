"""Knights & Knaves expressions, exact solver, and procedural generator.

The generator deliberately keeps the semantic representation alongside the
natural-language statement. This makes every generated example auditable and
lets the verifier avoid brittle language parsing.
"""

from __future__ import annotations

import itertools
import json
import random
from typing import Any, Iterable

KNIGHT = "knight"
KNAVE = "knave"
TYPES = (KNIGHT, KNAVE)


def person_is(person: str, value: str) -> dict[str, Any]:
    return {"op": "person_is", "person": person, "value": value}


def same_type(left: str, right: str) -> dict[str, Any]:
    return {"op": "same", "left": left, "right": right}


def different_type(left: str, right: str) -> dict[str, Any]:
    return {"op": "different", "left": left, "right": right}


def negate(expr: dict[str, Any]) -> dict[str, Any]:
    return {"op": "not", "expr": expr}


def conjunction(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    return {"op": "and", "left": left, "right": right}


def disjunction(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    return {"op": "or", "left": left, "right": right}


def eval_expr(expr: dict[str, Any], assignment: dict[str, str]) -> bool:
    """Evaluate a structured proposition under a knight/knave assignment."""

    op = expr["op"]
    if op == "person_is":
        return assignment[expr["person"]] == expr["value"]
    if op == "same":
        return assignment[expr["left"]] == assignment[expr["right"]]
    if op == "different":
        return assignment[expr["left"]] != assignment[expr["right"]]
    if op == "not":
        return not eval_expr(expr["expr"], assignment)
    if op == "and":
        return eval_expr(expr["left"], assignment) and eval_expr(expr["right"], assignment)
    if op == "or":
        return eval_expr(expr["left"], assignment) or eval_expr(expr["right"], assignment)
    raise ValueError(f"Unknown expression op: {op}")


def render_expr(expr: dict[str, Any]) -> str:
    """Render an expression into a stable, readable English sentence."""

    op = expr["op"]
    if op == "person_is":
        return f"{expr['person']} is a {expr['value']}"
    if op == "same":
        return f"{expr['left']} and {expr['right']} are the same type"
    if op == "different":
        return f"{expr['left']} and {expr['right']} are different types"
    if op == "not":
        return f"it is not true that {render_expr(expr['expr'])}"
    if op == "and":
        return f"both ({render_expr(expr['left'])}) and ({render_expr(expr['right'])})"
    if op == "or":
        return f"either ({render_expr(expr['left'])}) or ({render_expr(expr['right'])})"
    raise ValueError(f"Unknown expression op: {op}")


def all_assignments(people: list[str]) -> Iterable[dict[str, str]]:
    for values in itertools.product(TYPES, repeat=len(people)):
        yield dict(zip(people, values))


def is_consistent(puzzle: dict[str, Any], assignment: dict[str, str]) -> bool:
    for statement in puzzle["statements"]:
        statement_true = eval_expr(statement["expr"], assignment)
        speaker_is_knight = assignment[statement["speaker"]] == KNIGHT
        if statement_true != speaker_is_knight:
            return False
    return True


def solve_puzzle(puzzle: dict[str, Any]) -> list[dict[str, str]]:
    """Return every assignment satisfying the Knights & Knaves constraints."""

    people = list(puzzle["people"])
    return [assignment for assignment in all_assignments(people) if is_consistent(puzzle, assignment)]


def canonical_puzzle_key(puzzle: dict[str, Any]) -> str:
    payload = {
        "people": puzzle["people"],
        "statements": [
            {"speaker": item["speaker"], "expr": item["expr"]} for item in puzzle["statements"]
        ],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _random_atom(rng: random.Random, people: list[str]) -> dict[str, Any]:
    choice = rng.choice(("person_is", "same", "different"))
    if choice == "person_is":
        return person_is(rng.choice(people), rng.choice(TYPES))
    left, right = rng.sample(people, 2)
    return same_type(left, right) if choice == "same" else different_type(left, right)


def _random_expr(rng: random.Random, people: list[str], depth: int) -> dict[str, Any]:
    if depth <= 0 or rng.random() < 0.60:
        return _random_atom(rng, people)
    op = rng.choice(("not", "and", "or"))
    if op == "not":
        return negate(_random_expr(rng, people, depth - 1))
    left = _random_expr(rng, people, depth - 1)
    right = _random_expr(rng, people, depth - 1)
    return conjunction(left, right) if op == "and" else disjunction(left, right)


def _build_candidate_statement(
    rng: random.Random,
    speaker: str,
    people: list[str],
    target: dict[str, str],
    max_depth: int,
) -> dict[str, Any]:
    wanted_truth = target[speaker] == KNIGHT
    seen: set[str] = set()
    for _ in range(500):
        expr = _random_expr(rng, people, max_depth)
        expr_key = json.dumps(expr, sort_keys=True, separators=(",", ":"))
        if expr_key in seen:
            continue
        seen.add(expr_key)
        if eval_expr(expr, target) == wanted_truth:
            return {"speaker": speaker, "expr": expr, "text": render_expr(expr) + "."}
    raise RuntimeError("Could not find a statement consistent with the target assignment")


def generate_puzzle_v2(
    seed: int,
    people: list[str] | None = None,
    max_depth: int = 2,
    max_attempts: int = 10_000,
) -> dict[str, Any]:
    """Generate a unique puzzle by sampling statements before solving it.

    Unlike v1, no target assignment is sampled or used while constructing the
    statements. The exact solver determines the label after a candidate puzzle
    has been assembled.
    """
    people = people or ["Alice", "Bob", "Carol"]
    rng = random.Random(seed)
    for _ in range(max_attempts):
        statements = []
        for speaker in people:
            expr = _random_expr(rng, people, max_depth)
            statements.append({"speaker": speaker, "expr": expr, "text": render_expr(expr) + "."})
        puzzle = {"people": people, "statements": statements}
        solutions = solve_puzzle(puzzle)
        if len(solutions) == 1:
            return {
                "people": people,
                "statements": statements,
                "answer": solutions[0],
                "solution_count": len(solutions),
                "metadata": {
                    "generator_version": "kk-v2-statement-first",
                    "dataset_version": "kk-v2",
                    "seed": seed,
                    "max_depth": max_depth,
                    "generation_mode": "statement-first",
                },
            }
    raise RuntimeError(f"Failed to generate a unique v2 puzzle after {max_attempts} attempts")


def generate_puzzle(
    seed: int,
    people: list[str] | None = None,
    max_depth: int = 2,
    max_attempts: int = 10_000,
) -> dict[str, Any]:
    """Generate one uniquely solvable puzzle deterministically from ``seed``."""

    people = people or ["Alice", "Bob", "Carol"]
    rng = random.Random(seed)
    for _ in range(max_attempts):
        target = {person: rng.choice(TYPES) for person in people}
        statements = [
            _build_candidate_statement(rng, speaker, people, target, max_depth) for speaker in people
        ]
        puzzle = {"people": people, "statements": statements}
        solutions = solve_puzzle(puzzle)
        if len(solutions) == 1 and solutions[0] == target:
            return {
                "people": people,
                "statements": statements,
                "answer": target,
                "solution_count": 1,
                "metadata": {"generator_version": "kk-v1", "seed": seed, "max_depth": max_depth},
            }
    raise RuntimeError(f"Failed to generate a unique puzzle after {max_attempts} attempts")

