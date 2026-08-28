from kk_sft.logic import canonical_puzzle_key, eval_expr, generate_puzzle, solve_puzzle


def test_handcrafted_unique_solution():
    puzzle = {
        "people": ["Alice", "Bob", "Carol"],
        "statements": [
            {"speaker": "Alice", "expr": {"op": "person_is", "person": "Bob", "value": "knight"}},
            {"speaker": "Bob", "expr": {"op": "different", "left": "Alice", "right": "Carol"}},
            {"speaker": "Carol", "expr": {"op": "person_is", "person": "Alice", "value": "knight"}},
        ],
    }
    assert solve_puzzle(puzzle) == [{"Alice": "knave", "Bob": "knave", "Carol": "knave"}]


def test_expression_semantics():
    assignment = {"Alice": "knight", "Bob": "knave", "Carol": "knight"}
    assert eval_expr({"op": "same", "left": "Alice", "right": "Carol"}, assignment)
    assert eval_expr({"op": "different", "left": "Alice", "right": "Bob"}, assignment)
    assert not eval_expr({"op": "person_is", "person": "Bob", "value": "knight"}, assignment)


def test_generation_is_deterministic_and_unique():
    first = generate_puzzle(1234)
    second = generate_puzzle(1234)
    assert first == second
    assert len(solve_puzzle(first)) == 1
    assert canonical_puzzle_key(first) == canonical_puzzle_key(second)

