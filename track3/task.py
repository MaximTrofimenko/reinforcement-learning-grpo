"""Countdown tasks and a strict, safe answer checker."""

from __future__ import annotations

import ast
import itertools
import re
from collections import Counter
from fractions import Fraction


ANSWER_RE = re.compile(r"\s*<answer>\s*([0-9+*/()\s-]+)\s*</answer>\s*", re.DOTALL)


def prompt_for(task: dict) -> str:
    numbers = ", ".join(map(str, task["numbers"]))
    return (
        "Solve this number puzzle. Use each given number exactly once. "
        "You may use +, -, *, / and parentheses. The result must equal the target. "
        "Division by zero is forbidden. Reply with one expression only inside "
        "<answer>...</answer>. No explanation or other text.\n"
        "Example: numbers 1, 2, 3; target 7 -> <answer>1+2*3</answer>\n"
        f"Numbers: {numbers}; target: {task['target']}"
    )


def _evaluate(node: ast.AST, used: list[int]) -> Fraction:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body, used)
    if isinstance(node, ast.Constant) and type(node.value) is int:
        used.append(node.value)
        return Fraction(node.value)
    if isinstance(node, ast.BinOp):
        left = _evaluate(node.left, used)
        right = _evaluate(node.right, used)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise ZeroDivisionError
            return left / right
    raise ValueError("Operation is not allowed")


def check_answer(task: dict, response: str) -> dict:
    match = ANSWER_RE.fullmatch(response)
    if not match:
        return {"format_ok": False, "numbers_ok": False, "correct": False, "reason": "format"}
    expression = match.group(1).strip()
    if len(expression) > 80:
        return {"format_ok": True, "numbers_ok": False, "correct": False, "reason": "length"}
    try:
        tree = ast.parse(expression, mode="eval")
        used: list[int] = []
        value = _evaluate(tree, used)
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError):
        return {"format_ok": True, "numbers_ok": False, "correct": False, "reason": "expression"}
    numbers_ok = Counter(used) == Counter(task["numbers"])
    correct = numbers_ok and value == Fraction(task["target"])
    return {
        "format_ok": True,
        "numbers_ok": numbers_ok,
        "correct": bool(correct),
        "reason": "correct" if correct else "numbers" if not numbers_ok else "target",
    }


def reward(task: dict, response: str) -> float:
    return 1.0 if check_answer(task, response)["correct"] else 0.0


def task_pool() -> list[dict]:
    """Build unique solvable tasks; the generating expression is never stored."""
    tasks = set()
    for numbers in itertools.combinations(range(2, 13), 3):
        for a, b, c in itertools.permutations(numbers):
            targets = (a + b + c, a * b + c, a * b - c, (a + b) * c)
            for target in targets:
                if 5 <= target <= 60 and target not in numbers:
                    tasks.add((numbers, target))
    return [
        {"numbers": list(numbers), "target": target}
        for numbers, target in sorted(tasks)
    ]
