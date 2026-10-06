"""Small, bounded arithmetic evaluator for the chat workspace; never executes Python."""

import ast
import math
import operator


_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}


def calculate(expression: str) -> str:
    """Evaluate bounded arithmetic using numbers and +, -, *, /, //, %, **.

    Args:
        expression: Arithmetic expression, such as '(25 * 4) + 50'.
    """
    if len(expression) > 512:
        raise ValueError("Expression must be at most 512 characters")
    tree = ast.parse(expression, mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 128:
        raise ValueError("Expression is too complex")

    def checked(value):
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or abs(value) > 10**100
        ):
            raise ValueError("Arithmetic result is outside the supported numeric range")
        return value

    def evaluate(node):
        if isinstance(node, ast.Constant):
            return checked(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = evaluate(node.operand)
            return checked(value if isinstance(node.op, ast.UAdd) else -value)
        if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("Exponent magnitude must be at most 100")
            return checked(_OPERATORS[type(node.op)](left, right))
        raise ValueError("Only numeric literals and arithmetic operators are allowed")

    return str(evaluate(tree.body))
