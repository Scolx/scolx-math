"""Plain-text parsing and validation edge cases (consolidated)."""

import pytest
import sympy as sp

from scolx_math.core.parsing import parse_expression, validate_variable_name


def test_parse_basic_arithmetic_parentheses() -> None:
    expr = parse_expression("(x**2 + 1)", ["x"])
    x = sp.Symbol("x")
    assert sp.simplify(expr - (x**2 + 1)) == 0


def test_parse_handles_multiplication_and_division() -> None:
    expr = parse_expression("x*y/2", ["x", "y"])
    x, y = sp.symbols("x y")
    assert sp.simplify(expr - (x * y / 2)) == 0


def test_validate_variable_name_rejects_invalid() -> None:
    with pytest.raises(ValueError, match="Invalid variable name"):
        validate_variable_name("1x")


@pytest.mark.parametrize("expr_str", ["sin(x)/x", "exp(x)", "sqrt(x) + 2"])
def test_parse_common_functions(expr_str: str) -> None:
    expr = parse_expression(expr_str, ["x"])  # Should not raise
    assert isinstance(expr, sp.Expr)
