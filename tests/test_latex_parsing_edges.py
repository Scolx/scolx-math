"""Additional tests for LaTeX parsing normalization and edge cases."""

import pytest
import sympy as sp

from scolx_math.advanced_latex import (
    differentiate_latex_with_steps,
    integrate_latex_with_steps,
    limit_latex_with_steps,
    parse_latex_expression,
    series_latex_with_steps,
)


def test_parse_handles_left_right_parentheses() -> None:
    expr = parse_latex_expression(r"\left( x^2 + 1 \right)")
    assert sp.simplify(expr - (sp.Symbol("x") ** 2 + 1)) == 0


def test_parse_handles_tfrac_and_dfrac() -> None:
    expr_t = parse_latex_expression(r"\tfrac{1}{x}")
    expr_d = parse_latex_expression(r"\dfrac{1}{x}")
    expr_n = parse_latex_expression(r"\frac{1}{x}")
    x = sp.Symbol("x")
    assert sp.simplify(expr_t - 1 / x) == 0
    assert sp.simplify(expr_d - 1 / x) == 0
    assert sp.simplify(expr_n - 1 / x) == 0


def test_parse_handles_operatorname_and_mathrm() -> None:
    expr_op = parse_latex_expression(r"\operatorname{sin}(x)")
    expr_rm = parse_latex_expression(r"\mathrm{x}^2 + 1")
    x = sp.Symbol("x")
    assert sp.simplify(expr_op - sp.sin(x)) == 0
    assert sp.simplify(expr_rm - (x**2 + 1)) == 0


def test_parse_handles_multiplication_aliases() -> None:
    expr_cd = parse_latex_expression(r"x \\cdot y")
    expr_tx = parse_latex_expression(r"x \\times y")
    x, y = sp.symbols("x y")
    assert sp.simplify(expr_cd - x * y) == 0
    assert sp.simplify(expr_tx - x * y) == 0


def test_step_functions_validate_variable_names() -> None:
    with pytest.raises(ValueError, match="Invalid variable name"):
        # invalid variable name starting with a digit
        differentiate_latex_with_steps(r"x^2", "1x")


@pytest.mark.parametrize(
    ("latex_expr", "var", "point"),
    [
        (r"\\frac{\\sin(x)}{x}", "x", "0"),
        (r"e^{x}", "x", "0"),
    ],
)
def test_step_functions_accept_valid_inputs(
    latex_expr: str,
    var: str,
    point: str,
) -> None:
    # Ensure these functions don't raise and return strings for result
    res1, _ = integrate_latex_with_steps(r"x^2", var)
    assert isinstance(res1, str)

    res2, _ = differentiate_latex_with_steps(r"x^3", var)
    assert isinstance(res2, str)

    res3, _ = limit_latex_with_steps(latex_expr, var, point)
    assert isinstance(res3, str)

    res4, _ = series_latex_with_steps(r"e^{x}", var, point, 4)
    assert isinstance(res4, str)
