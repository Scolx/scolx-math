from scolx_math.core.operations import (
    differentiate_expr,
    integrate_expr,
    limit_expr,
    series_expr,
    simplify_expr,
    solve_equation,
)
from scolx_math.explain.explainers import (
    integrate_with_steps,
    limit_with_steps,
    series_with_steps,
)


def test_integrate() -> None:
    result, steps = integrate_with_steps("x**2", "x")
    assert str(result) == "x**3/3"
    assert isinstance(steps, list)
    assert len(steps) > 0


def test_solve_equation() -> None:
    result = solve_equation("x**2 - 4", "x")
    # Solution should be x = -2, 2
    assert len(result) == 2
    assert -2 in result
    assert 2 in result


def test_differentiate() -> None:
    result = differentiate_expr("x**3", "x")
    assert str(result) == "3*x**2"


def test_simplify() -> None:
    result = simplify_expr("(x+1)**2 - x**2 - 2*x - 1")  # Should simplify to 0
    assert str(result) == "0"


def test_integrate_expr() -> None:
    result = integrate_expr("x**2", "x")
    assert str(result) == "x**3/3"


def test_limit() -> None:
    result = limit_expr("sin(x)/x", "x", "0")
    assert str(result) == "1"


def test_series() -> None:
    result = series_expr("exp(x)", "x", "0", 3)
    # This should return the series expansion of e^x around 0 up to order 3
    # which is 1 + x + x^2/2 + O(x^3)
    assert "1 + x + x**2/2" in str(result)


def test_limit_with_steps() -> None:
    result, steps = limit_with_steps("sin(x)/x", "x", "0")
    assert str(result) == "1"
    assert isinstance(steps, list)
    assert len(steps) > 0


def test_series_with_steps() -> None:
    result, steps = series_with_steps("exp(x)", "x", "0", 3)
    assert "1 + x + x**2/2" in str(result)  # Check that it contains the expected terms
    assert isinstance(steps, list)
    assert len(steps) > 0
