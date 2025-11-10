"""Core symbolic operations used by the public API."""

from __future__ import annotations

from sympy import Eq, diff, integrate, limit, series, simplify, solve
from sympy import sympify as sp_sympify

try:  # pragma: no cover - optional acceleration layer
    import symengine as se
except ImportError:  # pragma: no cover - symengine optional
    se = None

from scolx_math.core.parsing import parse_plain_expression, validate_variable_name


def _symengine_symbol(name: str):
    """Create a SymEngine symbol if SymEngine is available."""
    if se is None:
        return None
    try:
        return se.Symbol(name)
    except Exception:  # pragma: no cover - defensive
        return None


def _symengine_expr(expr):
    """Convert expression to SymEngine if available."""
    if se is None:
        return None
    try:
        return se.sympify(expr)
    except Exception:
        return None


def _symengine_to_sympy(result):
    """Convert SymEngine result back to SymPy."""
    if result is None:
        return None
    try:
        if hasattr(result, "sympy"):
            return result.sympy()
        return sp_sympify(str(result))
    except Exception:
        return None


def _symengine_simplify(expr):
    """Simplify expression using SymEngine if available."""
    se_expr = _symengine_expr(expr)
    if se_expr is None:
        return None
    try:
        simplified = se.simplify(se_expr)
    except Exception:
        return None
    return _symengine_to_sympy(simplified)


def _symengine_diff(expr, var_name: str):
    """Differentiate expression using SymEngine if available."""
    se_expr = _symengine_expr(expr)
    if se_expr is None:
        return None
    se_var = _symengine_symbol(var_name)
    if se_var is None:
        return None
    try:
        result = se.diff(se_expr, se_var)
    except Exception:
        return None
    return _symengine_to_sympy(result)


def _symengine_integrate(expr, var_name: str):
    """Integrate expression using SymEngine if available."""
    se_expr = _symengine_expr(expr)
    if se_expr is None:
        return None
    se_var = _symengine_symbol(var_name)
    if se_var is None:
        return None
    try:
        result = se.integrate(se_expr, se_var)
    except Exception:
        return None
    return _symengine_to_sympy(result)


def solve_equation(expr_str: str, var_str: str):
    """Solve an equation for a given variable.

    Args:
        expr_str: Plain text mathematical equation (should equal 0)
        var_str: Variable to solve for

    Returns:
        List of solutions
    """
    var_name = validate_variable_name(var_str)
    x_expr = parse_plain_expression(var_name)
    expr = parse_plain_expression(expr_str, variables=[var_name])
    equation = Eq(expr, 0)
    return solve(equation, x_expr)


def integrate_expr(expr_str: str, var_str: str):
    """Integrate a mathematical expression with respect to a variable.

    Args:
        expr_str: Plain text mathematical expression to integrate
        var_str: Variable to integrate with respect to

    Returns:
        Integrated expression
    """
    var_name = validate_variable_name(var_str)
    var_symbol = parse_plain_expression(var_name)
    expr = parse_plain_expression(expr_str, variables=[var_name])
    se_result = _symengine_integrate(expr, var_name)
    if se_result is not None:
        return se_result
    return integrate(expr, var_symbol)


def differentiate_expr(expr_str: str, var_str: str):
    """Differentiate a mathematical expression with respect to a variable.

    Args:
        expr_str: Plain text mathematical expression to differentiate
        var_str: Variable to differentiate with respect to

    Returns:
        Differentiated expression
    """
    var_name = validate_variable_name(var_str)
    var_symbol = parse_plain_expression(var_name)
    expr = parse_plain_expression(expr_str, variables=[var_name])
    se_result = _symengine_diff(expr, var_name)
    if se_result is not None:
        return se_result
    return diff(expr, var_symbol)


def limit_expr(expr_str: str, var_str: str, point_str: str):
    """Calculate the limit of an expression as a variable approaches a point.

    Args:
        expr_str: Plain text mathematical expression for limit
        var_str: Variable for the limit
        point_str: Point at which to calculate the limit

    Returns:
        Limit value
    """
    var_name = validate_variable_name(var_str)
    var_symbol = parse_plain_expression(var_name)
    expr = parse_plain_expression(expr_str, variables=[var_name])
    point = parse_plain_expression(point_str, variables=[var_name])

    # Limit operations don't use SymEngine as it doesn't support limits
    return limit(expr, var_symbol, point)


def series_expr(expr_str: str, var_str: str, point_str: str = "0", order: int = 6):
    """Calculate the series expansion of an expression around a point.

    Args:
        expr_str: Plain text mathematical expression for series expansion
        var_str: Variable for the series expansion
        point_str: Point around which to expand (default 0 for Maclaurin series)
        order: Order of the expansion (default 6)

    Returns:
        Series expansion
    """
    var_name = validate_variable_name(var_str)
    var_symbol = parse_plain_expression(var_name)
    expr = parse_plain_expression(expr_str, variables=[var_name])
    point = parse_plain_expression(point_str, variables=[var_name])

    # Series operations don't use SymEngine as it doesn't support series
    return series(expr, var_symbol, point, n=order)


def simplify_expr(expr_str: str):
    """Simplify a mathematical expression.

    Args:
        expr_str: Plain text mathematical expression to simplify

    Returns:
        Simplified expression
    """
    expr = parse_plain_expression(expr_str)
    se_result = _symengine_simplify(expr)
    if se_result is not None:
        return se_result
    return simplify(expr)
