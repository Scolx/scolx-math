"""Step-by-step explanation functions for mathematical operations."""

import sympy as sp
from sympy import Basic, factor, simplify
from sympy import limit as sympy_limit
from sympy import series as sympy_series

from scolx_math.core.operations import (
    differentiate_expr,
    integrate_expr,
    simplify_expr,
    solve_equation,
)
from scolx_math.core.parsing import parse_plain_expression, validate_variable_name


def integrate_with_steps(expr_str: str, var_str: str) -> tuple[Basic, list[str]]:
    """Perform integration with step-by-step explanation.

    Args:
        expr_str: Plain text mathematical expression to integrate
        var_str: Variable to integrate with respect to

    Returns:
        A tuple containing the result and a list of explanation steps
    """
    steps = []
    steps.append("Parsing expression and simplifying")
    var_name = validate_variable_name(var_str)
    expr = parse_plain_expression(expr_str, variables=[var_name])
    simplified_expr = simplify(expr)
    if simplified_expr != expr:
        steps.append(f"Simplified expression: {simplified_expr}")
    else:
        steps.append("Expression is already in simplest form")

    steps.append(
        f"Integrating with respect to {var_name} using SymPy/SymEngine acceleration",
    )
    result = integrate_expr(str(simplified_expr), var_name)
    steps.append(f"Result before simplification: {result}")

    verification = simplify(differentiate_expr(str(result), var_name))
    steps.append(
        f"Verification by differentiation: d/d{var_name} of result = {verification}",
    )
    return result, steps


def limit_with_steps(
    expr_str: str,
    var_str: str,
    point_str: str,
) -> tuple[Basic, list[str]]:
    """Calculate limit with step-by-step explanation.

    Args:
        expr_str: Plain text mathematical expression for limit
        var_str: Variable for the limit
        point_str: Point at which to calculate the limit

    Returns:
        A tuple containing the result and a list of explanation steps
    """
    steps = []
    steps.append(
        f"Parsing limit expression: lim({var_str} -> {point_str}) of {expr_str}",
    )
    var_name = validate_variable_name(var_str)
    var_symbol = sp.Symbol(var_name)
    expr = parse_plain_expression(expr_str, variables=[var_name])
    point = parse_plain_expression(point_str, variables=[var_name])
    steps.append(f"Identifying limit variable: {var_str} approaching {point_str}")

    try:
        direct_substitution = simplify(expr.subs(var_symbol, point))
    except Exception:  # pragma: no cover - defensive
        direct_substitution = None

    is_finite = (
        getattr(direct_substitution, "is_finite", None)
        if direct_substitution is not None
        else None
    )
    if direct_substitution is not None and is_finite is True:
        steps.append(
            f"Direct substitution yields a finite result: {direct_substitution}. Limit equals this value.",
        )
        return direct_substitution, steps

    steps.append(
        "Direct substitution is indeterminate; applying SymPy limit evaluation techniques",
    )
    result = sympy_limit(expr, var_symbol, point)
    steps.append(f"Final result after evaluating the limit: {result}")
    return result, steps


def series_with_steps(
    expr_str: str,
    var_str: str,
    point_str: str = "0",
    order: int = 6,
) -> tuple[Basic, list[str]]:
    """Calculate series expansion with step-by-step explanation.

    Args:
        expr_str: Plain text mathematical expression for series expansion
        var_str: Variable for the series expansion
        point_str: Point around which to expand (default 0 for Maclaurin series)
        order: Order of the expansion (default 6)

    Returns:
        A tuple containing the result and a list of explanation steps
    """
    steps = []
    steps.append(f"Parsing series expansion expression: {expr_str} around {point_str}")
    var_name = validate_variable_name(var_str)
    expr = parse_plain_expression(expr_str, variables=[var_name])
    point = parse_plain_expression(point_str, variables=[var_name])
    steps.append(f"Identifying series expansion variable: {var_str} around {point_str}")
    result = sympy_series(expr, parse_plain_expression(var_name), point, n=order)
    steps.append(
        f"Computing the Taylor series up to order {order} with respect to {var_str}",
    )
    steps.append(f"Series expansion result: {result}")
    return result, steps


def differentiate_with_steps(expr_str: str, var_str: str) -> tuple[Basic, list[str]]:
    steps = []
    steps.append("Parsing expression and simplifying")
    var_name = validate_variable_name(var_str)
    expr = parse_plain_expression(expr_str, variables=[var_name])
    simplified_expr = simplify(expr)
    if simplified_expr != expr:
        steps.append(f"Simplified expression: {simplified_expr}")
    else:
        steps.append("Expression is already in simplest form")

    steps.append(f"Applying derivative with respect to {var_name}")
    derivative = differentiate_expr(str(simplified_expr), var_name)
    steps.append(f"Derivative result: {derivative}")
    return derivative, steps


def simplify_with_steps(expr_str: str) -> tuple[Basic, list[str]]:
    steps = []
    steps.append("Parsing expression for simplification")
    simplified = simplify_expr(expr_str)
    steps.append(f"Applying SymPy simplify -> {simplified}")

    factored = factor(simplified)
    if factored != simplified:
        steps.append(f"Optional factoring gives: {factored}")
        simplified = factored
    steps.append(f"Final simplified form: {simplified}")
    return simplified, steps


def solve_with_steps(expr_str: str, var_str: str) -> tuple[list, list[str]]:
    steps = []
    steps.append("Parsing equation and moving all terms to the left-hand side")
    var_name = validate_variable_name(var_str)
    expr = parse_plain_expression(expr_str, variables=[var_name])
    simplified_expr = simplify(expr)
    if simplified_expr != expr:
        steps.append(f"Simplified equation: {simplified_expr} = 0")
    else:
        steps.append("Equation is already simplified")

    factored = factor(simplified_expr)
    if factored != simplified_expr:
        steps.append(f"Factored form: {factored} = 0")

    steps.append(f"Applying SymPy solve for variable {var_name}")
    solutions = solve_equation(expr_str, var_str)
    if not solutions:
        raise ValueError("Equation has no solutions.")
    steps.append(f"Solutions: {', '.join(str(sol) for sol in solutions)}")
    return solutions, steps
