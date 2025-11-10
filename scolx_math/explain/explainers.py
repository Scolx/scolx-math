"""Step-by-step explanation functions for mathematical operations."""

from sympy import limit as sympy_limit
from sympy import series as sympy_series
from sympy import simplify

from scolx_math.core.operations import integrate_expr
from scolx_math.core.parsing import parse_plain_expression, validate_variable_name


def integrate_with_steps(expr_str: str, var_str: str):
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
    expr = simplify(expr)
    steps.append(
        "Applying integration rules with SymEngine acceleration when available"
    )
    result = integrate_expr(str(expr), var_name)
    steps.append(f"Final result: {result}")
    return result, steps


def limit_with_steps(expr_str: str, var_str: str, point_str: str):
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
        f"Parsing limit expression: lim({var_str} -> {point_str}) of {expr_str}"
    )
    var_name = validate_variable_name(var_str)
    expr = parse_plain_expression(expr_str, variables=[var_name])
    point = parse_plain_expression(point_str, variables=[var_name])
    steps.append(f"Identifying limit variable: {var_str} approaching {point_str}")
    result = sympy_limit(expr, parse_plain_expression(var_name), point)
    steps.append("Applying limit rules...")
    steps.append(f"Final result: {result}")
    return result, steps


def series_with_steps(
    expr_str: str, var_str: str, point_str: str = "0", order: int = 6
):
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
    steps.append(f"Expanding to order {order}...")
    steps.append(f"Final result: {result}")
    return result, steps
