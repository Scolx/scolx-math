"""
LaTeX parsing and advanced math operations for the Scolx Math API.

This module would handle:
- LaTeX to SymPy expression conversion
- Advanced mathematical operations
- More sophisticated step-by-step explanations
"""

import sympy as sp
from sympy.parsing.latex import parse_latex

# Note: The sympy.parsing.latex module is still experimental in some versions
# Alternative approach would be to use latex2sympy library if needed


def parse_latex_expression(latex_expr: str) -> sp.Expr:
    """
    Parse a LaTeX mathematical expression into a SymPy expression.

    Args:
        latex_expr: A string containing a LaTeX mathematical expression

    Returns:
        A SymPy expression
    """
    try:
        # Attempt to parse LaTeX using SymPy's experimental LaTeX parser
        return parse_latex(latex_expr)
    except Exception as e:
        # If parsing fails, raise with more context
        raise ValueError(f"Unable to parse LaTeX expression '{latex_expr}': {str(e)}")


def integrate_latex_with_steps(latex_expr: str, var_name: str) -> tuple[str, list[str]]:
    """
    Integrate a LaTeX expression with step-by-step explanation.

    Args:
        latex_expr: A string containing a LaTeX mathematical expression to integrate
        var_name: The variable to integrate with respect to

    Returns:
        A tuple containing the result as a string and a list of explanation steps
    """
    steps = []

    # Parse the LaTeX expression
    steps.append(f"Parsing LaTeX expression: {latex_expr}")
    expr = parse_latex_expression(latex_expr)

    # Create the symbol for integration
    var = sp.Symbol(var_name)
    steps.append(f"Identifying variable of integration: {var_name}")

    # Simplify the expression
    simplified_expr = sp.simplify(expr)
    steps.append(f"Simplifying expression: {simplified_expr}")

    # Perform integration
    result = sp.integrate(simplified_expr, var)
    steps.append(f"Performing integration with respect to {var_name}")
    steps.append("Applying integration rules...")
    steps.append(f"Final result: {result}")

    return str(result), steps


def solve_equation_latex_with_steps(
    latex_eq: str, var_name: str
) -> tuple[list[str], list[str]]:
    """
    Solve a LaTeX equation with step-by-step explanation.

    Args:
        latex_eq: A string containing a LaTeX mathematical equation to solve
        var_name: The variable to solve for

    Returns:
        A tuple containing the solutions as strings and a list of explanation steps
    """
    steps = []

    # Parse the LaTeX expression
    steps.append(f"Parsing LaTeX equation: {latex_eq}")
    expr = parse_latex_expression(latex_eq)

    # Create the symbol to solve for
    var = sp.Symbol(var_name)
    steps.append(f"Identifying variable to solve for: {var_name}")

    # Solve the equation
    solutions = sp.solve(expr, var)
    steps.append(f"Solving equation for {var_name}")
    steps.append("Applying algebraic methods...")
    steps.append(f"Solutions found: {solutions}")

    return [str(sol) for sol in solutions], steps


def differentiate_latex_with_steps(
    latex_expr: str, var_name: str
) -> tuple[str, list[str]]:
    """
    Differentiate a LaTeX expression with step-by-step explanation.

    Args:
        latex_expr: A string containing a LaTeX mathematical expression to differentiate
        var_name: The variable to differentiate with respect to

    Returns:
        A tuple containing the result as a string and a list of explanation steps
    """
    steps = []

    # Parse the LaTeX expression
    steps.append(f"Parsing LaTeX expression: {latex_expr}")
    expr = parse_latex_expression(latex_expr)

    # Create the symbol for differentiation
    var = sp.Symbol(var_name)
    steps.append(f"Identifying variable of differentiation: {var_name}")

    # Differentiate the expression
    result = sp.diff(expr, var)
    steps.append(f"Differentiating expression with respect to {var_name}")
    steps.append("Applying differentiation rules...")
    steps.append(f"Final result: {result}")

    return str(result), steps


def limit_latex_with_steps(
    latex_expr: str, var_name: str, point: str
) -> tuple[str, list[str]]:
    """
    Calculate the limit of a LaTeX expression with step-by-step explanation.

    Args:
        latex_expr: A string containing a LaTeX mathematical expression
        var_name: The variable for the limit
        point: The point at which to calculate the limit

    Returns:
        A tuple containing the result as a string and a list of explanation steps
    """
    steps = []

    # Parse the LaTeX expression
    steps.append(f"Parsing LaTeX expression: {latex_expr}")
    expr = parse_latex_expression(latex_expr)

    # Create the symbol and point
    var = sp.Symbol(var_name)
    pt = parse_latex_expression(
        point
    )  # Point could also be in LaTeX (e.g., "inf" for infinity)
    steps.append(f"Identifying limit variable: {var_name} approaching {point}")

    # Calculate the limit
    result = sp.limit(expr, var, pt)
    steps.append(f"Calculating limit as {var_name} approaches {point}")
    steps.append("Applying limit rules...")
    steps.append(f"Final result: {result}")

    return str(result), steps


def series_latex_with_steps(
    latex_expr: str, var_name: str, point: str = "0", order: int = 6
) -> tuple[str, list[str]]:
    """
    Calculate the series expansion of a LaTeX expression with step-by-step explanation.

    Args:
        latex_expr: A string containing a LaTeX mathematical expression
        var_name: The variable for the series expansion
        point: The point around which to expand (default 0 for Maclaurin series)
        order: Order of the expansion (default 6)

    Returns:
        A tuple containing the result as a string and a list of explanation steps
    """
    steps = []

    # Parse the LaTeX expression
    steps.append(f"Parsing LaTeX expression: {latex_expr}")
    expr = parse_latex_expression(latex_expr)

    # Create the symbol and point
    var = sp.Symbol(var_name)
    pt = parse_latex_expression(point)
    steps.append(f"Identifying series expansion variable: {var_name} around {point}")

    # Calculate the series expansion
    result = sp.series(expr, var, pt, n=order)
    steps.append(f"Calculating series expansion around {point}")
    steps.append(f"Expanding to order {order}...")
    steps.append(f"Final result: {result}")

    return str(result), steps
