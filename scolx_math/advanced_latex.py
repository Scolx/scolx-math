"""
LaTeX parsing and advanced math operations for the Scolx Math API.

This module would handle:
- LaTeX to SymPy expression conversion
- Advanced mathematical operations
- More sophisticated step-by-step explanations
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import sympy as sp
from sympy.parsing.latex import parse_latex

from scolx_math.core.parsing import validate_variable_name

_LATEX_SIMPLE_REPLACEMENTS: dict[str, str] = {
    "\\,": "",
    "\\!": "",
    "\\;": "",
    "\\:": "",
    "\\tfrac": "\\frac",
    "\\dfrac": "\\frac",
    "\\cdot": "*",
    "\\times": "*",
    "\\div": "/",
    "\\ln": "\\log",
}

_LATEX_REGEX_REPLACEMENTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\\left\s*"), ""),
    (re.compile(r"\\right\s*"), ""),
    (re.compile(r"\\mathrm\{\s*([A-Za-z])\s*\}"), r"\1"),
    (re.compile(r"\\operatorname\{\s*([A-Za-z0-9_]+)\s*\}"), r"\1"),
)


@dataclass(frozen=True)
class _ParsingAttemptError:
    """Carry context about a failed LaTeX parsing attempt."""

    message: str


def parse_latex_expression(latex_expr: str) -> sp.Expr:
    """
    Parse a LaTeX mathematical expression into a SymPy expression.

    Args:
        latex_expr: A string containing a LaTeX mathematical expression

    Returns:
        A SymPy expression

    Raises:
        ValueError: If the LaTeX expression cannot be parsed safely
    """

    if not latex_expr or not latex_expr.strip():
        raise ValueError("LaTeX expression cannot be empty.")

    if latex_expr.count("{") != latex_expr.count("}"):
        raise ValueError("LaTeX expression has unmatched braces.")

    attempts = []
    seen: set[str] = set()
    last_error: Exception | None = None

    for candidate in _candidate_latex_inputs(latex_expr):
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            expr = parse_latex(candidate)
        except Exception as exc:  # pragma: no cover - defensive branch
            last_error = exc
            attempts.append(_ParsingAttemptError(str(exc)))
            continue

        _ensure_safe_symbols(expr)
        return expr

    base_message = "Unable to parse LaTeX expression."
    if attempts:
        details = " ".join(
            f"Attempt {idx + 1} failed: {attempt.message}"
            for idx, attempt in enumerate(attempts)
        )
        base_message = f"{base_message} {details}".strip()

    if last_error is None:
        raise ValueError(base_message)
    raise ValueError(base_message) from last_error


def _candidate_latex_inputs(original_expr: str) -> list[str]:
    """Return sanitized LaTeX variants to improve parsing robustness."""

    candidates = [original_expr.strip()]
    normalized = _normalize_latex_expression(original_expr)
    if normalized and normalized != candidates[0]:
        candidates.append(normalized)
    return [candidate for candidate in candidates if candidate]


def _normalize_latex_expression(latex_expr: str) -> str:
    """Normalize LaTeX syntax that frequently fails SymPy's parser."""

    expr = latex_expr.strip()
    for pattern, replacement in _LATEX_REGEX_REPLACEMENTS:
        expr = pattern.sub(replacement, expr)
    for source, target in _LATEX_SIMPLE_REPLACEMENTS.items():
        expr = expr.replace(source, target)
    return expr


def _ensure_safe_symbols(expr: sp.Expr) -> None:
    """Assert that parsed expressions only contain safe symbol names."""

    for symbol in expr.free_symbols:
        validate_variable_name(symbol.name)


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
