"""Helpers for safely parsing user-supplied mathematical expressions."""

from __future__ import annotations

import re
from collections.abc import Iterable

import sympy as sp
from sympy.core.sympify import SympifyError

# Whitelist of supported SymPy functions/constants exposed to user expressions.
_ALLOWED_FUNCTIONS: dict[str, sp.Function | sp.Basic] = {
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "cot": sp.cot,
    "sec": sp.sec,
    "csc": sp.csc,
    "asin": sp.asin,
    "acos": sp.acos,
    "atan": sp.atan,
    "sinh": sp.sinh,
    "cosh": sp.cosh,
    "tanh": sp.tanh,
    "exp": sp.exp,
    "log": sp.log,
    "ln": sp.log,
    "sqrt": sp.sqrt,
    "Abs": sp.Abs,
}
_ALLOWED_CONSTANTS: dict[str, sp.Basic] = {
    "pi": sp.pi,
    "E": sp.E,
    "I": sp.I,
}
_SAFE_LOCALS: dict[str, sp.Function | sp.Basic] = {
    **_ALLOWED_FUNCTIONS,
    **_ALLOWED_CONSTANTS,
}
_SYMBOL_PATTERN = re.compile(r"^[A-Za-z]\w*$")

__all__ = [
    "get_safe_locals",
    "parse_expression",
    "parse_latex_expression",
    "parse_plain_expression",
    "validate_variable_name",
]


def validate_variable_name(name: str) -> str:
    """Ensure a provided variable name is safe to expose to SymPy."""

    if not name or not _SYMBOL_PATTERN.match(name):
        raise ValueError(
            "Invalid variable name. Use alphanumeric characters or underscores, "
            "starting with a letter.",
        )
    return name


def _ensure_safe_symbols(expr: sp.Expr) -> None:
    """Raise if an expression introduces symbols with unsafe names."""

    unsafe = [sym for sym in expr.free_symbols if not _SYMBOL_PATTERN.match(sym.name)]
    if unsafe:
        joined = ", ".join(sorted({sym.name for sym in unsafe}))
        raise ValueError(f"Expression contains invalid symbol names: {joined}")


def parse_plain_expression(
    expr_str: str,
    variables: Iterable[str] | None = None,
) -> sp.Expr:
    """Parse a plain-text expression into a SymPy expression using a safe namespace."""

    if not expr_str or not expr_str.strip():
        raise ValueError("Expression cannot be empty.")

    locals_map: dict[str, sp.Basic | sp.Function] = dict(_SAFE_LOCALS)
    for var in variables or []:
        var_name = validate_variable_name(var)
        locals_map[var_name] = sp.Symbol(var_name)

    try:
        expr = sp.sympify(expr_str, locals=locals_map, convert_xor=True)
    except SympifyError as exc:  # pragma: no cover - defensive branch
        raise ValueError("Invalid mathematical expression.") from exc

    _ensure_safe_symbols(expr)
    return expr


def parse_latex_expression(latex_expr: str) -> sp.Expr:
    """
    Parse a LaTeX mathematical expression into a SymPy expression.

    This function is imported from advanced_latex module to avoid circular imports.
    """
    from scolx_math.advanced_latex import (
        parse_latex_expression as _parse_latex_expression,
    )

    return _parse_latex_expression(latex_expr)


def parse_expression(
    expr_str: str,
    is_latex: bool,
    variables: Iterable[str] | None = None,
) -> sp.Expr:
    """
    Parse an expression that can be either plain text or LaTeX based on the is_latex flag.

    Args:
        expr_str: The expression string to parse
        is_latex: Whether the expression is in LaTeX format
        variables: Optional variables to include in the parsing context

    Returns:
        The parsed SymPy expression
    """
    if is_latex:
        return parse_latex_expression(expr_str)
    return parse_plain_expression(expr_str, variables)


def get_safe_locals() -> dict[str, sp.Function | sp.Basic]:
    """Return a copy of the safe locals dictionary for SymPy sympification."""

    return dict(_SAFE_LOCALS)
