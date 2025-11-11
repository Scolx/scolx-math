"""Service layer for mathematical operations."""

from __future__ import annotations

import contextlib
from typing import Any

import sympy as sp

from scolx_math.core.operations import (
    complex_argument_expr,
    complex_conjugate_expr,
    complex_from_polar_expr,
    complex_modulus_expr,
    complex_to_polar_expr,
    generate_plot_points,
    matrix_determinant,
    matrix_inverse,
    matrix_multiply,
    normal_cdf,
    normal_pdf,
    solve_ode,
    solve_system_numeric,
    stats_mean,
    stats_standard_deviation,
    stats_variance,
)
from scolx_math.core.parsing import parse_expression
from scolx_math.core.utils import run_cpu_bound_async
from scolx_math.explain.explainers import (
    differentiate_with_steps,
    integrate_with_steps,
    limit_with_steps,
    series_with_steps,
    solve_with_steps,
)


def _stringify_result(value: object) -> object:
    """Convert SymPy objects to serializable values while checking for undefined results."""

    if isinstance(value, sp.Expr):
        if value in (sp.zoo, sp.oo, -sp.oo, sp.nan):
            raise ValueError("Result is undefined (complex infinity).")
        if value.has(sp.zoo) or value.has(sp.nan):
            raise ValueError("Result contains undefined components.")
        return str(value)

    if isinstance(value, sp.MatrixBase):
        return [[_stringify_result(entry) for entry in row] for row in value.tolist()]

    if isinstance(value, (list, tuple)):
        return [_stringify_result(item) for item in value]

    # Fallback to string conversion and final check
    text = str(value)
    if text.lower() in {"zoo", "nan"} or "zoo" in text or "nan" in text:
        raise ValueError("Result is undefined (complex infinity).")
    return text


class MathOperationService:
    """Service class to handle mathematical operations.

    This service provides a unified interface for performing various mathematical
    operations including integration, differentiation, equation solving, limits,
    series expansions, and simplification. It supports plain-text expressions
    with optional step-by-step explanations.
    """

    @staticmethod
    async def handle_integral(
        expression: str,
        variable: str,
        *,
        steps: bool,
    ) -> dict[str, Any]:
        """Handle integral operations.

        Args:
            expression: Mathematical expression to integrate (plain text)
            variable: Variable to integrate with respect to
            steps: Whether to include step-by-step explanations

        Returns:
            Dictionary containing result string and steps list
        """
        try:
            if steps:
                result, all_steps = await run_cpu_bound_async(
                    integrate_with_steps,
                    expression,
                    variable,
                )
                return {
                    "result": _stringify_result(result),
                    "steps": all_steps if steps else [],
                }

            # For non-step case, parse and use core operation
            expr = parse_expression(expression, [variable])
            var = parse_expression(variable)  # Variable names are always plain text
            se_result = None
            try:
                import symengine as se

                if se is not None:
                    se_expr = se.sympify(str(expr)) if hasattr(se, "sympify") else None
                    se_var = se.Symbol(variable) if hasattr(se, "Symbol") else None
                    if se_expr is not None and se_var is not None:
                        with contextlib.suppress(Exception):
                            se_result = se.integrate(se_expr, se_var)
            except ImportError:
                pass  # SymEngine not available

            if se_result is not None:
                from scolx_math.core.operations import _symengine_to_sympy

                result = _symengine_to_sympy(se_result)
            else:
                result = sp.integrate(expr, var)

            return {"result": _stringify_result(result), "steps": []}
        except Exception as e:
            error_msg = f"Error in integral calculation: {e!s}"
            raise ValueError(error_msg) from e

    @staticmethod
    async def handle_derivative(
        expression: str,
        variable: str,
        *,
        steps: bool,
    ) -> dict[str, Any]:
        """Handle derivative operations.

        Args:
            expression: Mathematical expression to differentiate (plain text)
            variable: Variable to differentiate with respect to
            steps: Whether to include step-by-step explanations

        Returns:
            Dictionary containing result string and steps list
        """
        try:
            if steps:
                result, all_steps = await run_cpu_bound_async(
                    differentiate_with_steps,
                    expression,
                    variable,
                )
                return {"result": _stringify_result(result), "steps": all_steps}

            # For non-step case
            expr = parse_expression(expression, [variable])
            var = parse_expression(variable)

            se_result = None
            try:
                import symengine as se

                if se is not None:
                    se_expr = se.sympify(str(expr)) if hasattr(se, "sympify") else None
                    se_var = se.Symbol(variable) if hasattr(se, "Symbol") else None
                    if se_expr is not None and se_var is not None:
                        with contextlib.suppress(Exception):
                            se_result = se.diff(se_expr, se_var)
            except ImportError:
                pass  # SymEngine not available

            if se_result is not None:
                from scolx_math.core.operations import _symengine_to_sympy

                result = _symengine_to_sympy(se_result)
            else:
                result = sp.diff(expr, var)

            return {"result": _stringify_result(result), "steps": []}
        except Exception as e:
            error_msg = f"Error in derivative calculation: {e!s}"
            raise ValueError(error_msg) from e

    @staticmethod
    async def handle_solve(
        expression: str,
        variable: str,
        *,
        steps: bool,
    ) -> dict[str, Any]:
        """Handle equation solving operations.

        Args:
            expression: Mathematical equation to solve (plain text)
            variable: Variable to solve for
            steps: Whether to include step-by-step explanations

        Returns:
            Dictionary containing result list and steps list
        """
        try:
            if steps:
                result, all_steps = await run_cpu_bound_async(
                    solve_with_steps,
                    expression,
                    variable,
                )
                return {
                    "result": _stringify_result(result),
                    "steps": all_steps,
                }

            # For non-step case
            expr = parse_expression(expression, [variable])
            var = parse_expression(variable)
            equation = sp.Eq(expr, 0)
            result = sp.solve(equation, var)
            return {"result": _stringify_result(result), "steps": []}
        except Exception as e:
            error_msg = f"Error in equation solving: {e!s}"
            raise ValueError(error_msg) from e

    @staticmethod
    async def handle_simplify(
        expression: str,
        *,
        steps: bool,
    ) -> dict[str, Any]:
        """Handle simplification operations.

        Args:
            expression: Mathematical expression to simplify (plain text)
            steps: Whether to include step-by-step explanations

        Returns:
            Dictionary containing result string and steps list
        """
        try:
            if steps:
                from scolx_math.explain.explainers import simplify_with_steps

                result, detailed_steps = await run_cpu_bound_async(
                    simplify_with_steps,
                    expression,
                )
                return {"result": _stringify_result(result), "steps": detailed_steps}

            # For non-step case
            expr = parse_expression(expression)

            se_result = None
            try:
                import symengine as se

                if se is not None:
                    se_expr = se.sympify(str(expr)) if hasattr(se, "sympify") else None
                    if se_expr is not None:
                        with contextlib.suppress(Exception):
                            se_result = se.simplify(se_expr)
            except ImportError:
                pass  # SymEngine not available

            if se_result is not None:
                from scolx_math.core.operations import _symengine_to_sympy

                result = _symengine_to_sympy(se_result)
            else:
                result = sp.simplify(expr)

            return {"result": _stringify_result(result), "steps": []}
        except Exception as e:
            error_msg = f"Error in simplification: {e!s}"
            raise ValueError(error_msg) from e

    @staticmethod
    async def handle_limit(
        expression: str,
        variable: str,
        point: str,
        *,
        steps: bool,
    ) -> dict[str, Any]:
        """Handle limit operations.

        Args:
            expression: Mathematical expression for limit (plain text)
            variable: Variable for the limit
            point: Point at which to calculate the limit
            steps: Whether to include step-by-step explanations

        Returns:
            Dictionary containing result string and steps list
        """
        try:
            if steps:
                result, all_steps = await run_cpu_bound_async(
                    limit_with_steps,
                    expression,
                    variable,
                    point,
                )
                return {
                    "result": _stringify_result(result),
                    "steps": all_steps if steps else [],
                }

            # For non-step case
            expr = parse_expression(expression, [variable])
            var = parse_expression(variable)
            pt = parse_expression(point)

            result = sp.limit(expr, var, pt)
            return {"result": _stringify_result(result), "steps": []}
        except Exception as e:
            error_msg = f"Error in limit calculation: {e!s}"
            raise ValueError(error_msg) from e

    @staticmethod
    async def handle_series(
        expression: str,
        variable: str,
        point: str,
        order: int,
        *,
        steps: bool,
    ) -> dict[str, Any]:
        """Handle series operations.

        Args:
            expression: Mathematical expression for series expansion (plain text)
            variable: Variable for the series expansion
            point: Point around which to expand (default 0 for Maclaurin series)
            order: Order of the expansion
            steps: Whether to include step-by-step explanations

        Returns:
            Dictionary containing result string and steps list
        """
        try:
            if steps:
                result, all_steps = await run_cpu_bound_async(
                    series_with_steps,
                    expression,
                    variable,
                    point,
                    order,
                )
                return {
                    "result": _stringify_result(result),
                    "steps": all_steps if steps else [],
                }

            # For non-step case
            expr = parse_expression(expression, [variable])
            var = parse_expression(variable)
            pt = parse_expression(point)

            result = sp.series(expr, var, pt, n=order)
            return {"result": _stringify_result(result), "steps": []}
        except Exception as e:
            error_msg = f"Error in series calculation: {e!s}"
            raise ValueError(error_msg) from e

    @staticmethod
    async def handle_gradient(
        expression: str,
        variables: list[str],
        *,
        steps: bool,
    ) -> dict[str, Any]:
        """Handle gradient calculations for multivariate expressions.

        Args:
            expression: Mathematical expression for gradient (plain text)
            variables: Variables to differentiate with respect to
            steps: Whether to include step-by-step explanations

        Returns:
            Dictionary containing result list and steps list
        """
        try:
            # Parse the expression
            expr = parse_expression(expression, variables)
            sym_vars = [sp.Symbol(var) for var in variables]

            # Calculate gradient
            gradient = [sp.diff(expr, sym_var) for sym_var in sym_vars]

            result = [str(component) for component in gradient]

            all_steps = (
                [
                    f"Computing gradient for expression: {expression}",
                    f"Variables: {', '.join(variables)}",
                    f"Result: {result}",
                ]
                if steps
                else []
            )
            return {"result": result, "steps": all_steps}
        except Exception as e:  # pragma: no cover - defensive
            error_msg = f"Error in gradient calculation: {e!s}"
            raise ValueError(error_msg) from e

    @staticmethod
    async def handle_ode(
        equation: str,
        function: str,
        variable: str,
        initial_conditions: dict[str, str] | None,
        *,
        numeric: bool,
        numeric_start: str | None,
        numeric_end: str | None,
        samples: int,
    ) -> dict[str, Any]:
        """Solve an ordinary differential equation."""
        try:
            solution = await run_cpu_bound_async(
                solve_ode,
                equation,  # Pass original string; core.solve_ode handles parsing safely
                function,
                variable,
                initial_conditions,
                numeric=numeric,
                numeric_start=numeric_start,
                numeric_end=numeric_end,
                samples=samples,
            )
        except Exception as e:  # pragma: no cover - defensive
            error_msg = f"Error while solving differential equation: {e!s}"
            raise ValueError(error_msg) from e

        if isinstance(solution, list):
            return {"points": solution}
        return {"result": str(solution)}

    @staticmethod
    async def handle_matrix_determinant(matrix: list[list[object]]) -> dict[str, Any]:
        """Compute determinant of a matrix."""

        try:
            det_value = await run_cpu_bound_async(matrix_determinant, matrix)
        except Exception as e:  # pragma: no cover - defensive
            error_msg = f"Error in determinant calculation: {e!s}"
            raise ValueError(error_msg) from e

        return {"result": _stringify_result(det_value), "steps": []}

    @staticmethod
    async def handle_matrix_inverse(matrix: list[list[object]]) -> dict[str, Any]:
        """Compute the inverse of a matrix."""

        try:
            inverse_matrix = await run_cpu_bound_async(matrix_inverse, matrix)
        except Exception as e:  # pragma: no cover - defensive
            error_msg = f"Error in matrix inversion: {e!s}"
            raise ValueError(error_msg) from e

        result = [[str(entry) for entry in row] for row in inverse_matrix.tolist()]
        return {"result": result, "steps": []}

    @staticmethod
    async def handle_matrix_multiply(
        left: list[list[object]],
        right: list[list[object]],
    ) -> dict[str, Any]:
        """Multiply two matrices."""

        try:
            product = await run_cpu_bound_async(matrix_multiply, left, right)
        except Exception as e:  # pragma: no cover - defensive
            error_msg = f"Error in matrix multiplication: {e!s}"
            raise ValueError(error_msg) from e

        result = [[str(entry) for entry in row] for row in product.tolist()]
        return {"result": result, "steps": []}

    @staticmethod
    async def handle_complex_conjugate(
        expression: str,
    ) -> dict[str, Any]:
        """Handle complex conjugate."""
        try:
            # Parse the expression
            expr = parse_expression(expression)
            result = await run_cpu_bound_async(complex_conjugate_expr, str(expr))
        except Exception as e:  # pragma: no cover - defensive
            error_msg = f"Error computing complex conjugate: {e!s}"
            raise ValueError(error_msg) from e
        return {"result": _stringify_result(result)}

    @staticmethod
    async def handle_complex_modulus(
        expression: str,
    ) -> dict[str, Any]:
        """Handle complex modulus."""
        try:
            # Parse the expression
            expr = parse_expression(expression)
            result = await run_cpu_bound_async(complex_modulus_expr, str(expr))
        except Exception as e:  # pragma: no cover - defensive
            error_msg = f"Error computing complex modulus: {e!s}"
            raise ValueError(error_msg) from e
        return {"result": _stringify_result(result)}

    @staticmethod
    async def handle_complex_argument(
        expression: str,
    ) -> dict[str, Any]:
        """Handle complex argument."""
        try:
            # Parse the expression
            expr = parse_expression(expression)
            result = await run_cpu_bound_async(complex_argument_expr, str(expr))
        except Exception as e:  # pragma: no cover - defensive
            error_msg = f"Error computing complex argument: {e!s}"
            raise ValueError(error_msg) from e
        return {"result": _stringify_result(result)}

    @staticmethod
    async def handle_complex_to_polar(
        expression: str,
    ) -> dict[str, Any]:
        """Handle complex to polar conversion."""
        try:
            # Parse the expression
            expr = parse_expression(expression)
            magnitude, angle = await run_cpu_bound_async(
                complex_to_polar_expr,
                str(expr),
            )
        except Exception as e:  # pragma: no cover - defensive
            error_msg = f"Error converting to polar form: {e!s}"
            raise ValueError(error_msg) from e
        return {
            "magnitude": _stringify_result(magnitude),
            "angle": _stringify_result(angle),
        }

    @staticmethod
    async def handle_complex_from_polar(
        radius: str,
        angle: str,
    ) -> dict[str, Any]:
        """Handle complex from polar conversion."""
        try:
            # Parse the radius and angle
            radius_expr = parse_expression(radius)
            angle_expr = parse_expression(angle)
            result = await run_cpu_bound_async(
                complex_from_polar_expr,
                str(radius_expr),
                str(angle_expr),
            )
        except Exception as e:  # pragma: no cover - defensive
            error_msg = f"Error converting from polar form: {e!s}"
            raise ValueError(error_msg) from e
        return {"result": _stringify_result(result)}

    @staticmethod
    async def handle_stats_mean(values: list[object]) -> dict[str, Any]:
        try:
            result = await run_cpu_bound_async(stats_mean, values)
        except Exception as e:  # pragma: no cover - defensive
            error_msg = f"Error computing mean: {e!s}"
            raise ValueError(error_msg) from e
        return {"result": _stringify_result(result)}

    @staticmethod
    async def handle_stats_variance(
        values: list[object],
        *,
        sample: bool,
    ) -> dict[str, Any]:
        try:
            result = await run_cpu_bound_async(stats_variance, values, sample=sample)
        except Exception as e:  # pragma: no cover - defensive
            error_msg = f"Error computing variance: {e!s}"
            raise ValueError(error_msg) from e
        return {"result": _stringify_result(result)}

    @staticmethod
    async def handle_stats_stddev(
        values: list[object],
        *,
        sample: bool,
    ) -> dict[str, Any]:
        try:
            result = await run_cpu_bound_async(
                stats_standard_deviation,
                values,
                sample=sample,
            )
        except Exception as e:  # pragma: no cover - defensive
            error_msg = f"Error computing standard deviation: {e!s}"
            raise ValueError(error_msg) from e
        return {"result": _stringify_result(result)}

    @staticmethod
    async def handle_normal_pdf(
        value: str,
        mean: str,
        std: str,
    ) -> dict[str, Any]:
        try:
            # Parse the value, mean, and std
            value_expr = parse_expression(value)
            mean_expr = parse_expression(mean)
            std_expr = parse_expression(std)
            result = await run_cpu_bound_async(
                normal_pdf,
                str(value_expr),
                str(mean_expr),
                str(std_expr),
            )
        except Exception as e:  # pragma: no cover - defensive
            error_msg = f"Error computing normal PDF: {e!s}"
            raise ValueError(error_msg) from e
        return {"result": _stringify_result(result)}

    @staticmethod
    async def handle_normal_cdf(
        value: str,
        mean: str,
        std: str,
    ) -> dict[str, Any]:
        try:
            # Parse the value, mean, and std
            value_expr = parse_expression(value)
            mean_expr = parse_expression(mean)
            std_expr = parse_expression(std)
            result = await run_cpu_bound_async(
                normal_cdf,
                str(value_expr),
                str(mean_expr),
                str(std_expr),
            )
        except Exception as e:  # pragma: no cover - defensive
            error_msg = f"Error computing normal CDF: {e!s}"
            raise ValueError(error_msg) from e
        return {"result": _stringify_result(result)}

    @staticmethod
    async def handle_numeric_solve(
        equations: list[str],
        variables: list[str],
        initial_guess: list[str | float] | None,
        max_iterations: int,
        tolerance: float,
    ) -> dict[str, Any]:
        try:
            solution = await run_cpu_bound_async(
                solve_system_numeric,
                equations,
                variables,
                initial_guess,
                max_iterations,
                tolerance,
            )
        except Exception as e:  # pragma: no cover - defensive
            error_msg = f"Error solving numeric system: {e!s}"
            raise ValueError(error_msg) from e

        return {"solution": solution}

    @staticmethod
    async def handle_hessian(
        expression: str,
        variables: list[str],
        *,
        steps: bool,
    ) -> dict[str, Any]:
        """Handle Hessian matrix calculations for multivariate expressions.

        Args:
            expression: Mathematical expression for Hessian (plain text)
            variables: Variables for partial derivatives
            steps: Whether to include step-by-step explanations

        Returns:
            Dictionary containing result matrix and steps list
        """
        try:
            # Parse the expression
            expr = parse_expression(expression, variables)
            sym_vars = [sp.Symbol(var) for var in variables]

            # Calculate Hessian
            hessian_matrix = sp.hessian(expr, sym_vars)

            result = [[str(entry) for entry in row] for row in hessian_matrix.tolist()]

            all_steps = (
                [
                    f"Computing Hessian for expression: {expression}",
                    f"Variables: {', '.join(variables)}",
                    f"Result: {result}",
                ]
                if steps
                else []
            )
            return {"result": result, "steps": all_steps}
        except Exception as e:  # pragma: no cover - defensive
            error_msg = f"Error in Hessian calculation: {e!s}"
            raise ValueError(error_msg) from e

    @staticmethod
    async def handle_plot(
        expression: str,
        variable: str,
        start: str,
        end: str,
        samples: int,
    ) -> dict[str, Any]:
        """Sample an expression over a range for plotting."""

        try:
            # Parse all expressions
            parsed_expr = parse_expression(expression, [variable])
            start_expr = parse_expression(start, [variable])
            end_expr = parse_expression(end, [variable])

            points = await run_cpu_bound_async(
                generate_plot_points,
                str(parsed_expr),
                variable,
                str(start_expr),
                str(end_expr),
                samples,
            )
        except Exception as e:
            error_msg = f"Error while generating plot data: {e!s}"
            raise ValueError(error_msg) from e

        return {"points": points}
