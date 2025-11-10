"""Service layer for mathematical operations."""

from __future__ import annotations

from typing import Any

import sympy as sp

from scolx_math.advanced_latex import (
    differentiate_latex_with_steps,
    integrate_latex_with_steps,
    limit_latex_with_steps,
    series_latex_with_steps,
    solve_equation_latex_with_steps,
)
from scolx_math.core.operations import (
    complex_argument_expr,
    complex_conjugate_expr,
    complex_from_polar_expr,
    complex_modulus_expr,
    complex_to_polar_expr,
    differentiate_expr,
    generate_plot_points,
    gradient_expr,
    hessian_expr,
    integrate_expr,
    limit_expr,
    matrix_determinant,
    matrix_inverse,
    matrix_multiply,
    normal_cdf,
    normal_pdf,
    series_expr,
    simplify_expr,
    solve_equation,
    solve_ode,
    solve_system_numeric,
    stats_mean,
    stats_standard_deviation,
    stats_variance,
)
from scolx_math.core.utils import run_cpu_bound_async
from scolx_math.explain.explainers import (
    differentiate_with_steps,
    integrate_with_steps,
    limit_with_steps,
    series_with_steps,
    simplify_with_steps,
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
    series expansions, and simplification. It supports both plain text and LaTeX
    expressions with optional step-by-step explanations.
    """

    @staticmethod
    async def handle_integral_latex(
        expression: str,
        variable: str,
        steps: bool,
    ) -> dict[str, Any]:
        """Handle LaTeX integral operations.

        Args:
            expression: LaTeX mathematical expression to integrate
            variable: Variable to integrate with respect to
            steps: Whether to include step-by-step explanations

        Returns:
            Dictionary containing result string and steps list
        """
        try:
            result, all_steps = await run_cpu_bound_async(
                integrate_latex_with_steps,
                expression,
                variable,
            )
            return {
                "result": _stringify_result(result),
                "steps": all_steps if steps else [],
            }
        except Exception as e:
            error_msg = f"Error in integral calculation: {e!s}"
            raise ValueError(error_msg) from e

    @staticmethod
    async def handle_derivative_latex(
        expression: str,
        variable: str,
        steps: bool,
    ) -> dict[str, Any]:
        """Handle LaTeX derivative operations.

        Args:
            expression: LaTeX mathematical expression to differentiate
            variable: Variable to differentiate with respect to
            steps: Whether to include step-by-step explanations

        Returns:
            Dictionary containing result string and steps list
        """
        try:
            result, all_steps = await run_cpu_bound_async(
                differentiate_latex_with_steps,
                expression,
                variable,
            )
            return {
                "result": _stringify_result(result),
                "steps": all_steps if steps else [],
            }
        except Exception as e:
            error_msg = f"Error in derivative calculation: {e!s}"
            raise ValueError(error_msg) from e

    @staticmethod
    async def handle_solve_latex(
        expression: str,
        variable: str,
        steps: bool,
    ) -> dict[str, Any]:
        """Handle LaTeX equation solving operations.

        Args:
            expression: LaTeX mathematical equation to solve
            variable: Variable to solve for
            steps: Whether to include step-by-step explanations

        Returns:
            Dictionary containing result list and steps list
        """
        try:
            result, all_steps = await run_cpu_bound_async(
                solve_equation_latex_with_steps,
                expression,
                variable,
            )
            return {
                "result": _stringify_result(result),
                "steps": all_steps if steps else [],
            }
        except Exception as e:
            error_msg = f"Error in equation solving: {e!s}"
            raise ValueError(error_msg) from e

    @staticmethod
    async def handle_gradient(
        expression: str,
        variables: list[str],
        steps: bool,
    ) -> dict[str, Any]:
        """Handle gradient calculations for multivariate expressions."""

        try:
            gradient = await run_cpu_bound_async(gradient_expr, expression, variables)
        except Exception as e:  # pragma: no cover - defensive
            error_msg = f"Error in gradient calculation: {e!s}"
            raise ValueError(error_msg) from e

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

    @staticmethod
    async def handle_ode(
        equation: str,
        function: str,
        variable: str,
        initial_conditions: dict[str, str] | None,
        numeric: bool,
        numeric_start: str | None,
        numeric_end: str | None,
        samples: int,
    ) -> dict[str, Any]:
        """Solve an ordinary differential equation."""

        try:
            solution = await run_cpu_bound_async(
                solve_ode,
                equation,
                function,
                variable,
                initial_conditions,
                numeric,
                numeric_start,
                numeric_end,
                samples,
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
    async def handle_complex_conjugate(expression: str) -> dict[str, Any]:
        try:
            result = await run_cpu_bound_async(complex_conjugate_expr, expression)
        except Exception as e:  # pragma: no cover - defensive
            error_msg = f"Error computing complex conjugate: {e!s}"
            raise ValueError(error_msg) from e
        return {"result": _stringify_result(result)}

    @staticmethod
    async def handle_complex_modulus(expression: str) -> dict[str, Any]:
        try:
            result = await run_cpu_bound_async(complex_modulus_expr, expression)
        except Exception as e:  # pragma: no cover - defensive
            error_msg = f"Error computing complex modulus: {e!s}"
            raise ValueError(error_msg) from e
        return {"result": _stringify_result(result)}

    @staticmethod
    async def handle_complex_argument(expression: str) -> dict[str, Any]:
        try:
            result = await run_cpu_bound_async(complex_argument_expr, expression)
        except Exception as e:  # pragma: no cover - defensive
            error_msg = f"Error computing complex argument: {e!s}"
            raise ValueError(error_msg) from e
        return {"result": _stringify_result(result)}

    @staticmethod
    async def handle_complex_to_polar(expression: str) -> dict[str, Any]:
        try:
            magnitude, angle = await run_cpu_bound_async(
                complex_to_polar_expr,
                expression,
            )
        except Exception as e:  # pragma: no cover - defensive
            error_msg = f"Error converting to polar form: {e!s}"
            raise ValueError(error_msg) from e
        return {
            "magnitude": _stringify_result(magnitude),
            "angle": _stringify_result(angle),
        }

    @staticmethod
    async def handle_complex_from_polar(radius: str, angle: str) -> dict[str, Any]:
        try:
            result = await run_cpu_bound_async(complex_from_polar_expr, radius, angle)
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
        sample: bool,
    ) -> dict[str, Any]:
        try:
            result = await run_cpu_bound_async(stats_variance, values, sample)
        except Exception as e:  # pragma: no cover - defensive
            error_msg = f"Error computing variance: {e!s}"
            raise ValueError(error_msg) from e
        return {"result": _stringify_result(result)}

    @staticmethod
    async def handle_stats_stddev(values: list[object], sample: bool) -> dict[str, Any]:
        try:
            result = await run_cpu_bound_async(stats_standard_deviation, values, sample)
        except Exception as e:  # pragma: no cover - defensive
            error_msg = f"Error computing standard deviation: {e!s}"
            raise ValueError(error_msg) from e
        return {"result": _stringify_result(result)}

    @staticmethod
    async def handle_normal_pdf(value: str, mean: str, std: str) -> dict[str, Any]:
        try:
            result = await run_cpu_bound_async(normal_pdf, value, mean, std)
        except Exception as e:  # pragma: no cover - defensive
            error_msg = f"Error computing normal PDF: {e!s}"
            raise ValueError(error_msg) from e
        return {"result": _stringify_result(result)}

    @staticmethod
    async def handle_normal_cdf(value: str, mean: str, std: str) -> dict[str, Any]:
        try:
            result = await run_cpu_bound_async(normal_cdf, value, mean, std)
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
        steps: bool,
    ) -> dict[str, Any]:
        """Handle Hessian matrix calculations for multivariate expressions."""

        try:
            hessian_matrix = await run_cpu_bound_async(
                hessian_expr,
                expression,
                variables,
            )
        except Exception as e:  # pragma: no cover - defensive
            error_msg = f"Error in Hessian calculation: {e!s}"
            raise ValueError(error_msg) from e

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

    @staticmethod
    async def handle_limit_latex(
        expression: str,
        variable: str,
        point: str,
        steps: bool,
    ) -> dict[str, Any]:
        """Handle LaTeX limit operations.

        Args:
            expression: LaTeX mathematical expression for limit
            variable: Variable for the limit
            point: Point at which to calculate the limit
            steps: Whether to include step-by-step explanations

        Returns:
            Dictionary containing result string and steps list
        """
        try:
            result, all_steps = await run_cpu_bound_async(
                limit_latex_with_steps,
                expression,
                variable,
                point,
            )
            return {
                "result": _stringify_result(result),
                "steps": all_steps if steps else [],
            }
        except Exception as e:
            error_msg = f"Error in limit calculation: {e!s}"
            raise ValueError(error_msg) from e

    @staticmethod
    async def handle_series_latex(
        expression: str,
        variable: str,
        point: str,
        order: int,
        steps: bool,
    ) -> dict[str, Any]:
        """Handle LaTeX series operations.

        Args:
            expression: LaTeX mathematical expression for series expansion
            variable: Variable for the series expansion
            point: Point around which to expand (default 0 for Maclaurin series)
            order: Order of the expansion
            steps: Whether to include step-by-step explanations

        Returns:
            Dictionary containing result string and steps list
        """
        try:
            result, all_steps = await run_cpu_bound_async(
                series_latex_with_steps,
                expression,
                variable,
                point,
                order,
            )
            return {
                "result": _stringify_result(result),
                "steps": all_steps if steps else [],
            }
        except Exception as e:
            error_msg = f"Error in series calculation: {e!s}"
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
            points = await run_cpu_bound_async(
                generate_plot_points,
                expression,
                variable,
                start,
                end,
                samples,
            )
        except Exception as e:
            error_msg = f"Error while generating plot data: {e!s}"
            raise ValueError(error_msg) from e

        return {"points": points}

    @staticmethod
    async def handle_integral(
        expression: str,
        variable: str,
        steps: bool,
    ) -> dict[str, Any]:
        """Handle plain text integral operations.

        Args:
            expression: Plain text mathematical expression to integrate
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
            result = await run_cpu_bound_async(integrate_expr, expression, variable)
            return {"result": _stringify_result(result), "steps": []}
        except Exception as e:
            error_msg = f"Error in integral calculation: {e!s}"
            raise ValueError(error_msg) from e

    @staticmethod
    async def handle_derivative(
        expression: str,
        variable: str,
        steps: bool,
    ) -> dict[str, Any]:
        """Handle plain text derivative operations.

        Args:
            expression: Plain text mathematical expression to differentiate
            variable: Variable to differentiate with respect to
            steps: Whether to include step-by-step explanations

        Returns:
            Dictionary containing result string and steps list
        """
        try:
            if steps:
                result, detailed_steps = await run_cpu_bound_async(
                    differentiate_with_steps,
                    expression,
                    variable,
                )
                return {"result": _stringify_result(result), "steps": detailed_steps}

            result = await run_cpu_bound_async(differentiate_expr, expression, variable)
            return {"result": _stringify_result(result), "steps": []}
        except Exception as e:
            error_msg = f"Error in derivative calculation: {e!s}"
            raise ValueError(error_msg) from e

    @staticmethod
    async def handle_solve(
        expression: str,
        variable: str,
        steps: bool,
    ) -> dict[str, Any]:
        """Handle plain text equation solving operations.

        Args:
            expression: Plain text mathematical equation to solve
            variable: Variable to solve for
            steps: Whether to include step-by-step explanations

        Returns:
            Dictionary containing result list and steps list
        """
        try:
            if steps:
                result, detailed_steps = await run_cpu_bound_async(
                    solve_with_steps,
                    expression,
                    variable,
                )
                return {
                    "result": _stringify_result(result),
                    "steps": detailed_steps,
                }

            result = await run_cpu_bound_async(solve_equation, expression, variable)
            return {"result": _stringify_result(result), "steps": []}
        except Exception as e:
            error_msg = f"Error in equation solving: {e!s}"
            raise ValueError(error_msg) from e

    @staticmethod
    async def handle_simplify(expression: str, steps: bool) -> dict[str, Any]:
        """Handle plain text simplification operations.

        Args:
            expression: Plain text mathematical expression to simplify
            steps: Whether to include step-by-step explanations

        Returns:
            Dictionary containing result string and steps list
        """
        try:
            if steps:
                result, detailed_steps = await run_cpu_bound_async(
                    simplify_with_steps,
                    expression,
                )
                return {"result": _stringify_result(result), "steps": detailed_steps}

            result = await run_cpu_bound_async(simplify_expr, expression)
            return {"result": _stringify_result(result), "steps": []}
        except Exception as e:
            error_msg = f"Error in simplification: {e!s}"
            raise ValueError(error_msg) from e

    @staticmethod
    async def handle_limit(
        expression: str,
        variable: str,
        point: str,
        steps: bool,
    ) -> dict[str, Any]:
        """Handle plain text limit operations.

        Args:
            expression: Plain text mathematical expression for limit
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
            result = await run_cpu_bound_async(
                limit_expr,
                expression,
                variable,
                point,
            )
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
        steps: bool,
    ) -> dict[str, Any]:
        """Handle plain text series operations.

        Args:
            expression: Plain text mathematical expression for series expansion
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
            result = await run_cpu_bound_async(
                series_expr,
                expression,
                variable,
                point,
                order,
            )
            return {"result": _stringify_result(result), "steps": []}
        except Exception as e:
            error_msg = f"Error in series calculation: {e!s}"
            raise ValueError(error_msg) from e
