"""Service layer for mathematical operations."""

from __future__ import annotations

from typing import Any

from scolx_math.advanced_latex import (
    differentiate_latex_with_steps,
    integrate_latex_with_steps,
    limit_latex_with_steps,
    series_latex_with_steps,
    solve_equation_latex_with_steps,
)
from scolx_math.core.operations import (
    differentiate_expr,
    generate_plot_points,
    gradient_expr,
    hessian_expr,
    integrate_expr,
    limit_expr,
    matrix_determinant,
    matrix_inverse,
    matrix_multiply,
    series_expr,
    simplify_expr,
    solve_equation,
    solve_ode,
)
from scolx_math.core.utils import run_cpu_bound_async
from scolx_math.explain.explainers import (
    integrate_with_steps,
    limit_with_steps,
    series_with_steps,
)


class MathOperationService:
    """Service class to handle mathematical operations.

    This service provides a unified interface for performing various mathematical
    operations including integration, differentiation, equation solving, limits,
    series expansions, and simplification. It supports both plain text and LaTeX
    expressions with optional step-by-step explanations.
    """

    @staticmethod
    async def handle_integral_latex(
        expression: str, variable: str, steps: bool
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
                integrate_latex_with_steps, expression, variable
            )
            return {
                "result": str(result),
                "steps": all_steps if steps else [],
            }
        except Exception as e:
            raise ValueError(f"Error in integral calculation: {str(e)}") from e

    @staticmethod
    async def handle_derivative_latex(
        expression: str, variable: str, steps: bool
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
                differentiate_latex_with_steps, expression, variable
            )
            return {
                "result": str(result),
                "steps": all_steps if steps else [],
            }
        except Exception as e:
            raise ValueError(f"Error in derivative calculation: {str(e)}") from e

    @staticmethod
    async def handle_solve_latex(
        expression: str, variable: str, steps: bool
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
                solve_equation_latex_with_steps, expression, variable
            )
            return {
                "result": result,
                "steps": all_steps if steps else [],
            }
        except Exception as e:
            raise ValueError(f"Error in equation solving: {str(e)}") from e

    @staticmethod
    async def handle_gradient(
        expression: str, variables: list[str], steps: bool
    ) -> dict[str, Any]:
        """Handle gradient calculations for multivariate expressions."""

        try:
            gradient = await run_cpu_bound_async(gradient_expr, expression, variables)
        except Exception as e:  # pragma: no cover - defensive
            raise ValueError(f"Error in gradient calculation: {str(e)}") from e

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
            raise ValueError(
                f"Error while solving differential equation: {str(e)}"
            ) from e

        if isinstance(solution, list):
            return {"points": solution}
        return {"result": str(solution)}

    @staticmethod
    async def handle_matrix_determinant(matrix: list[list[object]]) -> dict[str, Any]:
        """Compute determinant of a matrix."""

        try:
            det_value = await run_cpu_bound_async(matrix_determinant, matrix)
        except Exception as e:  # pragma: no cover - defensive
            raise ValueError(f"Error in determinant calculation: {str(e)}") from e

        return {"result": str(det_value), "steps": []}

    @staticmethod
    async def handle_matrix_inverse(matrix: list[list[object]]) -> dict[str, Any]:
        """Compute the inverse of a matrix."""

        try:
            inverse_matrix = await run_cpu_bound_async(matrix_inverse, matrix)
        except Exception as e:  # pragma: no cover - defensive
            raise ValueError(f"Error in matrix inversion: {str(e)}") from e

        result = [[str(entry) for entry in row] for row in inverse_matrix.tolist()]
        return {"result": result, "steps": []}

    @staticmethod
    async def handle_matrix_multiply(
        left: list[list[object]], right: list[list[object]]
    ) -> dict[str, Any]:
        """Multiply two matrices."""

        try:
            product = await run_cpu_bound_async(matrix_multiply, left, right)
        except Exception as e:  # pragma: no cover - defensive
            raise ValueError(f"Error in matrix multiplication: {str(e)}") from e

        result = [[str(entry) for entry in row] for row in product.tolist()]
        return {"result": result, "steps": []}

    @staticmethod
    async def handle_hessian(
        expression: str, variables: list[str], steps: bool
    ) -> dict[str, Any]:
        """Handle Hessian matrix calculations for multivariate expressions."""

        try:
            hessian_matrix = await run_cpu_bound_async(
                hessian_expr, expression, variables
            )
        except Exception as e:  # pragma: no cover - defensive
            raise ValueError(f"Error in Hessian calculation: {str(e)}") from e

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
        expression: str, variable: str, point: str, steps: bool
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
                limit_latex_with_steps, expression, variable, point
            )
            return {
                "result": str(result),
                "steps": all_steps if steps else [],
            }
        except Exception as e:
            raise ValueError(f"Error in limit calculation: {str(e)}") from e

    @staticmethod
    async def handle_series_latex(
        expression: str, variable: str, point: str, order: int, steps: bool
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
                series_latex_with_steps, expression, variable, point, order
            )
            return {
                "result": str(result),
                "steps": all_steps if steps else [],
            }
        except Exception as e:
            raise ValueError(f"Error in series calculation: {str(e)}") from e

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
                generate_plot_points, expression, variable, start, end, samples
            )
        except Exception as e:
            raise ValueError(f"Error while generating plot data: {str(e)}") from e

        return {"points": points}

    @staticmethod
    async def handle_integral(
        expression: str, variable: str, steps: bool
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
                    integrate_with_steps, expression, variable
                )
                return {"result": str(result), "steps": all_steps if steps else []}
            else:
                result = await run_cpu_bound_async(integrate_expr, expression, variable)
                return {"result": str(result), "steps": []}
        except Exception as e:
            raise ValueError(f"Error in integral calculation: {str(e)}") from e

    @staticmethod
    async def handle_derivative(
        expression: str, variable: str, steps: bool
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
            result = await run_cpu_bound_async(differentiate_expr, expression, variable)
            all_steps = (
                ["Differentiating expression", f"Result: {result}"] if steps else []
            )
            return {"result": str(result), "steps": all_steps}
        except Exception as e:
            raise ValueError(f"Error in derivative calculation: {str(e)}") from e

    @staticmethod
    async def handle_solve(
        expression: str, variable: str, steps: bool
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
            result = await run_cpu_bound_async(solve_equation, expression, variable)
            all_steps = ["Solving equation", f"Solutions: {result}"] if steps else []
            return {
                "result": [str(sol) for sol in result],
                "steps": all_steps,
            }
        except Exception as e:
            raise ValueError(f"Error in equation solving: {str(e)}") from e

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
            result = await run_cpu_bound_async(simplify_expr, expression)
            all_steps = ["Simplifying expression", f"Result: {result}"] if steps else []
            return {"result": str(result), "steps": all_steps}
        except Exception as e:
            raise ValueError(f"Error in simplification: {str(e)}") from e

    @staticmethod
    async def handle_limit(
        expression: str, variable: str, point: str, steps: bool
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
                    limit_with_steps, expression, variable, point
                )
                return {"result": str(result), "steps": all_steps if steps else []}
            else:
                result = await run_cpu_bound_async(
                    limit_expr, expression, variable, point
                )
                return {"result": str(result), "steps": []}
        except Exception as e:
            raise ValueError(f"Error in limit calculation: {str(e)}") from e

    @staticmethod
    async def handle_series(
        expression: str, variable: str, point: str, order: int, steps: bool
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
                    series_with_steps, expression, variable, point, order
                )
                return {"result": str(result), "steps": all_steps if steps else []}
            else:
                result = await run_cpu_bound_async(
                    series_expr, expression, variable, point, order
                )
                return {"result": str(result), "steps": []}
        except Exception as e:
            raise ValueError(f"Error in series calculation: {str(e)}") from e
