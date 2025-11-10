"""Service layer for mathematical operations."""

from __future__ import annotations

from typing import Any

from fastapi.concurrency import run_in_threadpool

from scolx_math.advanced_latex import (
    differentiate_latex_with_steps,
    integrate_latex_with_steps,
    limit_latex_with_steps,
    series_latex_with_steps,
    solve_equation_latex_with_steps,
)
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
            result, all_steps = await run_in_threadpool(
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
            result, all_steps = await run_in_threadpool(
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
            result, all_steps = await run_in_threadpool(
                solve_equation_latex_with_steps, expression, variable
            )
            return {
                "result": result,
                "steps": all_steps if steps else [],
            }
        except Exception as e:
            raise ValueError(f"Error in equation solving: {str(e)}") from e

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
            result, all_steps = await run_in_threadpool(
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
            result, all_steps = await run_in_threadpool(
                series_latex_with_steps, expression, variable, point, order
            )
            return {
                "result": str(result),
                "steps": all_steps if steps else [],
            }
        except Exception as e:
            raise ValueError(f"Error in series calculation: {str(e)}") from e

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
                result, all_steps = await run_in_threadpool(
                    integrate_with_steps, expression, variable
                )
                return {"result": str(result), "steps": all_steps if steps else []}
            else:
                result = await run_in_threadpool(integrate_expr, expression, variable)
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
            result = await run_in_threadpool(differentiate_expr, expression, variable)
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
            result = await run_in_threadpool(solve_equation, expression, variable)
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
            result = await run_in_threadpool(simplify_expr, expression)
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
                result, all_steps = await run_in_threadpool(
                    limit_with_steps, expression, variable, point
                )
                return {"result": str(result), "steps": all_steps if steps else []}
            else:
                result = await run_in_threadpool(
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
                result, all_steps = await run_in_threadpool(
                    series_with_steps, expression, variable, point, order
                )
                return {"result": str(result), "steps": all_steps if steps else []}
            else:
                result = await run_in_threadpool(
                    series_expr, expression, variable, point, order
                )
                return {"result": str(result), "steps": []}
        except Exception as e:
            raise ValueError(f"Error in series calculation: {str(e)}") from e
