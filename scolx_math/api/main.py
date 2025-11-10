"""Main API module for the Scolx Math API."""

from enum import Enum
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator

from scolx_math.api.service import MathOperationService

LATEX_TYPES = {
    "integral_latex",
    "derivative_latex",
    "solve_latex",
    "limit_latex",
    "series_latex",
}
VARIABLE_REQUIRED = {
    "integral",
    "integral_latex",
    "derivative",
    "derivative_latex",
    "solve",
    "solve_latex",
    "limit",
    "limit_latex",
    "series",
    "series_latex",
}
POINT_REQUIRED = {"limit", "limit_latex", "series", "series_latex"}


class OperationType(str, Enum):
    """Enumeration of supported mathematical operations."""

    INTEGRAL = "integral"
    DERIVATIVE = "derivative"
    SOLVE = "solve"
    SIMPLIFY = "simplify"
    LIMIT = "limit"
    SERIES = "series"
    INTEGRAL_LATEX = "integral_latex"
    DERIVATIVE_LATEX = "derivative_latex"
    SOLVE_LATEX = "solve_latex"
    LIMIT_LATEX = "limit_latex"
    SERIES_LATEX = "series_latex"


app = FastAPI(title="Scolx Math API")


@app.exception_handler(RequestValidationError)
async def _handle_validation_errors(request: Request, exc: RequestValidationError):
    """Handle request validation errors.

    Args:
        request: The incoming request
        exc: The validation error exception

    Returns:
        JSON response with error details
    """
    errors = exc.errors()
    if any(err.get("loc", []) and err["loc"][-1] == "type" for err in errors):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": "Unsupported operation type."},
        )
    return await request_validation_exception_handler(request, exc)


class MathRequest(BaseModel):
    """Request model for mathematical operations."""

    type: OperationType = Field(
        ...,
        description="Operation to perform (e.g. integral, solve_latex)",
    )
    expression: str = Field(..., min_length=1)
    variable: str | None = Field(None, description="Variable name when required")
    point: str | None = Field(None, description="Point for limit/series operations")
    order: int = Field(6, description="Series expansion order")
    steps: bool = Field(True, description="Include step-by-step explanations")
    is_latex: bool = Field(
        False,
        description="Whether expression is provided as LaTeX (redundant for *_latex types)",
    )

    @field_validator("expression")
    @classmethod
    def _strip_expression(cls, value: str) -> str:
        """Validate that expression is not empty."""
        if not value.strip():
            raise ValueError("Expression cannot be empty.")
        return value

    @field_validator("variable", "point", mode="before")
    @classmethod
    def _normalize_optional(cls, value: Any) -> Any:
        """Normalize optional string fields."""
        if isinstance(value, str):
            value = value.strip()
        return value or None

    @field_validator("order")
    @classmethod
    def _validate_order(cls, value: int) -> int:
        """Validate that series order is positive."""
        if value <= 0:
            raise ValueError("Series order must be a positive integer.")
        return value

    @model_validator(mode="after")
    def _validate_combinations(self) -> "MathRequest":
        """Validate field combinations based on operation type."""
        op_value = self.type.value
        if op_value in LATEX_TYPES and not self.is_latex:
            raise ValueError("Set is_latex=true for LaTeX-specific operations.")
        if op_value not in LATEX_TYPES and self.is_latex:
            raise ValueError("Plain-text operations must not set is_latex=true.")

        if op_value in VARIABLE_REQUIRED and not self.variable:
            raise ValueError("Variable is required for the selected operation.")

        if op_value in POINT_REQUIRED and not self.point:
            raise ValueError("Point is required for limit or series operations.")

        return self


@app.post("/solve")
async def solve_math(req: MathRequest):
    """Solve mathematical operations.

    This endpoint handles various mathematical operations including integration,
    differentiation, equation solving, limits, series expansions, and simplification.
    It supports both plain text and LaTeX expressions with optional step-by-step
    explanations.

    Args:
        req: Mathematical operation request with expression and parameters

    Returns:
        Dictionary containing result and optional step-by-step explanations
    """
    try:
        # LaTeX operations
        if req.type is OperationType.INTEGRAL_LATEX:
            return await MathOperationService.handle_integral_latex(
                req.expression, req.variable, req.steps
            )
        elif req.type is OperationType.DERIVATIVE_LATEX:
            return await MathOperationService.handle_derivative_latex(
                req.expression, req.variable, req.steps
            )
        elif req.type is OperationType.SOLVE_LATEX:
            return await MathOperationService.handle_solve_latex(
                req.expression, req.variable, req.steps
            )
        elif req.type is OperationType.LIMIT_LATEX:
            return await MathOperationService.handle_limit_latex(
                req.expression, req.variable, req.point, req.steps
            )
        elif req.type is OperationType.SERIES_LATEX:
            return await MathOperationService.handle_series_latex(
                req.expression, req.variable, req.point, req.order, req.steps
            )

        # Plain text operations
        elif req.type is OperationType.INTEGRAL:
            return await MathOperationService.handle_integral(
                req.expression, req.variable, req.steps
            )
        elif req.type is OperationType.DERIVATIVE:
            return await MathOperationService.handle_derivative(
                req.expression, req.variable, req.steps
            )
        elif req.type is OperationType.SOLVE:
            return await MathOperationService.handle_solve(
                req.expression, req.variable, req.steps
            )
        elif req.type is OperationType.SIMPLIFY:
            return await MathOperationService.handle_simplify(req.expression, req.steps)
        elif req.type is OperationType.LIMIT:
            return await MathOperationService.handle_limit(
                req.expression, req.variable, req.point, req.steps
            )
        elif req.type is OperationType.SERIES:
            return await MathOperationService.handle_series(
                req.expression, req.variable, req.point, req.order, req.steps
            )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported operation type.",
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:  # pragma: no cover - unexpected error handler
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error.",
        ) from exc
