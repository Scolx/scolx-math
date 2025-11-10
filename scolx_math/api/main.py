"""Main API module for the Scolx Math API."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from enum import Enum

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator

from scolx_math.api.service import MathOperationService
from scolx_math.core.utils import cleanup_threadpool

LATEX_TYPES = {
    "integral_latex",
    "derivative_latex",
    "solve_latex",
    "limit_latex",
    "series_latex",
}

MATRIX_SINGLE_TYPES = {"matrix_determinant", "matrix_inverse"}
MATRIX_DOUBLE_TYPES = {"matrix_multiply"}
MATRIX_TYPES = MATRIX_SINGLE_TYPES | MATRIX_DOUBLE_TYPES
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
    GRADIENT = "gradient"
    HESSIAN = "hessian"
    INTEGRAL_LATEX = "integral_latex"
    DERIVATIVE_LATEX = "derivative_latex"
    SOLVE_LATEX = "solve_latex"
    LIMIT_LATEX = "limit_latex"
    SERIES_LATEX = "series_latex"
    MATRIX_DETERMINANT = "matrix_determinant"
    MATRIX_INVERSE = "matrix_inverse"
    MATRIX_MULTIPLY = "matrix_multiply"
    ODE = "ode"
    PLOT = "plot"
    COMPLEX_CONJUGATE = "complex_conjugate"
    COMPLEX_MODULUS = "complex_modulus"
    COMPLEX_ARGUMENT = "complex_argument"
    COMPLEX_TO_POLAR = "complex_to_polar"
    COMPLEX_FROM_POLAR = "complex_from_polar"
    STATS_MEAN = "stats_mean"
    STATS_VARIANCE = "stats_variance"
    STATS_STDDEV = "stats_stddev"
    NORMAL_PDF = "normal_pdf"
    NORMAL_CDF = "normal_cdf"
    SOLVE_NUMERIC = "solve_numeric"


EXPRESSION_REQUIRED = {
    OperationType.INTEGRAL.value,
    OperationType.DERIVATIVE.value,
    OperationType.SOLVE.value,
    OperationType.SIMPLIFY.value,
    OperationType.LIMIT.value,
    OperationType.SERIES.value,
    OperationType.INTEGRAL_LATEX.value,
    OperationType.DERIVATIVE_LATEX.value,
    OperationType.SOLVE_LATEX.value,
    OperationType.LIMIT_LATEX.value,
    OperationType.SERIES_LATEX.value,
    OperationType.GRADIENT.value,
    OperationType.HESSIAN.value,
    OperationType.PLOT.value,
    OperationType.COMPLEX_CONJUGATE.value,
    OperationType.COMPLEX_MODULUS.value,
    OperationType.COMPLEX_ARGUMENT.value,
    OperationType.COMPLEX_TO_POLAR.value,
}


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None]:
    """Lifespan event handler for application startup and shutdown."""
    try:
        # Application startup
        yield
    finally:
        # Application shutdown
        cleanup_threadpool()


app = FastAPI(title="Scolx Math API", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def _handle_validation_errors(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
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
    expression: str | None = Field(None)
    variable: str | None = Field(None, description="Variable name when required")
    variables: list[str] | None = Field(
        None,
        description="Variables for multivariate operations (e.g. gradient)",
    )
    point: str | None = Field(None, description="Point for limit/series operations")
    order: int = Field(6, description="Series expansion order")
    steps: bool = Field(default=True, description="Include step-by-step explanations")
    is_latex: bool = Field(
        default=False,
        description="Whether expression is provided as LaTeX (redundant for *_latex types)",
    )
    plot_range: tuple[str, str] | None = Field(
        None,
        description="Tuple specifying start and end for plotting",
    )
    numeric_range: tuple[str, str] | None = Field(
        None,
        description="Tuple specifying start and end for numeric ODE solving",
    )
    samples: int = Field(
        100,
        ge=2,
        le=1000,
        description="Sample count for plotting or numeric solutions",
    )
    numeric: bool = Field(
        default=False,
        description="Use numeric methods when available",
    )
    matrix: list[list[object]] | None = Field(
        None,
        description="Matrix input for determinant/inverse operations",
    )
    left_matrix: list[list[object]] | None = Field(
        None,
        description="Left matrix for multiplication",
    )
    right_matrix: list[list[object]] | None = Field(
        None,
        description="Right matrix for multiplication",
    )
    function: str | None = Field(
        None,
        description="Dependent function name for ODE solving",
    )
    initial_conditions: dict[str, str] | None = Field(
        None,
        description="Initial conditions for ODE solving, keyed by f(x0)",
    )
    radius: str | None = Field(
        None,
        description="Radius for polar-to-complex conversion",
    )
    angle: str | None = Field(None, description="Angle for polar-to-complex conversion")
    values: list[object] | None = Field(
        None,
        description="Values for statistical calculations",
    )
    sample: bool = Field(
        default=False,
        description="Use sample statistics when applicable",
    )
    distribution_value: str | None = Field(
        None,
        description="Value at which to evaluate probability distributions",
    )
    mean_value: str | None = Field(
        None,
        description="Mean parameter for probability distributions",
    )
    std_value: str | None = Field(
        None,
        description="Standard deviation parameter for probability distributions",
    )
    equations: list[str] | None = Field(
        None,
        description="System of equations for numeric solving",
    )
    equation_variables: list[str] | None = Field(
        None,
        description="Variables involved in numeric solving",
    )
    initial_guess: list[str | float] | None = Field(
        None,
        description="Initial guess values for numeric solving",
    )
    max_iterations: int = Field(
        100,
        ge=1,
        le=10000,
        description="Maximum iterations for numeric solvers",
    )
    tolerance: float = Field(1e-9, gt=0.0, description="Tolerance for numeric solvers")

    @field_validator("expression")
    @classmethod
    def _strip_expression(cls, value: str | None) -> str | None:
        """Validate that expression is not empty."""
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @field_validator("variable", "point", mode="before")
    @classmethod
    def _normalize_optional(cls, value: str | None) -> str | None:
        """Normalize optional string fields."""
        if isinstance(value, str):
            value = value.strip()
        return value or None

    @field_validator("matrix", "left_matrix", "right_matrix", mode="before")
    @classmethod
    def _normalize_matrices(cls, value: object) -> list[list[object]] | None:
        """Ensure matrices are lists of lists and normalize tuples."""

        if value is None:
            return None

        if isinstance(value, tuple):
            value = list(value)
        if not isinstance(value, list):
            raise TypeError("Matrix must be provided as a list of lists.")

        normalized: list[list[object]] = []
        for row in value:
            converted_row = list(row) if isinstance(row, tuple) else row
            if not isinstance(converted_row, list):
                raise TypeError("Matrix rows must be provided as lists.")
            normalized.append(list(converted_row))

        return normalized

    @field_validator("plot_range", mode="before")
    @classmethod
    def _normalize_plot_range(cls, value: object) -> tuple[str, str] | None:
        """Normalize plot range to a tuple of two strings."""

        if value is None:
            return None

        _RANGE_LENGTH = 2
        if isinstance(value, (list, tuple)) and len(value) == _RANGE_LENGTH:
            start, end = value
            if not isinstance(start, str) or not isinstance(end, str):
                raise TypeError("Plot range must contain two string expressions.")
            start = start.strip()
            end = end.strip()
            if not start or not end:
                raise ValueError("Plot range expressions cannot be empty.")
            return start, end

        raise TypeError("Plot range must be a sequence with two string elements.")

    @field_validator("numeric_range", mode="before")
    @classmethod
    def _normalize_numeric_range(cls, value: object) -> tuple[str, str] | None:
        """Normalize numeric range for ODE solving."""

        if value is None:
            return None

        _RANGE_LENGTH = 2
        if isinstance(value, (list, tuple)) and len(value) == _RANGE_LENGTH:
            start, end = value
            if not isinstance(start, str) or not isinstance(end, str):
                raise TypeError("Numeric range must contain two string expressions.")
            start = start.strip()
            end = end.strip()
            if not start or not end:
                raise ValueError("Numeric range expressions cannot be empty.")
            return start, end

        raise TypeError("Numeric range must be a sequence with two string elements.")

    @field_validator("distribution_value", "mean_value", "std_value", mode="before")
    @classmethod
    def _strip_optional(cls, value: str | None) -> str | None:
        if isinstance(value, str):
            value = value.strip()
        return value or None

    @field_validator("equations", mode="before")
    @classmethod
    def _normalize_equations(cls, value: object) -> list[str] | None:
        if value is None:
            return None
        if not isinstance(value, (list, tuple)):
            raise TypeError("Equations must be provided as a list of strings.")
        normalized = []
        for item in value:
            if not isinstance(item, str):
                raise TypeError("Each equation must be a string.")
            stripped = item.strip()
            if stripped:
                normalized.append(stripped)
        return normalized or None

    @field_validator("equation_variables", mode="before")
    @classmethod
    def _normalize_equation_variables(cls, value: object) -> list[str] | None:
        if value is None:
            return None
        if not isinstance(value, (list, tuple)):
            raise TypeError("Variables must be provided as a list of strings.")
        normalized = []
        for item in value:
            if not isinstance(item, str):
                raise TypeError("Each variable must be a string.")
            stripped = item.strip()
            if stripped:
                normalized.append(stripped)
        return normalized or None

    @field_validator("variables", mode="before")
    @classmethod
    def _normalize_variables(cls, value: object) -> list[str] | None:
        """Normalize variables list for multivariate operations."""

        if value is None:
            return None

        if isinstance(value, str):
            value = [value]

        if isinstance(value, (list, tuple)):
            cleaned: list[str] = []
            for item in value:
                if item is None:
                    continue
                if not isinstance(item, str):
                    raise TypeError("Variable names must be strings.")
                trimmed = item.strip()
                if trimmed and trimmed not in cleaned:
                    cleaned.append(trimmed)
            return cleaned or None

        raise TypeError("Variables must be provided as a list of strings.")

    @field_validator("order")
    @classmethod
    def _validate_order(cls, value: int) -> int:
        """Validate that series order is positive."""
        if value <= 0:
            raise ValueError("Series order must be a positive integer.")
        return value

    @model_validator(mode="after")
    def _validate_combinations(self) -> MathRequest:
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

        if (
            op_value
            in {
                OperationType.GRADIENT.value,
                OperationType.HESSIAN.value,
            }
            and not self.variables
        ):
            raise ValueError("Variables list is required for multivariate operations.")

        if op_value in EXPRESSION_REQUIRED and not self.expression:
            raise ValueError("Expression is required for the selected operation.")

        if op_value in MATRIX_SINGLE_TYPES and not self.matrix:
            raise ValueError("Matrix input is required for this operation.")

        if op_value in MATRIX_DOUBLE_TYPES and (
            not self.left_matrix or not self.right_matrix
        ):
            raise ValueError(
                "Both left_matrix and right_matrix are required for matrix multiplication.",
            )

        if op_value == OperationType.PLOT.value:
            if not self.plot_range:
                raise ValueError("Plot range must be provided as [start, end].")
            if not self.variable:
                raise ValueError("Variable is required for plotting operations.")
            if not self.expression:
                raise ValueError("Expression is required for plotting operations.")

        if op_value == OperationType.ODE.value:
            if not self.expression:
                raise ValueError("ODE equation expression is required.")
            if not self.variable:
                raise ValueError("ODE variable is required.")
            if not self.function:
                raise ValueError("Function name is required for ODE solving.")
            if self.numeric and not self.numeric_range:
                raise ValueError(
                    "Numeric ODE solving requires numeric_range to be provided.",
                )

        if op_value == OperationType.COMPLEX_FROM_POLAR.value and (
            not self.radius or not self.angle
        ):
            raise ValueError(
                "Radius and angle are required for complex_from_polar operations.",
            )

        if (
            op_value
            in {
                OperationType.COMPLEX_CONJUGATE.value,
                OperationType.COMPLEX_MODULUS.value,
                OperationType.COMPLEX_ARGUMENT.value,
                OperationType.COMPLEX_TO_POLAR.value,
            }
            and not self.expression
        ):
            raise ValueError("Expression is required for complex number operations.")

        if (
            op_value
            in {
                OperationType.STATS_MEAN.value,
                OperationType.STATS_VARIANCE.value,
                OperationType.STATS_STDDEV.value,
            }
            and not self.values
        ):
            raise ValueError("Values list is required for statistical operations.")

        if op_value in {
            OperationType.NORMAL_PDF.value,
            OperationType.NORMAL_CDF.value,
        }:
            missing = [
                field
                for field, field_value in {
                    "distribution_value": self.distribution_value,
                    "mean_value": self.mean_value,
                    "std_value": self.std_value,
                }.items()
                if not field_value
            ]
            if missing:
                raise ValueError(
                    "Value, mean, and std must be provided for normal distribution operations.",
                )

        if op_value == OperationType.SOLVE_NUMERIC.value:
            if not self.equations:
                raise ValueError("Provide at least one equation for numeric solving.")
            if not self.equation_variables:
                raise ValueError("Provide variables for numeric solving.")
            if len(self.equations) != len(self.equation_variables):
                raise ValueError(
                    "Number of equations must match number of variables for numeric solving.",
                )

        return self

    @property
    def ordered_variables(self) -> list[str] | None:
        """Return variables preserving original order."""

        return self.variables

    @property
    def matrix_data(self) -> list[list[object]] | None:
        return self.matrix

    @property
    def left_matrix_data(self) -> list[list[object]] | None:
        return self.left_matrix

    @property
    def right_matrix_data(self) -> list[list[object]] | None:
        return self.right_matrix

    @property
    def plot_start(self) -> str | None:
        return self.plot_range[0] if self.plot_range else None

    @property
    def plot_end(self) -> str | None:
        return self.plot_range[1] if self.plot_range else None

    @property
    def ode_initial_conditions(self) -> dict[str, str] | None:
        return self.initial_conditions

    @property
    def numeric_start(self) -> str | None:
        return self.numeric_range[0] if self.numeric_range else None

    @property
    def numeric_end(self) -> str | None:
        return self.numeric_range[1] if self.numeric_range else None

    @property
    def polar_radius(self) -> str | None:
        return self.radius

    @property
    def polar_angle(self) -> str | None:
        return self.angle

    @property
    def stats_values(self) -> list[object] | None:
        return self.values

    @property
    def distribution_parameters(self) -> tuple[str | None, str | None, str | None]:
        return self.distribution_value, self.mean_value, self.std_value

    @property
    def numeric_equations(self) -> list[str] | None:
        return self.equations

    @property
    def numeric_variables(self) -> list[str] | None:
        return self.equation_variables

    @property
    def numeric_initial_guess(self) -> list[str | float] | None:
        return self.initial_guess


@app.post("/solve")
async def solve_math(req: MathRequest) -> dict[str, object]:
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
                req.expression,
                req.variable,
                steps=req.steps,
            )
        if req.type is OperationType.DERIVATIVE_LATEX:
            return await MathOperationService.handle_derivative_latex(
                req.expression,
                req.variable,
                steps=req.steps,
            )
        if req.type is OperationType.SOLVE_LATEX:
            return await MathOperationService.handle_solve_latex(
                req.expression,
                req.variable,
                steps=req.steps,
            )
        if req.type is OperationType.LIMIT_LATEX:
            return await MathOperationService.handle_limit_latex(
                req.expression,
                req.variable,
                req.point,
                steps=req.steps,
            )
        if req.type is OperationType.SERIES_LATEX:
            return await MathOperationService.handle_series_latex(
                req.expression,
                req.variable,
                req.point,
                req.order,
                steps=req.steps,
            )

        # Plain text operations
        if req.type is OperationType.INTEGRAL:
            return await MathOperationService.handle_integral(
                req.expression,
                req.variable,
                steps=req.steps,
            )
        if req.type is OperationType.DERIVATIVE:
            return await MathOperationService.handle_derivative(
                req.expression,
                req.variable,
                steps=req.steps,
            )
        if req.type is OperationType.SOLVE:
            return await MathOperationService.handle_solve(
                req.expression,
                req.variable,
                steps=req.steps,
            )
        if req.type is OperationType.SIMPLIFY:
            return await MathOperationService.handle_simplify(
                req.expression,
                steps=req.steps,
            )
        if req.type is OperationType.LIMIT:
            return await MathOperationService.handle_limit(
                req.expression,
                req.variable,
                req.point,
                steps=req.steps,
            )
        if req.type is OperationType.SERIES:
            return await MathOperationService.handle_series(
                req.expression,
                req.variable,
                req.point,
                req.order,
                steps=req.steps,
            )
        if req.type is OperationType.MATRIX_DETERMINANT:
            return await MathOperationService.handle_matrix_determinant(
                req.matrix_data or [],
            )
        if req.type is OperationType.MATRIX_INVERSE:
            return await MathOperationService.handle_matrix_inverse(
                req.matrix_data or [],
            )
        if req.type is OperationType.MATRIX_MULTIPLY:
            return await MathOperationService.handle_matrix_multiply(
                req.left_matrix_data or [],
                req.right_matrix_data or [],
            )
        if req.type is OperationType.GRADIENT:
            return await MathOperationService.handle_gradient(
                req.expression,
                req.ordered_variables or [],
                steps=req.steps,
            )
        if req.type is OperationType.HESSIAN:
            return await MathOperationService.handle_hessian(
                req.expression,
                req.ordered_variables or [],
                steps=req.steps,
            )
        if req.type is OperationType.ODE:
            return await MathOperationService.handle_ode(
                req.expression,
                req.function,
                req.variable,
                req.ode_initial_conditions,
                numeric=req.numeric,
                numeric_start=req.numeric_start,
                numeric_end=req.numeric_end,
                samples=req.samples,
            )
        if req.type is OperationType.COMPLEX_CONJUGATE:
            return await MathOperationService.handle_complex_conjugate(req.expression)
        if req.type is OperationType.COMPLEX_MODULUS:
            return await MathOperationService.handle_complex_modulus(req.expression)
        if req.type is OperationType.COMPLEX_ARGUMENT:
            return await MathOperationService.handle_complex_argument(req.expression)
        if req.type is OperationType.COMPLEX_TO_POLAR:
            return await MathOperationService.handle_complex_to_polar(req.expression)
        if req.type is OperationType.COMPLEX_FROM_POLAR:
            return await MathOperationService.handle_complex_from_polar(
                req.polar_radius or "0",
                req.polar_angle or "0",
            )
        if req.type is OperationType.STATS_MEAN:
            return await MathOperationService.handle_stats_mean(req.stats_values or [])
        if req.type is OperationType.STATS_VARIANCE:
            return await MathOperationService.handle_stats_variance(
                req.stats_values or [],
                sample=req.sample,
            )
        if req.type is OperationType.STATS_STDDEV:
            return await MathOperationService.handle_stats_stddev(
                req.stats_values or [],
                sample=req.sample,
            )
        if req.type is OperationType.NORMAL_PDF:
            value, mean, std = req.distribution_parameters
            return await MathOperationService.handle_normal_pdf(
                value or "0",
                mean or "0",
                std or "1",
            )
        if req.type is OperationType.NORMAL_CDF:
            value, mean, std = req.distribution_parameters
            return await MathOperationService.handle_normal_cdf(
                value or "0",
                mean or "0",
                std or "1",
            )
        if req.type is OperationType.SOLVE_NUMERIC:
            return await MathOperationService.handle_numeric_solve(
                req.numeric_equations or [],
                req.numeric_variables or [],
                req.numeric_initial_guess,
                req.max_iterations,
                req.tolerance,
            )
        if req.type is OperationType.PLOT:
            return await MathOperationService.handle_plot(
                req.expression,
                req.variable,
                req.plot_start or "0",
                req.plot_end or "0",
                req.samples,
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


@app.get("/livez", status_code=status.HTTP_200_OK)
async def health_livez() -> dict[str, str]:
    """Liveness probe indicating the service is running."""

    return {"status": "ok"}


@app.get("/readyz", status_code=status.HTTP_200_OK)
async def health_readyz() -> dict[str, str]:
    """Readiness probe for load balancers."""

    return {"status": "ready"}


@app.get("/startupz", status_code=status.HTTP_200_OK)
async def health_startupz() -> dict[str, str]:
    """Startup probe indicating initialization is complete."""

    return {"status": "started"}
