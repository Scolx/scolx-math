"""Core symbolic operations used by the public API."""

from __future__ import annotations

import logging
import math

try:  # pragma: no cover - optional dependency
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    np = None

try:  # pragma: no cover - optional dependency
    from scipy import optimize
    from scipy.integrate import solve_ivp
except ImportError:  # pragma: no cover - optional dependency
    solve_ivp = None
    optimize = None

from collections.abc import Sequence

import sympy as sp
from sympy import (
    Basic,
    Eq,
    Matrix,
    diff,
    dsolve,
    hessian,
    integrate,
    limit,
    series,
    simplify,
    solve,
    symbols,
)
from sympy import sympify as sp_sympify

try:  # pragma: no cover - optional acceleration layer
    import symengine as se
except ImportError:  # pragma: no cover - symengine optional
    se = None

from scolx_math.core.parsing import (
    get_safe_locals,
    parse_plain_expression,
    validate_variable_name,
)

logger = logging.getLogger(__name__)


def _symengine_symbol(name: str) -> object:
    """Create a SymEngine symbol if SymEngine is available."""
    if se is None:
        return None
    try:
        return se.Symbol(name)
    except Exception:  # pragma: no cover - defensive
        return None


def _symengine_expr(expr: object) -> object:
    """Convert expression to SymEngine if available."""
    if se is None:
        return None
    try:
        return se.sympify(expr)
    except Exception:
        return None


def _symengine_to_sympy(result: object) -> object:
    """Convert SymEngine result back to SymPy."""
    if result is None:
        return None
    try:
        if hasattr(result, "sympy"):
            return result.sympy()
        return sp_sympify(str(result))
    except Exception:
        return None


def _symengine_simplify(expr: object) -> object:
    """Simplify expression using SymEngine if available."""
    se_expr = _symengine_expr(expr)
    if se_expr is None:
        return None
    try:
        simplified = se.simplify(se_expr)
    except Exception:
        return None
    return _symengine_to_sympy(simplified)


def _symengine_diff(expr: object, var_name: str) -> object:
    """Differentiate expression using SymEngine if available."""
    se_expr = _symengine_expr(expr)
    if se_expr is None:
        return None
    se_var = _symengine_symbol(var_name)
    if se_var is None:
        return None
    try:
        result = se.diff(se_expr, se_var)
    except Exception:
        return None
    return _symengine_to_sympy(result)


def _symengine_integrate(expr: object, var_name: str) -> object:
    """Integrate expression using SymEngine if available."""
    se_expr = _symengine_expr(expr)
    if se_expr is None:
        return None
    se_var = _symengine_symbol(var_name)
    if se_var is None:
        return None
    try:
        result = se.integrate(se_expr, se_var)
    except Exception:
        return None
    return _symengine_to_sympy(result)


def _symengine_matrix_from_matrix(matrix: Matrix) -> object:
    """Convert a SymPy matrix to a SymEngine DenseMatrix if possible."""

    if se is None:
        return None
    try:
        rows: list[list[se.Basic]] = []
        for row in matrix.tolist():
            se_row = []
            for entry in row:
                se_entry = se.sympify(str(entry))
                se_row.append(se_entry)
            rows.append(se_row)
        return se.DenseMatrix(rows)
    except Exception:  # pragma: no cover - optional path
        return None


def _symengine_matrix_to_sympy(se_matrix: object) -> Matrix | None:
    """Convert a SymEngine matrix to a SymPy Matrix."""
    if se_matrix is None:
        return None
    try:
        rows = []
        for i in range(se_matrix.nrows()):
            row = []
            for j in range(se_matrix.ncols()):
                entry = se_matrix[i, j]
                sympy_entry = _symengine_to_sympy(entry)
                row.append(
                    sympy_entry if sympy_entry is not None else sp_sympify(str(entry)),
                )
            rows.append(row)
        return Matrix(rows)
    except Exception:  # pragma: no cover - optional path
        return None


def _parse_matrix(matrix_data: list[list[object]]) -> Matrix:
    """Parse a list of lists into a SymPy Matrix."""
    if not matrix_data:
        raise ValueError("Matrix cannot be empty.")
    try:
        return Matrix(matrix_data)
    except Exception as exc:
        raise ValueError("Invalid matrix data.") from exc


def _parse_numeric_sequence(values: Sequence[object]) -> list[sp.Expr]:
    """Parse a sequence of values into SymPy expressions."""
    if not values:
        raise ValueError("Values sequence cannot be empty.")
    parsed = []
    for val in values:
        if isinstance(val, sp.Expr):
            parsed.append(val)
        else:
            try:
                parsed.append(sp_sympify(val))
            except Exception as exc:
                raise ValueError(f"Invalid numeric value: {val}") from exc
    return parsed


def solve_equation(expr_str: str, var_str: str) -> list:
    """Solve an equation for a given variable.

    Args:
        expr_str: Plain text mathematical equation (should equal 0)
        var_str: Variable to solve for

    Returns:
        List of solutions
    """
    var_name = validate_variable_name(var_str)
    x_expr = parse_plain_expression(var_name)
    expr = parse_plain_expression(expr_str, variables=[var_name])
    equation = Eq(expr, 0)
    return solve(equation, x_expr)


def integrate_expr(expr_str: str, var_str: str) -> sp.Expr:
    """Integrate a mathematical expression with respect to a variable.

    Args:
        expr_str: Plain text mathematical expression to integrate
        var_str: Variable to integrate with respect to

    Returns:
        Integrated expression
    """
    var_name = validate_variable_name(var_str)
    var_symbol = parse_plain_expression(var_name)
    expr = parse_plain_expression(expr_str, variables=[var_name])
    se_result = _symengine_integrate(expr, var_name)
    if se_result is not None:
        return se_result
    return integrate(expr, var_symbol)


def differentiate_expr(expr_str: str, var_str: str) -> sp.Expr:
    """Differentiate a mathematical expression with respect to a variable.

    Args:
        expr_str: Plain text mathematical expression to differentiate
        var_str: Variable to differentiate with respect to

    Returns:
        Differentiated expression
    """
    var_name = validate_variable_name(var_str)
    var_symbol = parse_plain_expression(var_name)
    expr = parse_plain_expression(expr_str, variables=[var_name])
    se_result = _symengine_diff(expr, var_name)
    if se_result is not None:
        return se_result
    return diff(expr, var_symbol)


def limit_expr(expr_str: str, var_str: str, point_str: str) -> sp.Expr:
    """Calculate the limit of an expression as a variable approaches a point.

    Args:
        expr_str: Plain text mathematical expression for limit
        var_str: Variable for the limit
        point_str: Point at which to calculate the limit

    Returns:
        Limit value
    """
    var_name = validate_variable_name(var_str)
    var_symbol = parse_plain_expression(var_name)
    expr = parse_plain_expression(expr_str, variables=[var_name])
    point = parse_plain_expression(point_str, variables=[var_name])

    # Limit operations don't use SymEngine as it doesn't support limits
    return limit(expr, var_symbol, point)


def series_expr(
    expr_str: str,
    var_str: str,
    point_str: str = "0",
    order: int = 6,
) -> sp.Expr:
    """Calculate the series expansion of an expression around a point.

    Args:
        expr_str: Plain text mathematical expression for series expansion
        var_str: Variable for the series expansion
        point_str: Point around which to expand (default 0 for Maclaurin series)
        order: Order of the expansion (default 6)

    Returns:
        Series expansion
    """
    var_name = validate_variable_name(var_str)
    var_symbol = parse_plain_expression(var_name)
    expr = parse_plain_expression(expr_str, variables=[var_name])
    point = parse_plain_expression(point_str, variables=[var_name])

    # Series operations don't use SymEngine as it doesn't support series
    return series(expr, var_symbol, point, n=order)


def simplify_expr(expr_str: str) -> sp.Expr:
    """Simplify a mathematical expression.

    Args:
        expr_str: Plain text mathematical expression to simplify

    Returns:
        Simplified expression
    """
    expr = parse_plain_expression(expr_str)
    se_result = _symengine_simplify(expr)
    if se_result is not None:
        return se_result
    return simplify(expr)


def matrix_determinant(matrix_data: list[list[object]]) -> Basic:
    """Compute the determinant of a square matrix."""

    matrix = _parse_matrix(matrix_data)
    if matrix.rows != matrix.cols:
        raise ValueError("Determinant requires a square matrix.")
    se_matrix = _symengine_matrix_from_matrix(matrix)
    if se_matrix is not None:
        try:
            se_det = se_matrix.det()
            converted = _symengine_to_sympy(se_det)
            if converted is not None:
                return converted
        except Exception:  # pragma: no cover - optional path
            # SymEngine failed, fallback to SymPy
            logger.debug(
                "SymEngine determinant failed; falling back to SymPy.",
                exc_info=True,
            )
    return matrix.det()


def matrix_inverse(matrix_data: list[list[object]]) -> Matrix:
    """Compute the inverse of a square, non-singular matrix."""

    matrix = _parse_matrix(matrix_data)
    if matrix.rows != matrix.cols:
        raise ValueError("Inverse requires a square matrix.")
    se_matrix = _symengine_matrix_from_matrix(matrix)
    if se_matrix is not None:
        try:
            se_inv = se_matrix.inv()
            converted = _symengine_matrix_to_sympy(se_inv)
            if converted is not None:
                return converted
        except Exception:  # pragma: no cover - optional path
            # SymEngine failed, fallback to SymPy
            logger.debug(
                "SymEngine inverse failed; falling back to SymPy.",
                exc_info=True,
            )

    try:
        return matrix.inv()
    except ValueError as exc:
        raise ValueError("Matrix is singular and cannot be inverted.") from exc


def matrix_multiply(
    matrix_a: list[list[object]],
    matrix_b: list[list[object]],
) -> Matrix:
    """Compute the product of two matrices."""

    left = _parse_matrix(matrix_a)
    right = _parse_matrix(matrix_b)
    if left.cols != right.rows:
        raise ValueError(
            "Matrix dimensions are incompatible for multiplication (columns of A must match rows of B).",
        )
    se_left = _symengine_matrix_from_matrix(left)
    se_right = _symengine_matrix_from_matrix(right)
    if se_left is not None and se_right is not None:
        try:
            product = se_left * se_right
            converted = _symengine_matrix_to_sympy(product)
            if converted is not None:
                return converted
        except Exception:  # pragma: no cover - optional path
            # SymEngine failed, fallback to SymPy
            logger.debug(
                "SymEngine multiply failed; falling back to SymPy.",
                exc_info=True,
            )
    return left * right


def solve_ode(
    ode_str: str,
    func_name: str,
    indep_var: str,
    ics: dict[str, str] | None = None,
    *,
    numeric: bool = False,
    numeric_start: str | None = None,
    numeric_end: str | None = None,
    samples: int = 100,
) -> sp.Eq | sp.Expr | list[dict[str, float | None]]:
    """Solve an ordinary differential equation, optionally using numeric methods."""

    if not func_name or not indep_var:
        raise ValueError(
            "Both function name and independent variable are required for ODE solving.",
        )

    indep_symbol = sp.Symbol(validate_variable_name(indep_var))
    func_symbol = sp.Function(validate_variable_name(func_name))(indep_symbol)

    locals_map = get_safe_locals()
    locals_map[func_name] = func_symbol.__class__

    try:
        ode_expr = sp.sympify(ode_str, locals=locals_map)
    except Exception as exc:  # pragma: no cover - defensive branch
        raise ValueError("Invalid ODE expression.") from exc

    if not isinstance(ode_expr, Eq):
        ode_expr = Eq(ode_expr, 0)

    sympy_ics: dict[sp.Symbol, float] | None = None
    if ics:
        sympy_ics = {}
        for ic_key, ic_value in ics.items():
            point_key = ic_key.strip()
            if not point_key.startswith(f"{func_name}(") or not point_key.endswith(")"):
                raise ValueError("Initial condition key must look like f(x0).")
            point_value_str = point_key[len(func_name) + 1 : -1]
            point_value = float(sp.sympify(point_value_str))
            value = float(sp.sympify(ic_value))
            sympy_ics[func_symbol.subs(indep_symbol, point_value)] = value

    analytic_allowed = not numeric
    if analytic_allowed:
        try:
            return dsolve(ode_expr, func_symbol, ics=sympy_ics)
        except Exception as exc:
            if numeric_start is None or numeric_end is None:
                raise ValueError(
                    "Unable to solve the differential equation analytically. Provide numeric=True with numeric_range to use numerical methods.",
                ) from exc
            numeric = True

    if solve_ivp is None:
        raise ValueError(
            "SciPy is required for numeric ODE solving but is not available. Install scipy to enable this feature.",
        )

    if numeric_start is None or numeric_end is None:
        raise ValueError(
            "Numeric ODE solving requires numeric_start and numeric_end values.",
        )

    t0 = float(sp.sympify(numeric_start))
    tf = float(sp.sympify(numeric_end))
    if not math.isfinite(t0) or not math.isfinite(tf):
        raise ValueError("Numeric range must consist of finite numeric values.")
    if tf <= t0:
        raise ValueError(
            "numeric_range end must be greater than start for numeric ODE solving.",
        )

    derivative_symbol = sp.diff(func_symbol, indep_symbol)
    try:
        rhs_candidates = sp.solve(ode_expr, derivative_symbol)
    except Exception as exc:  # pragma: no cover - defensive branch
        raise ValueError(
            "Numeric ODE solving currently supports first-order explicit equations.",
        ) from exc

    if not rhs_candidates:
        raise ValueError(
            "Numeric ODE solving currently supports first-order explicit equations.",
        )

    rhs_expr = rhs_candidates[0]

    if ics:
        y0 = None
        for ic_key, ic_value in ics.items():
            point_key = ic_key.strip()
            point_value_str = point_key[len(func_name) + 1 : -1]
            point_value = float(sp.sympify(point_value_str))
            if math.isclose(point_value, t0, rel_tol=1e-9, abs_tol=1e-9):
                y0 = float(sp.sympify(ic_value))
                break
        if y0 is None:
            raise ValueError(
                "Numeric ODE solving requires an initial condition defined at the numeric_range start.",
            )
    else:
        raise ValueError("Numeric ODE solving requires initial conditions.")

    func_lambda = sp.lambdify(
        (indep_symbol, func_symbol),
        rhs_expr,
        modules=["numpy", "math"],
    )

    num_samples = max(2, min(1000, samples))
    t_eval = np.linspace(t0, tf, num_samples) if np is not None else None

    def ode_rhs(t: float, y: list[float]) -> float:
        try:
            return float(func_lambda(t, y[0]))
        except Exception as exc:  # pragma: no cover - defensive branch
            raise ValueError(
                "Failed to evaluate ODE right-hand side numerically.",
            ) from exc

    solution = solve_ivp(
        ode_rhs,
        (t0, tf),
        [y0],
        t_eval=t_eval,
        dense_output=t_eval is None,
    )
    if not solution.success:
        raise ValueError(f"Numeric ODE solver failed: {solution.message}")

    times = solution.t if t_eval is not None else np.linspace(t0, tf, num_samples)
    values = solution.y[0]
    points: list[dict[str, float | None]] = []
    for t_val, y_val in zip(times, values, strict=True):
        try:
            t_float = float(t_val)
            y_float = float(y_val)
        except Exception:
            t_float, y_float = float(t_val), None
        points.append({"x": t_float, "y": y_float})

    return points


def generate_plot_points(
    expr_str: str,
    var_str: str,
    start_str: str,
    end_str: str,
    samples: int = 100,
) -> list[dict[str, float | None]]:
    """Sample an expression over a numeric range for plotting."""

    var_name = validate_variable_name(var_str)
    expr = parse_plain_expression(expr_str, variables=[var_name])
    start_expr = parse_plain_expression(start_str, variables=[var_name])
    end_expr = parse_plain_expression(end_str, variables=[var_name])

    start = float(start_expr.evalf())
    end = float(end_expr.evalf())

    if not (math.isfinite(start) and math.isfinite(end)):
        raise ValueError("Plot range must be finite numeric values.")

    if end <= start:
        raise ValueError("Plot range end must be greater than start.")

    num_samples = max(2, min(1000, samples))
    symbol = symbols(var_name)

    if np is not None:
        try:
            func = sp.lambdify(symbol, expr, modules=["numpy"])
            x_values = np.linspace(start, end, num_samples, dtype=float)
            y_values = func(x_values)
            y_array = np.asarray(y_values, dtype=float)
            if y_array.shape != x_values.shape:
                y_array = np.resize(y_array, x_values.shape)
            points = []
            for x_val, y_val in zip(x_values, y_array, strict=True):
                if math.isfinite(float(y_val)):
                    points.append({"x": float(x_val), "y": float(y_val)})
                else:
                    points.append({"x": float(x_val), "y": None})
            return points
        except Exception:  # pragma: no cover - optional path
            logger.debug(
                "NumPy vectorized plot evaluation failed; using scalar loop.",
                exc_info=True,
            )

    step = (end - start) / (num_samples - 1)
    points: list[dict[str, float | None]] = []
    for idx in range(num_samples):
        x_val = start + idx * step
        try:
            substituted = expr.subs(symbol, sp_sympify(x_val))
            y_numeric = substituted.evalf()
            y_val = float(y_numeric)
            if not math.isfinite(y_val):
                raise ValueError
        except Exception:
            y_val = None
        points.append({"x": x_val, "y": y_val})

    return points


def stats_mean(values: Sequence[object]) -> sp.Expr:
    """Compute the arithmetic mean of the supplied values."""

    data = _parse_numeric_sequence(values)
    total = sum(data)
    mean_expr = total / len(data)
    return sp.simplify(mean_expr)


def stats_variance(values: Sequence[object], *, sample: bool) -> sp.Expr:
    """Compute variance (population or sample) of the supplied values."""

    data = _parse_numeric_sequence(values)
    count = len(data)
    _MIN_SAMPLE_SIZE = 2
    if sample and count < _MIN_SAMPLE_SIZE:
        raise ValueError("At least two values are required for sample variance.")

    mean_expr = stats_mean(values)
    denominator = count - 1 if sample else count
    variance_expr = sum((value - mean_expr) ** 2 for value in data) / denominator
    return sp.simplify(variance_expr)


def stats_standard_deviation(values: Sequence[object], *, sample: bool) -> sp.Expr:
    """Compute standard deviation for the supplied values."""

    variance_expr = stats_variance(values, sample=sample)
    return sp.simplify(sp.sqrt(variance_expr))


def normal_pdf(value_str: str, mean_str: str, std_str: str) -> sp.Expr:
    """Return the normal distribution PDF evaluated at value."""

    value = parse_plain_expression(value_str)
    mean = parse_plain_expression(mean_str)
    std = parse_plain_expression(std_str)

    if std == 0:
        raise ValueError("Standard deviation must be non-zero.")

    pdf_expr = (1 / (std * sp.sqrt(2 * sp.pi))) * sp.exp(
        -((value - mean) ** 2) / (2 * std**2),
    )
    return sp.simplify(pdf_expr)


def normal_cdf(value_str: str, mean_str: str, std_str: str) -> sp.Expr:
    """Return the normal distribution CDF evaluated at value."""

    value = parse_plain_expression(value_str)
    mean = parse_plain_expression(mean_str)
    std = parse_plain_expression(std_str)

    if std == 0:
        raise ValueError("Standard deviation must be non-zero.")

    z = (value - mean) / (sp.sqrt(2) * std)
    cdf_expr = sp.Rational(1, 2) * (1 + sp.erf(z))
    return sp.simplify(cdf_expr)


def complex_conjugate_expr(expr_str: str) -> sp.Expr:
    """Return the complex conjugate of the given expression."""

    expr = parse_plain_expression(expr_str)
    return sp.conjugate(expr)


def complex_modulus_expr(expr_str: str) -> sp.Expr:
    """Return the complex modulus of the given expression."""

    expr = parse_plain_expression(expr_str)
    return sp.Abs(expr)


def complex_argument_expr(expr_str: str) -> sp.Expr:
    """Return the complex argument of the given expression."""

    expr = parse_plain_expression(expr_str)
    return sp.arg(expr)


def complex_to_polar_expr(expr_str: str) -> tuple[sp.Expr, sp.Expr]:
    """Convert a complex expression to polar form (magnitude, angle)."""

    expr = parse_plain_expression(expr_str)
    magnitude = sp.Abs(expr)
    angle = sp.arg(expr)
    return magnitude, angle


def complex_from_polar_expr(radius_str: str, angle_str: str) -> sp.Expr:
    """Convert polar form inputs into a complex expression."""

    radius = parse_plain_expression(radius_str)
    angle = parse_plain_expression(angle_str)
    complex_expr = radius * sp.exp(sp.I * angle)
    return sp.simplify(complex_expr)


def solve_system_numeric(
    equations: Sequence[str],
    variables: Sequence[str],
    initial_guess: Sequence[str | float] | None = None,
    max_iterations: int = 100,
    tolerance: float = 1e-9,
) -> dict[str, float]:
    """Solve a system of nonlinear equations numerically using SciPy."""

    if optimize is None:
        raise ValueError(
            "SciPy is required for numeric root solving but is not available. Install scipy to enable this feature.",
        )

    if not equations:
        raise ValueError("At least one equation must be provided.")

    if len(equations) != len(variables):
        raise ValueError("Number of equations must match number of variables.")

    var_symbols = [sp.Symbol(validate_variable_name(name)) for name in variables]
    safe_locals = get_safe_locals()
    for var_symbol in var_symbols:
        safe_locals[var_symbol.name] = var_symbol

    parsed_equations: list[sp.Expr] = []
    for eq_str in equations:
        try:
            expr = sp.sympify(eq_str, locals=safe_locals)
        except Exception as exc:  # pragma: no cover - defensive branch
            raise ValueError("Invalid equation supplied for numeric solving.") from exc
        if isinstance(expr, Eq):
            parsed_equations.append(expr.lhs - expr.rhs)
        else:
            parsed_equations.append(expr)

    if initial_guess is None:
        guess = [0.0] * len(variables)
    else:
        if len(initial_guess) != len(variables):
            raise ValueError("Initial guess must provide a value for each variable.")
        guess = [float(sp.sympify(val)) for val in initial_guess]

    modules = ["numpy"] if np is not None else ["math"]
    system_lambda = sp.lambdify(var_symbols, parsed_equations, modules=modules)

    def func(values: Sequence[float]) -> list[float]:
        try:
            result = system_lambda(*values)
        except Exception as exc:  # pragma: no cover - defensive branch
            raise ValueError("Failed to evaluate equations numerically.") from exc
        if np is not None:
            result_array = np.asarray(result, dtype=float)
            return result_array.tolist()
        if isinstance(result, (list, tuple)):
            return [float(sp.sympify(val)) for val in result]
        return [float(sp.sympify(result))]

    max_function_evals = max(max_iterations * len(variables), len(variables) * 50)
    solution = optimize.root(
        func,
        guess,
        tol=tolerance,
        options={"maxfev": max_function_evals},
    )
    if not solution.success:
        raise ValueError(f"Numeric solver failed: {solution.message}")

    solved_values: dict[str, float] = {}
    for name, value in zip(variables, solution.x, strict=False):
        solved_values[name] = float(value)
    return solved_values


def gradient_expr(expr_str: str, variables: list[str]) -> list[sp.Expr]:
    """Compute the gradient (vector of partial derivatives) of an expression."""

    if not variables:
        raise ValueError("At least one variable is required for gradient calculation.")

    var_names = [validate_variable_name(var) for var in variables]
    expr = parse_plain_expression(expr_str, variables=var_names)
    sym_vars = symbols(var_names)

    return [diff(expr, sym_var) for sym_var in sym_vars]


def hessian_expr(expr_str: str, variables: list[str]) -> Matrix:
    """Compute the Hessian matrix of second-order partial derivatives."""

    if not variables:
        raise ValueError("At least one variable is required for Hessian calculation.")

    var_names = [validate_variable_name(var) for var in variables]
    expr = parse_plain_expression(expr_str, variables=var_names)
    sym_vars = symbols(var_names)

    return hessian(expr, sym_vars)
