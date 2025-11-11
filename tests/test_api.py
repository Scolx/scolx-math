"""Tests for the FastAPI endpoints."""

from typing import Any

import pytest
from fastapi.testclient import TestClient
from requests import Response

from scolx_math.api.main import app

client = TestClient(app)


def _post(payload: dict[str, Any]) -> Response:
    return client.post("/solve", json=payload)


def test_solve_endpoint_integral() -> None:
    """Test integral calculation."""
    response = _post(
        {
            "type": "integral",
            "expression": "x**2",
            "variable": "x",
            "steps": True,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["result"] == "x**3/3"
    assert data["steps"][0] == "Parsing expression and simplifying"


def test_solve_endpoint_derivative() -> None:
    """Test derivative calculation."""
    response = _post(
        {
            "type": "derivative",
            "expression": "x**3",
            "variable": "x",
            "steps": True,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "result" in data
    assert "steps" in data
    assert data["result"] == "3*x**2"


def test_derivative_endpoint_no_steps() -> None:
    response = _post(
        {
            "type": "derivative",
            "expression": "(x**2 + 1)",
            "variable": "x",
            "steps": False,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["result"] == "2*x"


def test_solve_endpoint_solve() -> None:
    """Test equation solving."""
    response = _post(
        {
            "type": "solve",
            "expression": "x**2 - 4",
            "variable": "x",
            "steps": True,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "result" in data
    assert "steps" in data
    assert "-2" in data["result"]
    assert "2" in data["result"]


def test_solve_endpoint_simplify() -> None:
    """Test expression simplification."""
    response = _post(
        {
            "type": "simplify",
            "expression": "(x+1)**2 - x**2 - 2*x - 1",
            "steps": True,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "result" in data
    assert "steps" in data
    assert data["result"] == "0"


def test_solve_endpoint_gradient() -> None:
    """Test gradient calculation for multivariate expressions."""
    response = _post(
        {
            "type": "gradient",
            "expression": "x**2 + y**2",
            "variables": ["x", "y"],
            "steps": True,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["result"] == ["2*x", "2*y"]
    assert data["steps"]


def test_solve_endpoint_hessian() -> None:
    """Test Hessian matrix calculation for multivariate expressions."""
    response = _post(
        {
            "type": "hessian",
            "expression": "x**2 + y**2",
            "variables": ["x", "y"],
            "steps": False,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["result"] == [["2", "0"], ["0", "2"]]
    assert data["steps"] == []


def test_matrix_determinant_endpoint() -> None:
    response = _post(
        {
            "type": "matrix_determinant",
            "matrix": [[1, 2], [3, 4]],
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["result"] == "-2"
    assert data["steps"] == []


def test_matrix_inverse_endpoint() -> None:
    response = _post(
        {
            "type": "matrix_inverse",
            "matrix": [[1, 2], [3, 4]],
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["result"] == [["-2", "1"], ["3/2", "-1/2"]]


def test_matrix_multiply_endpoint() -> None:
    response = _post(
        {
            "type": "matrix_multiply",
            "left_matrix": [[1, 2], [3, 4]],
            "right_matrix": [[5, 6], [7, 8]],
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["result"] == [["19", "22"], ["43", "50"]]


def test_plot_endpoint() -> None:
    response = _post(
        {
            "type": "plot",
            "expression": "x**2",
            "variable": "x",
            "plot_range": ["0", "2"],
            "samples": 5,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "points" in data
    assert len(data["points"]) == 5
    assert data["points"][0]["x"] == 0.0
    assert data["points"][0]["y"] == 0.0


def test_ode_endpoint_with_initial_condition() -> None:
    response = _post(
        {
            "type": "ode",
            "expression": "Eq(diff(y(x), x), y(x))",
            "variable": "x",
            "function": "y",
            "initial_conditions": {"y(0)": "1"},
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "exp(x)" in data["result"]


def test_solve_endpoint_invalid_type() -> None:
    """Unsupported operation should return 400."""
    response = _post(
        {"type": "invalid", "expression": "x**2", "variable": "x", "steps": True},
    )
    assert response.status_code == 400
    assert response.json()["detail"].endswith("Unsupported operation type.")


def test_solve_endpoint_no_steps() -> None:
    """Test request without step-by-step explanation."""
    response = _post(
        {"type": "integral", "expression": "x", "variable": "x", "steps": False},
    )
    assert response.status_code == 200
    data = response.json()
    assert "result" in data
    assert "steps" in data
    assert data["result"] == "x**2/2"
    assert data["steps"] == []


def test_plain_limit_not_implemented() -> None:
    # Now that limit is implemented, test that it works
    response = _post(
        {
            "type": "limit",
            "expression": "sin(x)/x",
            "variable": "x",
            "point": "0",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "result" in data
    assert "steps" in data
    # The limit of sin(x)/x as x approaches 0 should be 1
    assert data["result"] == "1"


def test_series_endpoint_plain() -> None:
    response = _post(
        {
            "type": "series",
            "expression": "exp(x)",
            "variable": "x",
            "point": "0",
            "order": 5,
            "steps": True,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "result" in data


def test_missing_variable_validation_error() -> None:
    response = _post({"type": "integral", "expression": "x**2"})
    assert response.status_code == 422
    detail_messages = [entry["msg"] for entry in response.json()["detail"]]
    assert any(
        msg.endswith("Variable is required for the selected operation.")
        for msg in detail_messages
    )


def test_integration_plain_alias() -> None:
    response = _post(
        {
            "type": "integral",
            "expression": "x**2",
            "variable": "x",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "result" in data
    # Check that the result is mathematically correct
    # integral of x^2 = x^3/3 + C
    assert "x**3/3" in data["result"] or "x^3/3" in data["result"]


def test_differentiation_plain_alias() -> None:
    response = _post(
        {
            "type": "derivative",
            "expression": "x**3",
            "variable": "x",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "result" in data
    # Check that the result is mathematically correct
    # derivative of x^3 = 3*x^2
    assert "3*x**2" in data["result"] or "3*x^2" in data["result"]


def test_solve_plain_alias() -> None:
    response = _post(
        {
            "type": "solve",
            "expression": "x**2 - 4",
            "variable": "x",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "result" in data
    assert "2" in data["result"]
    assert "-2" in data["result"]


def test_limit_plain_alias() -> None:
    response = _post(
        {
            "type": "limit",
            "expression": "sin(x)/x",
            "variable": "x",
            "point": "0",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "result" in data
    # The limit of sin(x)/x as x approaches 0 is 1
    assert data["result"] == "1"


def test_gradient_requires_variables() -> None:
    response = _post(
        {
            "type": "gradient",
            "expression": "x**2 + y**2",
        },
    )
    assert response.status_code == 422
    detail_messages = [entry["msg"] for entry in response.json()["detail"]]
    assert any(
        msg.endswith("Variables list is required for multivariate operations.")
        for msg in detail_messages
    )


def test_gradient_normalizes_single_variable_string() -> None:
    response = _post(
        {
            "type": "gradient",
            "expression": "x**2",
            "variables": "x",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["result"] == ["2*x"]


def test_matrix_determinant_requires_matrix() -> None:
    response = _post(
        {
            "type": "matrix_determinant",
        },
    )
    assert response.status_code == 422
    detail_messages = [entry["msg"] for entry in response.json()["detail"]]
    assert any(
        msg.endswith("Matrix input is required for this operation.")
        for msg in detail_messages
    )


def test_matrix_multiply_requires_both_matrices() -> None:
    response = _post(
        {
            "type": "matrix_multiply",
            "left_matrix": [[1]],
        },
    )
    assert response.status_code == 422
    detail_messages = [entry["msg"] for entry in response.json()["detail"]]
    assert any(
        msg.endswith(
            "Both left_matrix and right_matrix are required for matrix multiplication.",
        )
        for msg in detail_messages
    )


def test_matrix_multiply_dimension_mismatch() -> None:
    response = _post(
        {
            "type": "matrix_multiply",
            "left_matrix": [[1, 2]],
            "right_matrix": [[3, 4]],
        },
    )
    assert response.status_code == 400
    assert "incompatible" in response.json()["detail"]


def test_plot_requires_range() -> None:
    response = _post(
        {
            "type": "plot",
            "expression": "x",
            "variable": "x",
        },
    )
    assert response.status_code == 422
    detail_messages = [entry["msg"] for entry in response.json()["detail"]]
    assert any(
        msg.endswith("Plot range must be provided as [start, end].")
        for msg in detail_messages
    )


def test_plot_requires_expression() -> None:
    response = _post(
        {
            "type": "plot",
            "variable": "x",
            "plot_range": ["0", "1"],
        },
    )
    assert response.status_code == 422
    detail_messages = [entry["msg"] for entry in response.json()["detail"]]
    assert any(
        msg.endswith("Expression is required for the selected operation.")
        for msg in detail_messages
    )


def test_ode_requires_function_variable_expression() -> None:
    response = _post(
        {
            "type": "ode",
            "expression": "Eq(diff(y(x), x), y(x))",
        },
    )
    assert response.status_code == 422
    detail_messages = [entry["msg"] for entry in response.json()["detail"]]
    assert any(msg.endswith("ODE variable is required.") for msg in detail_messages)


def test_ode_numeric_requires_range() -> None:
    response = _post(
        {
            "type": "ode",
            "expression": "Eq(diff(y(x), x), y(x)**2 + x)",
            "variable": "x",
            "function": "y",
            "numeric": True,
        },
    )
    assert response.status_code == 422
    detail_messages = [entry["msg"] for entry in response.json()["detail"]]
    assert any(
        msg.endswith("Numeric ODE solving requires numeric_range to be provided.")
        for msg in detail_messages
    )


def test_complex_conjugate_endpoint() -> None:
    response = _post(
        {
            "type": "complex_conjugate",
            "expression": "1 + 2*I",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["result"] == "1 - 2*I"


def test_complex_modulus_endpoint() -> None:
    response = _post(
        {
            "type": "complex_modulus",
            "expression": "3 + 4*I",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["result"] == "5"


def test_complex_argument_endpoint() -> None:
    response = _post(
        {
            "type": "complex_argument",
            "expression": "I",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["result"] == "pi/2"


def test_complex_to_polar_endpoint() -> None:
    response = _post(
        {
            "type": "complex_to_polar",
            "expression": "1 + I",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["magnitude"] == "sqrt(2)"
    assert data["angle"] == "pi/4"


def test_complex_from_polar_endpoint() -> None:
    response = _post(
        {
            "type": "complex_from_polar",
            "radius": "2",
            "angle": "pi/2",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["result"] == "2*I"


def test_stats_mean_endpoint() -> None:
    response = _post(
        {
            "type": "stats_mean",
            "values": [1, 2, 3, 4],
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["result"] == "5/2"


def test_stats_variance_population_endpoint() -> None:
    response = _post(
        {
            "type": "stats_variance",
            "values": [1, 2, 3],
            "sample": False,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["result"] == "2/3"


def test_stats_stddev_sample_endpoint() -> None:
    response = _post(
        {
            "type": "stats_stddev",
            "values": [1, 3, 5],
            "sample": True,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["result"] == "2"


def test_normal_pdf_endpoint() -> None:
    response = _post(
        {
            "type": "normal_pdf",
            "distribution_value": "0",
            "mean_value": "0",
            "std_value": "1",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["result"] == "sqrt(2)/(2*sqrt(pi))"


def test_normal_cdf_endpoint() -> None:
    response = _post(
        {
            "type": "normal_cdf",
            "distribution_value": "0",
            "mean_value": "0",
            "std_value": "1",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["result"] == "1/2"


def test_normal_pdf_requires_parameters() -> None:
    response = _post(
        {
            "type": "normal_pdf",
            "distribution_value": "0",
        },
    )
    assert response.status_code == 422


def test_stats_mean_requires_values() -> None:
    response = _post({"type": "stats_mean"})
    assert response.status_code == 422


def test_numeric_solve_endpoint() -> None:
    response = _post(
        {
            "type": "solve_numeric",
            "equations": ["x + y - 3", "x - y - 1"],
            "equation_variables": ["x", "y"],
            "initial_guess": [1, 1],
            "tolerance": 1e-9,
            "max_iterations": 200,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["solution"]["x"] == pytest.approx(2.0, rel=1e-6)
    assert data["solution"]["y"] == pytest.approx(1.0, rel=1e-6)


def test_numeric_solve_requires_equations() -> None:
    response = _post(
        {
            "type": "solve_numeric",
            "equation_variables": ["x"],
        },
    )
    assert response.status_code == 422
