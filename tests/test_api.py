"""Tests for the FastAPI endpoints."""

from typing import Any

from fastapi.testclient import TestClient

from scolx_math.api.main import app

client = TestClient(app)


def _post(payload: dict[str, Any]):
    return client.post("/solve", json=payload)


def test_solve_endpoint_integral():
    """Test integral calculation."""
    response = _post(
        {
            "type": "integral",
            "expression": "x**2",
            "variable": "x",
            "steps": True,
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["result"] == "x**3/3"
    assert data["steps"][0] == "Parsing expression and simplifying"


def test_solve_endpoint_derivative():
    """Test derivative calculation."""
    response = _post(
        {
            "type": "derivative",
            "expression": "x**3",
            "variable": "x",
            "steps": True,
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "result" in data
    assert "steps" in data
    assert data["result"] == "3*x**2"


def test_solve_endpoint_solve():
    """Test equation solving."""
    response = _post(
        {
            "type": "solve",
            "expression": "x**2 - 4",
            "variable": "x",
            "steps": True,
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "result" in data
    assert "steps" in data
    assert "-2" in data["result"]
    assert "2" in data["result"]


def test_solve_endpoint_simplify():
    """Test expression simplification."""
    response = _post(
        {
            "type": "simplify",
            "expression": "(x+1)**2 - x**2 - 2*x - 1",
            "steps": True,
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "result" in data
    assert "steps" in data
    assert data["result"] == "0"


def test_solve_endpoint_gradient():
    """Test gradient calculation for multivariate expressions."""
    response = _post(
        {
            "type": "gradient",
            "expression": "x**2 + y**2",
            "variables": ["x", "y"],
            "steps": True,
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["result"] == ["2*x", "2*y"]
    assert data["steps"]


def test_solve_endpoint_hessian():
    """Test Hessian matrix calculation for multivariate expressions."""
    response = _post(
        {
            "type": "hessian",
            "expression": "x**2 + y**2",
            "variables": ["x", "y"],
            "steps": False,
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["result"] == [["2", "0"], ["0", "2"]]
    assert data["steps"] == []


def test_solve_endpoint_invalid_type():
    """Unsupported operation should return 400."""
    response = _post(
        {"type": "invalid", "expression": "x**2", "variable": "x", "steps": True}
    )
    assert response.status_code == 400
    assert response.json()["detail"].endswith("Unsupported operation type.")


def test_solve_endpoint_no_steps():
    """Test request without step-by-step explanation."""
    response = _post(
        {"type": "integral", "expression": "x", "variable": "x", "steps": False}
    )
    assert response.status_code == 200
    data = response.json()
    assert "result" in data
    assert "steps" in data
    assert data["result"] == "x**2/2"
    assert data["steps"] == []


def test_plain_limit_not_implemented():
    # Now that limit is implemented, test that it works
    response = _post(
        {
            "type": "limit",
            "expression": "sin(x)/x",
            "variable": "x",
            "point": "0",
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "result" in data
    assert "steps" in data
    # The limit of sin(x)/x as x approaches 0 should be 1
    assert data["result"] == "1"


def test_missing_variable_validation_error():
    response = _post({"type": "integral", "expression": "x**2"})
    assert response.status_code == 422
    detail_messages = [entry["msg"] for entry in response.json()["detail"]]
    assert any(
        msg.endswith("Variable is required for the selected operation.")
        for msg in detail_messages
    )


def test_latex_requires_flag():
    response = _post(
        {
            "type": "integral_latex",
            "expression": "x^2",
            "variable": "x",
            "is_latex": False,
        }
    )
    assert response.status_code == 422
    detail_messages = [entry["msg"] for entry in response.json()["detail"]]
    assert any(
        msg.endswith("Set is_latex=true for LaTeX-specific operations.")
        for msg in detail_messages
    )


def test_plain_operation_rejects_latex_flag():
    response = _post(
        {
            "type": "integral",
            "expression": "x**2",
            "variable": "x",
            "is_latex": True,
        }
    )
    assert response.status_code == 422
    detail_messages = [entry["msg"] for entry in response.json()["detail"]]
    assert any(
        msg.endswith("Plain-text operations must not set is_latex=true.")
        for msg in detail_messages
    )


def test_limit_latex_requires_point():
    response = _post(
        {
            "type": "limit_latex",
            "expression": "\\frac{\\sin(x)}{x}",
            "variable": "x",
            "is_latex": True,
        }
    )
    assert response.status_code == 422
    detail_messages = [entry["msg"] for entry in response.json()["detail"]]
    assert any(
        msg.endswith("Point is required for limit or series operations.")
        for msg in detail_messages
    )


def test_gradient_requires_variables():
    response = _post(
        {
            "type": "gradient",
            "expression": "x**2 + y**2",
        }
    )
    assert response.status_code == 422
    detail_messages = [entry["msg"] for entry in response.json()["detail"]]
    assert any(
        msg.endswith("Variables list is required for multivariate operations.")
        for msg in detail_messages
    )


def test_gradient_normalizes_single_variable_string():
    response = _post(
        {
            "type": "gradient",
            "expression": "x**2",
            "variables": "x",
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["result"] == ["2*x"]
