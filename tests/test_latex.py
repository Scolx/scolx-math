"""Tests for LaTeX parsing and advanced mathematical operations."""

from fastapi.testclient import TestClient

from scolx_math.api.main import app

client = TestClient(app)


def test_latex_integration():
    """Test integral calculation with LaTeX input."""
    response = client.post(
        "/solve",
        json={
            "type": "integral_latex",
            "expression": "\\int x^2 \\, dx",
            "variable": "x",
            "steps": True,
            "is_latex": True,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "result" in data
    assert "steps" in data
    # The result should be equivalent to x^3/3 (though possibly in different form)


def test_latex_differentiation():
    """Test derivative calculation with LaTeX input."""
    response = client.post(
        "/solve",
        json={
            "type": "derivative_latex",
            "expression": "x^3",
            "variable": "x",
            "steps": True,
            "is_latex": True,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "result" in data
    assert "steps" in data
    # The result should be equivalent to 3*x^2


def test_latex_equation_solving():
    """Test equation solving with LaTeX input."""
    response = client.post(
        "/solve",
        json={
            "type": "solve_latex",
            "expression": "x^2 - 4",
            "variable": "x",
            "steps": True,
            "is_latex": True,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "result" in data
    assert "steps" in data
    assert len(data["result"]) >= 2  # Should have at least two solutions


def test_latex_limit():
    """Test limit calculation with LaTeX input."""
    response = client.post(
        "/solve",
        json={
            "type": "limit_latex",
            "expression": "\\frac{\\sin(x)}{x}",
            "variable": "x",
            "point": "0",
            "steps": True,
            "is_latex": True,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "result" in data
    assert "steps" in data


def test_latex_series():
    """Test series expansion with LaTeX input."""
    response = client.post(
        "/solve",
        json={
            "type": "series_latex",
            "expression": "e^x",
            "variable": "x",
            "point": "0",
            "order": 5,
            "steps": True,
            "is_latex": True,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "result" in data
    assert "steps" in data
