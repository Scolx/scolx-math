from fastapi.testclient import TestClient

from scolx_math.api.main import app

client = TestClient(app)


def test_plain_integration() -> None:
    response = client.post(
        "/solve",
        json={
            "type": "integral",
            "expression": "x**2",
            "variable": "x",
            "steps": True,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "result" in data
    assert "steps" in data


def test_plain_derivative_no_steps() -> None:
    response = client.post(
        "/solve",
        json={
            "type": "derivative",
            "expression": "(x**2 + 1)",
            "variable": "x",
            "steps": False,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["result"] == "2*x"


def test_plain_equation_solving() -> None:
    response = client.post(
        "/solve",
        json={
            "type": "solve",
            "expression": "x**2 - 4",
            "variable": "x",
            "steps": True,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "result" in data
    assert len(data["result"]) >= 2


def test_plain_limit_and_series() -> None:
    response = client.post(
        "/solve",
        json={
            "type": "limit",
            "expression": "sin(x)/x",
            "variable": "x",
            "point": "0",
            "steps": True,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "result" in data

    response = client.post(
        "/solve",
        json={
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
