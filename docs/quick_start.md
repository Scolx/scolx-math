# Quick Start Guide for Scolx Math API

This guide will help you get started with the Scolx Math API quickly.

## 1. Installation

```bash
# Clone the repository
git clone https://gitlab.cherkaoui.ch/scolx/scolx-math.git
cd scolx-math

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 2. Running the API

```bash
# Start the development server
uvicorn scolx_math.api.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`

### Running with Docker

```bash
docker build -t scolx-math:latest .
docker run --rm -p 8000:8000 scolx-math:latest
```

See [docs/deployment.md](deployment.md) for production deployment guidance.

## 3. API Documentation

- Interactive API docs: `http://127.0.0.1:8000/docs`
- Redoc API docs: `http://127.0.0.1:8000/redoc`

## 4. Making API Requests

### Integration Example
```bash
curl -X POST "http://127.0.0.1:8000/solve" \
  -H "Content-Type: application/json" \
  -d '{
    "type": "integral",
    "expression": "x**2",
    "variable": "x",
    "steps": true
  }'
```

### LaTeX Integration Example
```bash
curl -X POST "http://127.0.0.1:8000/solve" \
  -H "Content-Type: application/json" \
  -d '{
    "type": "integral_latex",
    "expression": "x^2",
    "variable": "x",
    "steps": true,
    "is_latex": true
  }'
```

## 5. Python Usage

```python
from scolx_math.advanced_latex import integrate_latex_with_steps

# Perform integration with step-by-step explanation
result, steps = integrate_latex_with_steps("x^2", "x")
print(f"Result: {result}")
print("Steps:")
for step in steps:
    print(f"  - {step}")
```

## 6. Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=scolx_math
```

## 7. Running Benchmarks

Measure performance of heavy-weight operations to track optimization progress:

```bash
python -m benchmarks.heavy_operations
```

Optional arguments:

```bash
# Run only selected cases with custom iterations/warmup counts
python -m benchmarks.heavy_operations --case matrix_inverse_6x6 --iterations 10 --warmup 2
```

> **Note:** Numeric solver benchmarks require SciPy; cases will be skipped automatically when the dependency is missing.

## 8. Examples

Run the included examples to see the API in action:

```bash
python examples.py
```

## 8. API Endpoint Reference

### POST /solve

Request body:
```json
{
  "type": "string",           // Operation type
  "expression": "string",     // Mathematical expression
  "variable": "string",       // Variable name (optional for some operations)
  "point": "string",          // Point for limit operations (optional)
  "order": "int",             // Order for series expansion (default: 6)
  "steps": "bool",            // Include step-by-step explanation (default: true)
  "is_latex": "bool"          // Whether expression is in LaTeX format (default: false)
}
```

### Supported Operation Types

- `integral`: Calculate integral
- `derivative`: Calculate derivative
- `solve`: Solve equation
- `simplify`: Simplify expression
- `limit`: Calculate limit (fully implemented)
- `series`: Series expansion (fully implemented)
- `integral_latex`: LaTeX integral
- `derivative_latex`: LaTeX derivative
- `solve_latex`: LaTeX equation solving
- `limit_latex`: LaTeX limit
- `series_latex`: LaTeX series expansion

## 9. Project Structure

- `scolx_math/api/main.py`: Main FastAPI application
- `scolx_math/core/operations.py`: Basic mathematical operations
- `scolx_math/core/parsing.py`: Safe parsing helpers
- `scolx_math/explain/explainers.py`: Step-by-step explanations
- `scolx_math/advanced_latex.py`: LaTeX parsing and advanced operations
- `tests/test_api.py`: API endpoint tests
- `tests/test_latex.py`: LaTeX functionality tests
- `tests/test_smoke.py`: Basic integration test
- `docs/documentation.md`: Full documentation
- `examples.py`: Usage examples with both plain text and LaTeX