# Scolx Math API - Comprehensive Documentation

## Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Project Structure](#project-structure)
4. [API Endpoints](#api-endpoints)
5. [Core Modules](#core-modules)
6. [LaTeX Parsing](#latex-parsing)
7. [Usage Examples](#usage-examples)
8. [Running the Project](#running-the-project)
9. [Testing](#testing)
10. [Contributing](#contributing)
11. [Deployment](#deployment)
12. [Plotting](#plotting)

## Overview

Scolx Math API is a Python-based mathematical computation service that provides step-by-step solutions to mathematical problems. It supports both plain text and LaTeX mathematical expressions, offering integration, differentiation, equation solving, simplification, limits, and series expansion capabilities.

### Key Features
- **Mathematical Operations**: Integration, differentiation, equation solving, simplification
- **Step-by-Step Explanations**: Detailed explanations of how mathematical problems are solved
- **LaTeX Support**: Ability to parse and work with LaTeX mathematical expressions
- **RESTful API**: FastAPI-based web service for mathematical computations
- **Flexible Input**: Support for both plain text and LaTeX expressions

## Installation

### Prerequisites
- Python 3.11 or higher
- pip package manager

### Setup
1. Clone the repository:
   ```bash
   git clone https://gitlab.cherkaoui.ch/scolx/scolx-math.git
   cd scolx-math
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install development dependencies (optional):
   ```bash
   pip install -r requirements-dev.txt
   ```

## Project Structure

```
scolx-math/
├── scolx_math/                 # Main package
│   ├── __init__.py
│   ├── advanced_latex.py       # LaTeX parsing and advanced operations
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py             # FastAPI application and endpoints
│   ├── core/
│   │   ├── __init__.py
│   │   ├── operations.py       # Basic mathematical operations
│   │   └── parsing.py          # Safe parsing helpers
│   └── explain/
│       ├── __init__.py
│       └── explainers.py       # Step-by-step explanation generators
├── tests/                      # Test suite
│   ├── test_api.py             # API endpoint tests
│   ├── test_latex.py           # LaTeX functionality tests
│   └── test_smoke.py           # Basic integration test
├── docs/                       # Documentation
│   ├── documentation.md        # Comprehensive documentation
│   ├── quick_start.md          # Quick start guide
│   └── README.md
├── pyproject.toml              # Project configuration
├── requirements.txt            # Production dependencies
├── requirements-dev.txt        # Development dependencies
├── examples.py                 # Usage examples
├── TODO.md                     # Development roadmap
└── CONTRIBUTING.md             # Contribution guidelines
```

## API Endpoints

### POST /solve

The main endpoint for mathematical computations. Accepts a JSON payload with the following structure:

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

### Operation Types

| Type | Description | Required Fields |
|------|-------------|-----------------|
| `integral` | Calculate integral | `expression`, `variable` |
| `derivative` | Calculate derivative | `expression`, `variable` |
| `solve` | Solve equation | `expression`, `variable` |
| `simplify` | Simplify expression | `expression` |
| `limit` | Calculate limit (implemented) | `expression`, `variable`, `point` |
| `series` | Series expansion (implemented) | `expression`, `variable`, `point`, `order` |
| `integral_latex` | LaTeX integral | `expression`, `variable`, `is_latex: true` |
| `derivative_latex` | LaTeX derivative | `expression`, `variable`, `is_latex: true` |
| `solve_latex` | LaTeX equation solving | `expression`, `variable`, `is_latex: true` |
| `limit_latex` | LaTeX limit | `expression`, `variable`, `point`, `is_latex: true` |
| `series_latex` | LaTeX series expansion | `expression`, `variable`, `point`, `order`, `is_latex: true` |

### Example Requests

#### Integration (Plain Text)
```json
{
  "type": "integral",
  "expression": "x**2",
  "variable": "x",
  "steps": true
}
```

#### Integration (LaTeX)
```json
{
  "type": "integral_latex",
  "expression": "x^2",
  "variable": "x",
  "steps": true,
  "is_latex": true
}
```

#### Equation Solving (LaTeX)
```json
{
  "type": "solve_latex",
  "expression": "x^2 - 4",
  "variable": "x",
  "steps": true,
  "is_latex": true
}
```

### Response Format

Successful responses follow this format:
```json
{
  "result": "string or array",  // Computed result
  "steps": ["string"]          // Step-by-step explanation (if requested)
}
```

Error responses:
```json
{
  "detail": "string"           // Error message
}
```

## Core Modules

### 1. Core Operations (`scolx_math/core/operations.py`)

This module contains basic mathematical operations using SymPy:

- `solve_equation(expr_str, var_str)`: Solve equations
- `integrate_expr(expr_str, var_str)`: Calculate integrals
- `differentiate_expr(expr_str, var_str)`: Calculate derivatives
- `simplify_expr(expr_str)`: Simplify expressions

### 2. Safe Parsing Helpers (`scolx_math/core/parsing.py`)

Provides safe parsing of user-supplied mathematical expressions:

- `parse_plain_expression(expr_str, variables)`: Parse plain text expression with safe namespace
- `validate_variable_name(name)`: Validate variable names to prevent code injection

### 3. Explanation Generator (`scolx_math/explain/explainers.py`)

Provides step-by-step explanations for mathematical operations:

- `integrate_with_steps(expr_str, var_str)`: Integration with explanation steps

### 4. LaTeX Processing (`scolx_math/advanced_latex.py`)

Handles LaTeX parsing and advanced mathematical operations:

- `parse_latex_expression(latex_expr)`: Convert LaTeX to SymPy expression
- `integrate_latex_with_steps(latex_expr, var_name)`: LaTeX integration with steps
- `solve_equation_latex_with_steps(latex_eq, var_name)`: LaTeX equation solving with steps
- `differentiate_latex_with_steps(latex_expr, var_name)`: LaTeX differentiation with steps
- `limit_latex_with_steps(latex_expr, var_name, point)`: LaTeX limit calculation with steps
- `series_latex_with_steps(latex_expr, var_name, point, order)`: LaTeX series expansion with steps

### 5. API (`scolx_math/api/main.py`)

The FastAPI application that exposes the mathematical operations via REST endpoints.

## LaTeX Parsing

The Scolx Math API includes robust LaTeX parsing capabilities using SymPy's experimental LaTeX parser. This allows users to input mathematical expressions in LaTeX format, which are then converted to SymPy expressions for computation.

### Supported LaTeX Features

- Basic mathematical expressions: `x^2`, `\frac{a}{b}`, `\sqrt{x}`
- Trigonometric functions: `\sin(x)`, `\cos(x)`, `\tan(x)`
- Logarithmic functions: `\log(x)`, `\ln(x)`
- Exponential functions: `e^x`, `\exp(x)`
- Integration notation: `\int x^2 dx`

### Implementation Details

The LaTeX parsing functionality is implemented using `sympy.parsing.latex.parse_latex()`. Note that this module is still experimental in some SymPy versions, so results may vary depending on the complexity of the LaTeX expression.

## Usage Examples

### Running Examples

The project includes example code in `examples.py`:

```python
from scolx_math.api.main import app
from scolx_math.advanced_latex import (
    integrate_latex_with_steps,
    differentiate_latex_with_steps,
    solve_equation_latex_with_steps
)

def example_plain_text():
    """Examples using plain text mathematical expressions."""
    print("=== Plain Text Examples ===")

    # Integration example
    from scolx_math.explain.explainers import integrate_with_steps
    result, steps = integrate_with_steps("x**2", "x")
    print(f"Integral of x^2: {result}")
    print(f"Steps: {steps}")
    print()

    # Differentiation example
    from scolx_math.core.operations import differentiate_expr
    result = differentiate_expr("x**3", "x")
    print(f"Derivative of x^3: {result}")
    print()

    # Equation solving example
    from scolx_math.core.operations import solve_equation
    result = solve_equation("x**2 - 4", "x")
    print(f"Solutions to x^2 - 4 = 0: {result}")
    print()

def example_latex():
    """Examples using LaTeX mathematical expressions."""
    print("=== LaTeX Examples ===")

    # Integration with LaTeX
    result, steps = integrate_latex_with_steps("x^2", "x")
    print(f"Integral of x^2 (from LaTeX): {result}")
    print(f"Steps: {steps}")
    print()

    # Differentiation with LaTeX
    result, steps = differentiate_latex_with_steps("x^3", "x")
    print(f"Derivative of x^3 (from LaTeX): {result}")
    print(f"Steps: {steps}")
    print()

    # Equation solving with LaTeX
    result, steps = solve_equation_latex_with_steps("x^2 - 4", "x")
    print(f"Solutions to x^2 - 4 = 0 (from LaTeX): {result}")
    print(f"Steps: {steps}")
    print()

if __name__ == "__main__":
    example_plain_text()
    example_latex()
    print("All examples completed successfully!")
```

### Running the Examples
```bash
python examples.py
```

## Running the Project

### Development Server

To run the API in development mode:

```bash
uvicorn scolx_math.api.main:app --reload
```

This will start the server on `http://127.0.0.1:8000` with automatic reloading when code changes.

### Production Server

For production deployment:

```bash
uvicorn scolx_math.api.main:app --host 0.0.0.0 --port 8000
```

### API Documentation

Once the server is running, you can access:
- Interactive API documentation at `http://127.0.0.1:8000/docs`
- Alternative API documentation at `http://127.0.0.1:8000/redoc`

## Testing

The project includes a comprehensive test suite using pytest:

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_api.py

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=scolx_math
```

### Test Structure

- `tests/test_api.py`: Tests for basic API endpoints and operations
- `tests/test_latex.py`: Tests for LaTeX parsing functionality
- `tests/test_smoke.py`: Basic smoke tests

## Deployment

For production deployment instructions, see the dedicated guide at
[`docs/deployment.md`](deployment.md). It covers image builds, runtime
configuration, Docker Compose examples, and GitLab CI/CD integration.

## Plotting

Use the `plot` operation to sample an expression over a range. Provide the
expression, variable, `plot_range` (start/end), and optional `samples` count.
The API returns a list of `{ "x": value, "y": value }` points that can be fed
into charting libraries.

## Dependencies

### Production Dependencies
- `fastapi`: Web framework
- `uvicorn[standard]`: ASGI server
- `sympy`: Symbolic mathematics
- `symengine`: Fast symbolic manipulation
- `pydantic`: Data validation
- `antlr4-python3-runtime==4.11`: ANTLR runtime for LaTeX parsing

### Development Dependencies
- `pytest`: Testing framework
- `httpx`: HTTP client for testing
- `black`: Code formatter
- `ruff`: Linter and code quality checker

## Contributing

### Development Setup

1. Fork and clone the repository
2. Create a virtual environment
3. Install dependencies: `pip install -r requirements.txt -r requirements-dev.txt`
4. Make your changes
5. Run tests: `pytest`
6. Format code: `black .` and `ruff check --fix .`
7. Submit a pull request

### Code Standards

- Follow PEP 8 style guide
- Use type hints where appropriate
- Write docstrings for all public functions
- Include tests for new functionality
- Update documentation as needed

## License

This project is licensed under the terms specified in the LICENSE file.

## Project Status

This is an active project under development. See the TODO.md file for planned features and improvements.