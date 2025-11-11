# Scolx Math API - Comprehensive Documentation

## Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Project Structure](#project-structure)
4. [API Endpoints](#api-endpoints)
5. [Core Modules](#core-modules)
6. [Usage Examples](#usage-examples)
7. [Running the Project](#running-the-project)
8. [Testing](#testing)
9. [Contributing](#contributing)
10. [Deployment](#deployment)
11. [Plotting](#plotting)

## Overview

Scolx Math API is a Python-based mathematical computation service that provides step-by-step solutions to mathematical problems. It supports plain-text mathematical expressions, offering integration, differentiation, equation solving, simplification, limits, and series expansion capabilities.

### Key Features
- **Mathematical Operations**: Integration, differentiation, equation solving, simplification
- **Step-by-Step Explanations**: Detailed explanations of how mathematical problems are solved
- **RESTful API**: FastAPI-based web service for mathematical computations
- **Flexible Input**: Support for plain-text expressions

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
  "variables": ["string"],    // Variables for multivariate operations
  "point": "string",          // Point for limit operations (optional)
  "order": "int",             // Order for series expansion (default: 6)
  "steps": "bool",            // Include step-by-step explanation (default: true)

  "plot_range": ["string", "string"],  // Start and end for plotting
  "samples": "int",           // Sample count for plotting (default: 100)
  "matrix": [[...]],          // Matrix input for determinant/inverse
  "left_matrix": [[...]],     // Left matrix for multiplication
  "right_matrix": [[...]],    // Right matrix for multiplication
  "function": "string",       // Function name for ODE solving
  "initial_conditions": {...}, // Initial conditions for ODE solving
  "values": [...],            // Values for statistical calculations
  "sample": "bool",           // Use sample statistics (default: false)
  "distribution_value": "string", // Value for probability distributions
  "mean_value": "string",     // Mean parameter for probability distributions
  "std_value": "string",      // Standard deviation parameter for probability distributions
  "equations": ["string"],    // System of equations for numeric solving
  "equation_variables": ["string"], // Variables for numeric solving
  "initial_guess": [...],     // Initial guess values for numeric solving
  "max_iterations": "int",    // Maximum iterations for numeric solvers (default: 100)
  "tolerance": "float"        // Tolerance for numeric solvers (default: 1e-9)
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
| `gradient` | Calculate gradient of multivariate expression | `expression`, `variables` |
| `hessian` | Calculate Hessian matrix of multivariate expression | `expression`, `variables` |
| `matrix_determinant` | Compute determinant of a matrix | `matrix` |
| `matrix_inverse` | Compute inverse of a matrix | `matrix` |
| `matrix_multiply` | Multiply two matrices | `left_matrix`, `right_matrix` |
| `ode` | Solve ordinary differential equation | `expression`, `function`, `variable` |
| `plot` | Generate points for plotting | `expression`, `variable`, `plot_range` |
| `complex_conjugate` | Compute complex conjugate | `expression` |
| `complex_modulus` | Compute complex modulus | `expression` |
| `complex_argument` | Compute complex argument | `expression` |
| `complex_to_polar` | Convert complex to polar form | `expression` |
| `complex_from_polar` | Convert polar to complex form | `radius`, `angle` |
| `stats_mean` | Compute statistical mean | `values` |
| `stats_variance` | Compute statistical variance | `values` |
| `stats_stddev` | Compute statistical standard deviation | `values` |
| `normal_pdf` | Compute normal probability density function | `distribution_value`, `mean_value`, `std_value` |
| `normal_cdf` | Compute normal cumulative distribution function | `distribution_value`, `mean_value`, `std_value` |
| `solve_numeric` | Solve system of equations numerically | `equations`, `equation_variables` |


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



#### Gradient Calculation
```json
{
  "type": "gradient",
  "expression": "x**2 + y**2",
  "variables": ["x", "y"],
  "steps": true
}
```

#### Matrix Multiplication
```json
{
  "type": "matrix_multiply",
  "left_matrix": [[1, 2], [3, 4]],
  "right_matrix": [[5, 6], [7, 8]]
}
```

#### ODE Solving
```json
{
  "type": "ode",
  "expression": "Eq(diff(y(x), x), y(x))",
  "variable": "x",
  "function": "y",
  "initial_conditions": {"y(0)": "1"}
}
```

#### Statistics - Mean
```json
{
  "type": "stats_mean",
  "values": [1, 2, 3, 4, 5]
}
```

#### Complex Number - Conjugate
```json
{
  "type": "complex_conjugate",
  "expression": "3 + 4*I"
}
```

#### Plotting
```json
{
  "type": "plot",
  "expression": "x**2",
  "variable": "x",
  "plot_range": ["-5", "5"],
  "samples": 100
}
```

#### Numeric Solving
```json
{
  "type": "solve_numeric",
  "equations": ["x**2 + y**2 - 4", "x - y"],
  "equation_variables": ["x", "y"]
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



### 5. API (`scolx_math/api/main.py`)

The FastAPI application that exposes the mathematical operations via REST endpoints.



## Usage Examples

### Running Examples

The project includes example code in `examples.py`:

```python
from scolx_math.api.main import app
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

if __name__ == "__main__":
    example_plain_text()
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