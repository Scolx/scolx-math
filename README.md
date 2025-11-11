# Scolx Math API

Scolx Math API is a Python-based mathematical computation service that provides step-by-step solutions to mathematical problems. It supports plain-text mathematical expressions, offering integration, differentiation, equation solving, simplification, limits, series expansion, gradient, Hessian, matrix operations, differential equations, complex numbers, statistics, plotting, and numeric solving capabilities.

## Features

- **Plain-Text Operations**: Integration, differentiation, gradient and Hessian computation, matrix algebra (determinant, inverse, multiply), differential equation solving (analytic with SymPy or numeric via SciPy), complex number utilities, statistics/probability functions, numeric root solving, plotting samples, equation solving, and simplification using a hardened SymPy parser with optional SymEngine acceleration
- **Step-by-Step Explanations**: Optional breakdown for supported operations
- **RESTful API**: FastAPI-based web service with structured validation and error handling
- **Flexible Input**: Safe plain-text parsing
- **Health Checks**: Built-in health check endpoints for liveness, readiness, and startup

## Installation

### Prerequisites
- Python 3.14 or higher
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

4. (Recommended) Install development tooling and tests:
   ```bash
   pip install -r requirements-dev.txt
   ```

## Usage

### Running the API Server

```bash
uvicorn scolx_math.api.main:app --reload
```

The API will be available at `http://127.0.0.1:8000` with automatic reloading in development mode.

### Running with Docker

Build and run the container locally:

```bash
docker build -t scolx-math:latest .
docker run --rm -p 8000:8000 scolx-math:latest
```

The application will listen on port `8000`. Override the port by adjusting the `docker run` command if needed.

### API Documentation

Once running, you can access:
- Interactive API documentation at `http://127.0.0.1:8000/docs`
- Alternative API documentation at `http://127.0.0.1:8000/redoc`

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

 

#### Gradient (Plain Text)
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

#### Differential Equation Solving
```json
{
  "type": "ode",
  "expression": "Eq(diff(y(x), x), y(x))",
  "variable": "x",
  "function": "y",
  "initial_conditions": {"y(0)": "1"}
}
```

 

#### Statistics - Mean Calculation
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

### Using the Python Modules Directly

```python
from scolx_math.explain.explainers import differentiate_with_steps

result, steps = differentiate_with_steps("x**3", "x")
print(f"Result: {result}")
print("Steps:")
for step in steps:
    print(f"  - {step}")
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

## Testing

Run the test suite with:
```bash
pytest
```

All mathematical operations are now implemented including limits and series expansion for plain-text expressions. Integration, differentiation, and simplification automatically leverage SymEngine when available for faster results.

## Error Handling

The API performs rigorous request validation. Invalid payloads return `422 Unprocessable Entity` with detailed error messages. Unsupported operations return `400 Bad Request`, while plain-text limit and series requests currently return `501 Not Implemented`.

## Security Considerations

- Plain-text expressions are parsed with a constrained SymPy namespace to mitigate code-injection risks.
- Validation rules enforce required fields (e.g., variables, limit points).

## Dependencies

- `fastapi`: Web framework
- `uvicorn[standard]`: ASGI server
- `sympy`: Symbolic mathematics
- `symengine`: Optional acceleration backend used when available
- `numpy`: Numerical computations
- `scipy`: Scientific computing
- `pydantic`: Data validation
 

Development tooling additionally uses `pytest`, `httpx`, `black`, and `ruff` (see `requirements-dev.txt`).

## Contributing

We welcome contributions! Please review [CONTRIBUTING.md](CONTRIBUTING.md) for detailed developer guidelines and the end-to-end workflow.

> **Important:** Active development happens on [GitLab](https://gitlab.cherkaoui.ch/scolx/scolx-math). Issues or pull requests opened on GitHub are automatically closed—please submit contributions and bug reports through GitLab.

1. Fork the repository
2. Create a virtual environment
3. Install dependencies: `pip install -r requirements.txt -r requirements-dev.txt`
4. Make your changes
5. Run tests: `pytest`
6. Submit a pull request

## License

This project is licensed under the terms specified in the LICENSE file.

## Documentation

For comprehensive documentation, see the `docs/documentation.md` file in this repository.
