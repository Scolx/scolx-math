# Scolx Math API

Scolx Math API is a Python-based mathematical computation service that provides step-by-step solutions to mathematical problems. It supports both plain text and LaTeX mathematical expressions, offering integration, differentiation, equation solving, simplification, limits, and series expansion capabilities.

## Features

- **Plain-Text Operations**: Integration, differentiation, gradient and Hessian computation, equation solving, and simplification using a hardened SymPy parser with optional SymEngine acceleration
- **LaTeX Operations**: Integration, differentiation, solving, limits, and series expansion with detailed steps
- **Step-by-Step Explanations**: Optional breakdown for supported operations
- **RESTful API**: FastAPI-based web service with structured validation and error handling
- **Flexible Input**: Safe plain-text parsing plus dedicated LaTeX workflows

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

#### Gradient (Plain Text)
```json
{
  "type": "gradient",
  "expression": "x**2 + y**2",
  "variables": ["x", "y"],
  "steps": true
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

### Using the Python Modules Directly

```python
from scolx_math.advanced_latex import integrate_latex_with_steps

result, steps = integrate_latex_with_steps("x^2", "x")
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

## Testing

Run the test suite with:
```bash
pytest
```

All mathematical operations are now implemented including limits and series expansion for both plain text and LaTeX expressions. Integration, differentiation, and simplification automatically leverage SymEngine when available for faster results.

## Error Handling

The API performs rigorous request validation. Invalid payloads return `422 Unprocessable Entity` with detailed error messages. Unsupported operations return `400 Bad Request`, while plain-text limit and series requests currently return `501 Not Implemented`.

## Security Considerations

- Plain-text expressions are parsed with a constrained SymPy namespace to mitigate code-injection risks.
- Validation rules enforce required fields (e.g., variables, limit points) and prevent mismatched LaTeX/plain-text combinations.

## Dependencies

- `fastapi`: Web framework
- `uvicorn[standard]`: ASGI server
- `sympy`: Symbolic mathematics
- `symengine`: Optional acceleration backend used when available
- `pydantic`: Data validation
- `antlr4-python3-runtime==4.11`: ANTLR runtime for LaTeX parsing

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