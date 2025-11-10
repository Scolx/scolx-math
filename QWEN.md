# scolx-math Project Documentation

## Project Overview

scolx-math is a Python-based mathematics API that provides symbolic math capabilities with step-by-step explanations. The project is designed to be used by AI assistants to solve mathematical problems with detailed explanations of the solution process.

The core functionality includes:
- **Integration** with step-by-step explanations
- **Equation solving**
- **Differentiation**
- **Expression simplification**
- **FastAPI-based REST API** for integration with AI systems

The project uses SymPy for symbolic computation and FastAPI for the web interface, with Pydantic for data validation.

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
├── pyproject.toml              # Project configuration (ruff, black, pytest)
├── requirements.txt            # Production dependencies
├── requirements-dev.txt        # Development dependencies
├── examples.py                 # Usage examples
├── TODO.md                     # Development roadmap
├── CONTRIBUTING.md             # Contribution guidelines
└── README.md
```

## Dependencies

### Runtime Dependencies
- **fastapi**: Web framework for the API
- **uvicorn[standard]**: ASGI server for running the API
- **sympy**: Symbolic mathematics library
- **symengine**: Optional faster symbolic computations
- **numpy**: Numerical computations
- **scipy**: Scientific computing
- **pydantic**: Data validation and settings management
- **antlr4-python3-runtime==4.11**: ANTLR runtime for LaTeX parsing

### Development Dependencies
- **black**: Code formatter
- **ruff**: Fast Python linter

## Key Features

### Core Mathematical Operations
- `scolx_math.core.operations`: Provides basic symbolic math functions:
  - `solve_equation(expr, var)`: Solve equations
  - `integrate_expr(expr, var)`: Perform integration
  - `differentiate_expr(expr, var)`: Perform differentiation
  - `simplify_expr(expr)`: Simplify expressions

### Step-by-Step Explanations
- `scolx_math.explain.explainers`: Provides functions that generate detailed steps:
  - `integrate_with_steps(expr, var)`: Integration with explanation steps

### LaTeX Processing
- `scolx_math.advanced_latex`: New module for LaTeX parsing and advanced operations:
  - `parse_latex_expression(latex_expr)`: Convert LaTeX to SymPy expression
  - `integrate_latex_with_steps(latex_expr, var_name)`: LaTeX integration with steps
  - `solve_equation_latex_with_steps(latex_eq, var_name)`: LaTeX equation solving with steps
  - `differentiate_latex_with_steps(latex_expr, var_name)`: LaTeX differentiation with steps
  - `limit_latex_with_steps(latex_expr, var_name, point)`: LaTeX limit calculation with steps
  - `series_latex_with_steps(latex_expr, var_name, point, order)`: LaTeX series expansion with steps

### API Endpoints
- `/solve`: Main endpoint that accepts mathematical problems and returns solutions with optional step-by-step explanations
- Supports different problem types: integrals, derivatives, equation solving, simplification, limits, series expansion
- Supports both plain text and LaTeX expressions via the `is_latex` flag
- New parameters: `point` for limit operations, `order` for series expansion
- Uses Pydantic models for request/response validation

### LaTeX Parsing Capabilities
- Advanced LaTeX parsing using `sympy.parsing.latex.parse_latex()`
- Support for mathematical expressions in LaTeX format
- New `advanced_latex` module handles LaTeX-specific operations
- Available operations with LaTeX: integration, differentiation, equation solving, limits, series expansion

## Building and Running

### Setting up Development Environment

1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Running the API

```bash
uvicorn scolx_math.api.main:app --reload
```

The API will be available at `http://localhost:8000`

### Running Tests

```bash
pytest
```

### Code Formatting and Linting

Format code:
```bash
black .
```

Lint code:
```bash
ruff check .
```

Fix lint issues automatically where possible:
```bash
ruff check . --fix
```

## API Usage

### Example Request (Plain Text)

```json
{
  "type": "integral",
  "expression": "x**2",
  "variable": "x",
  "steps": true
}
```

### Example Request (LaTeX)

```json
{
  "type": "integral_latex",
  "expression": "x^2",
  "variable": "x",
  "steps": true,
  "is_latex": true
}
```

### Example Response

```json
{
  "result": "x**3/3",
  "steps": [
    "Parsing LaTeX expression: x^2",
    "Identifying variable of integration: x",
    "Simplifying expression: x**2",
    "Performing integration with respect to x",
    "Applying integration rules...",
    "Final result: x**3/3"
  ]
}
```

## Development Conventions

### Code Style
- Code formatting follows Black standards (88 character line length)
- Linting follows Ruff rules (similar to PEP 8)
- Type hints are encouraged for function parameters and return values

### Testing
- Tests are located in the `tests/` directory
- Use pytest for testing framework
- Test files should be named with `test_` prefix

### Import Structure
- Use explicit imports where possible
- Organize imports in standard library, third-party, then local packages
- Use absolute imports from the package root

## Planned Features (Based on Steps.md)

According to the implementation plan in `Steps.md`, future enhancements may include:
- Natural language explanations
- Multi-step problem support 
- Numeric fallback for unsolvable symbolic problems
- Logging and error tracking
- Performance optimizations with SymEngine

## Current Status

The project has a basic working implementation with:
- Integration with step-by-step explanations
- FastAPI web interface
- Basic test coverage
- Proper project structure with modular components
- Code quality tools (black, ruff) configured