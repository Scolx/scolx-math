# scolx-math Project Documentation

## Project Overview

scolx-math is a Python-based mathematics API that provides symbolic math capabilities with step-by-step explanations. The project is designed to be used by AI assistants to solve mathematical problems with detailed explanations of the solution process.

The core functionality includes:
- **Integration** with step-by-step explanations
- **Equation solving**
- **Differentiation**
- **Expression simplification**
- **Multivariate calculus** (gradient and Hessian)
- **Matrix operations** (determinant, inverse, multiplication)
- **Differential equations** (analytical and numerical)
- **Complex number operations**
- **Statistics and probability functions**
- **Plotting and visualization**
- **Numeric solvers**
- **FastAPI-based REST API** for integration with AI systems

The project uses SymPy for symbolic computation and FastAPI for the web interface, with Pydantic for data validation.

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
  - `limit_expr(expr, var, point)`: Calculate limits
  - `series_expr(expr, var, point, order)`: Calculate series expansion
  - `solve_ode(equation, function, variable, initial_conditions, numeric, ...)` : Solve ordinary differential equations (analytical and numerical)
  - `matrix_determinant(matrix)`: Compute determinant of a matrix
  - `matrix_inverse(matrix)`: Compute inverse of a matrix
  - `matrix_multiply(left, right)`: Multiply two matrices
  - `complex_conjugate_expr(expr)`: Compute complex conjugate
  - `complex_modulus_expr(expr)`: Compute complex modulus
  - `complex_argument_expr(expr)`: Compute complex argument
  - `complex_to_polar_expr(expr)`: Convert complex to polar form
  - `complex_from_polar_expr(radius, angle)`: Convert polar to complex form
  - `stats_mean(values)`: Compute statistical mean
  - `stats_variance(values, sample)`: Compute statistical variance
  - `stats_standard_deviation(values, sample)`: Compute statistical standard deviation
  - `normal_pdf(value, mean, std)`: Compute normal probability density function
  - `normal_cdf(value, mean, std)`: Compute normal cumulative distribution function
  - `solve_system_numeric(equations, variables, initial_guess, max_iterations, tolerance)`: Solve system of equations numerically
  - `generate_plot_points(expr, variable, start, end, samples)`: Generate points for plotting

### Step-by-Step Explanations
- `scolx_math.explain.explainers`: Provides functions that generate detailed steps:
  - `integrate_with_steps(expr, var)`: Integration with explanation steps
  - `differentiate_with_steps(expr, var)`: Differentiation with explanation steps
  - `solve_with_steps(expr, var)`: Equation solving with explanation steps
  - `limit_with_steps(expr, var, point)`: Limit calculation with explanation steps
  - `series_with_steps(expr, var, point, order)`: Series expansion with explanation steps
  - `simplify_with_steps(expr)`: Simplification with explanation steps



### API Endpoints
- `/solve`: Main endpoint that accepts mathematical problems and returns solutions with optional step-by-step explanations
- `/livez`: Liveness probe
- `/readyz`: Readiness probe
- `/startupz`: Startup probe
- Supports different problem types: integrals, derivatives, equation solving, simplification, limits, series expansion, gradient, Hessian, matrix operations, ODE solving, complex numbers, statistics, plotting, and numeric solving (all implemented for plain-text expressions)
- Supports plain-text expressions
- New parameters: `point` for limit operations, `order` for series expansion, `variables` for multivariate operations, `plot_range` for plotting, `matrix` for matrix operations, and many more
- Uses Pydantic models for request/response validation
- Comprehensive validation and error handling for all operation types



### Plain-Text Operations
- All mathematical operations now fully implemented for plain-text expressions
- Integration, differentiation, equation solving, simplification, limits, series expansion, gradient, Hessian, matrix operations, ODE solving, complex numbers, statistics, plotting, and numeric solving
- Safe expression parsing with whitelisted functions and constants
- Optional SymEngine acceleration for faster computations

### Multivariate Calculus
- `gradient` operation: Calculate gradient of multivariate expressions
- `hessian` operation: Calculate Hessian matrix of multivariate expressions

### Matrix Operations
- `matrix_determinant`: Compute determinant of a matrix
- `matrix_inverse`: Compute inverse of a matrix
- `matrix_multiply`: Multiply two matrices

### Differential Equations
- `ode` operation: Solve ordinary differential equations (analytical and numerical)
- Support for initial conditions and numeric solving

### Complex Numbers
- `complex_conjugate`: Compute complex conjugate
- `complex_modulus`: Compute complex modulus
- `complex_argument`: Compute complex argument
- `complex_to_polar`: Convert complex to polar form
- `complex_from_polar`: Convert polar to complex form

### Statistics and Probability
- `stats_mean`: Compute statistical mean
- `stats_variance`: Compute statistical variance
- `stats_stddev`: Compute statistical standard deviation
- `normal_pdf`: Compute normal probability density function
- `normal_cdf`: Compute normal cumulative distribution function

### Plotting and Visualization
- `plot` operation: Generate points for plotting expressions over a range
- Returns list of {x, y} points that can be used with charting libraries

### Numeric Solvers
- `solve_numeric` operation: Solve systems of equations numerically
- Support for initial guesses, maximum iterations, and tolerance settings

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

### Example Request (Limit - Plain Text)

```json
{
  "type": "limit",
  "expression": "sin(x)/x",
  "variable": "x",
  "point": "0",
  "steps": true
}
```

### Example Request (Series - Plain Text)

```json
{
  "type": "series",
  "expression": "exp(x)",
  "variable": "x",
  "point": "0",
  "order": 5,
  "steps": true
}
```

### Example Response

```json
{
  "result": "x**3/3",
  "steps": [
    "Parsing expression and simplifying",
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

The project has a comprehensive working implementation with:
- Integration with step-by-step explanations
- Differentiation, equation solving, and expression simplification
- Limits and series expansion (plain text)
- FastAPI web interface with comprehensive validation
- Complete test coverage for all operations (API and smoke tests)
- Proper project structure with modular components
- Code quality tools (black, ruff) configured
- Safe expression parsing with whitelisted functions and constants
- Optional SymEngine acceleration for faster computations
- Support for plain-text mathematical expressions
- Detailed step-by-step explanations for all supported operations