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



### Gradient Example
```bash
curl -X POST "http://127.0.0.1:8000/solve" \
  -H "Content-Type: application/json" \
  -d '{
    "type": "gradient",
    "expression": "x**2 + y**2",
    "variables": ["x", "y"],
    "steps": true
  }'
```

### Matrix Multiplication Example
```bash
curl -X POST "http://127.0.0.1:8000/solve" \
  -H "Content-Type: application/json" \
  -d '{
    "type": "matrix_multiply",
    "left_matrix": [[1, 2], [3, 4]],
    "right_matrix": [[5, 6], [7, 8]]
  }'
```

### ODE Solving Example
```bash
curl -X POST "http://127.0.0.1:8000/solve" \
  -H "Content-Type: application/json" \
  -d '{
    "type": "ode",
    "expression": "Eq(diff(y(x), x), y(x))",
    "variable": "x",
    "function": "y",
    "initial_conditions": {"y(0)": "1"}
  }'
```

## 5. Python Usage

```python

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

  "variables": ["string"],    // Variables for multivariate operations
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

### Supported Operation Types

- `integral`: Calculate integral
- `derivative`: Calculate derivative
- `solve`: Solve equation
- `simplify`: Simplify expression
- `limit`: Calculate limit (fully implemented)
- `series`: Series expansion (fully implemented)
- `gradient`: Calculate gradient of multivariate expression
- `hessian`: Calculate Hessian matrix of multivariate expression
- `matrix_determinant`: Compute determinant of a matrix
- `matrix_inverse`: Compute inverse of a matrix
- `matrix_multiply`: Multiply two matrices
- `ode`: Solve ordinary differential equation (analytical or numerical)
- `plot`: Generate points for plotting
- `complex_conjugate`: Compute complex conjugate
- `complex_modulus`: Compute complex modulus
- `complex_argument`: Compute complex argument
- `complex_to_polar`: Convert complex to polar form
- `complex_from_polar`: Convert polar to complex form
- `stats_mean`: Compute statistical mean
- `stats_variance`: Compute statistical variance
- `stats_stddev`: Compute statistical standard deviation
- `normal_pdf`: Compute normal probability density function
- `normal_cdf`: Compute normal cumulative distribution function
- `solve_numeric`: Solve system of equations numerically


## 9. Project Structure

- `scolx_math/api/main.py`: Main FastAPI application
- `scolx_math/core/operations.py`: Basic mathematical operations
- `scolx_math/core/parsing.py`: Safe parsing helpers
- `scolx_math/explain/explainers.py`: Step-by-step explanations
- `tests/test_api.py`: API endpoint tests
- `tests/test_smoke.py`: Basic integration test
- `docs/documentation.md`: Full documentation
- `examples.py`: Usage examples with plain text