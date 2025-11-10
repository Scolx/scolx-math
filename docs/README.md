# Scolx Math – Documentation

## Overview
Scolx Math is a Python project providing a FastAPI service for symbolic math operations with step-by-step explanations. The project now supports all major mathematical operations including integration, differentiation, equation solving, simplification, limits, and series expansion for both plain text and LaTeX expressions.

## Project Layout
- scolx_math/core: core symbolic operations (SymPy)
- scolx_math/explain: step-capturing wrappers
- scolx_math/api: FastAPI app
- scolx_math/advanced_latex: LaTeX parsing and advanced operations
- tests: pytest-based unit tests

## Quickstart
1) Create venv (Windows):
```
python -m venv venv
venv\Scripts\Activate.ps1
```
2) Install dependencies:
```
pip install -r requirements.txt
```
3) Run API:
```
uvicorn scolx_math.api.main:app --reload
```
Open http://127.0.0.1:8000/docs

## API
- POST /solve
  - body: { "type": "integral", "expression": "x**2", "variable": "x", "steps": true }
  - response: { "result": "x**3/3", "steps": [ ... ] }
  - Supports: integral, derivative, solve, simplify, limit, series, and their LaTeX equivalents

## Testing
```
pytest -q
```
