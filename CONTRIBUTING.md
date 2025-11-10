# Contributing to Scolx Math API

Thanks for your interest in contributing! This document outlines expectations for contributors and provides practical guidance for developing and maintaining the project.

## Quick Start Checklist

1. Fork the repository and clone your fork.
2. Create and activate a Python 3.11 virtual environment.
3. Install dependencies:
   - Runtime: `pip install -r requirements.txt`
   - Tooling & tests: `pip install -r requirements-dev.txt`
4. Run the test suite to ensure the environment is healthy: `pytest`
5. Create a feature branch (`feature/<short-description>`).
6. Make your changes, run formatters and tests, then open a pull request.

## Standards & Code Quality

- **Formatting**: Run `black .` before committing.
- **Linting**: Run `ruff check .` and address warnings. Use `ruff check --fix` when appropriate.
- **Imports**: Keep imports sorted and grouped logically. Ruff handles this automatically.
- **Typing**: Use type hints on all public functions and wherever it aids readability.
- **Docstrings**: Add docstrings for public modules, classes, and functions. Follow Google-style or NumPy-style consistently.
- **Testing**: Add or update tests for any new behavior or bug fix. Prefer pytest parametrization for input variations.
- **Security**: Do not weaken expression parsing safeguards. If your change requires new SymPy capabilities, extend the safe whitelist explicitly and document the rationale.

## Pull Request Guidelines

- Keep PRs focused. Separate large features into logical chunks.
- Reference related issues in the PR description (`Fixes #123`).
- Provide a concise summary of changes and testing performed.
- Ensure CI (once configured) passes prior to requesting review.
- Mark TODO items or follow-up tasks in issues instead of leaving inline comments.

## API & Feature Guidelines

- **Request Validation**: Ensure new endpoints or payloads use strict Pydantic validation. Return informative HTTP errors instead of generic responses.
- **SymPy Usage**: Prefer using the existing parsing helpers. If a new operation requires additional functions/constants, extend the whitelist in `scolx_math/core/parsing.py`.
- **Performance**: Offload any CPU-intensive SymPy work with `fastapi.concurrency.run_in_threadpool` to keep the event loop responsive.
- **Extensibility**: When adding operations, document them in README and docs, add tests, and update the TODO roadmap if necessary.

## Branching & Commits

- Branch naming: `feature/...`, `bugfix/...`, `docs/...`, `chore/...`.
- Commit messages: concise imperative form (e.g., `Add limit validation tests`).
- Squash commits when it simplifies history, but retain meaningful milestones if needed for review.

## Developer Workflow Tips

- Use `pytest -k <name>` for targeted test runs.
- Run `pytest --maxfail=1` for faster failure feedback.
- Keep an eye on dependency updates and ensure compatibility with Python 3.11+.
- Document any environment variables or configuration toggles in README or `/docs`.

## Testing Structure

The project includes multiple test files in the `tests/` directory:
- `test_api.py`: Tests for basic API endpoints and operations
- `test_latex.py`: Tests for LaTeX parsing functionality
- `test_smoke.py`: Basic smoke tests

When adding new functionality, add appropriate tests to the relevant test file or create a new one if needed.

## Reporting Issues

- Include environment details (OS, Python version, dependency versions).
- Provide minimal reproducible examples for computation bugs.
- Tag the issue appropriately (bug, enhancement, docs, performance).

Thank you for helping make Scolx Math API better! If you have questions, open an issue or start a discussion in the repository.
