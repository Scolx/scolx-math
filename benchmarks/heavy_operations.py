from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from scolx_math.advanced_latex import (
    integrate_latex_with_steps,
    solve_equation_latex_with_steps,
)
from scolx_math.core import operations as operations_module
from scolx_math.core.operations import (
    hessian_expr,
    matrix_inverse,
    series_expr,
    solve_system_numeric,
)

LATEX_INTEGRAL_EXPR = r"\frac{\exp\left(x^{2}\right) x^{5}}{1 + x^{2}}"
LATEX_SOLVE_EXPR = r"x^{6} - 3 x^{3} + 1"
SERIES_EXPRESSION = "exp(sin(x)) + log(1 + x)"
HESSIAN_EXPRESSION = "x**4 + y**4 + z**4 + w**4 + x*y*z*w + sin(x*y) * cos(z*w)"
MATRIX_6X6 = [[i * 6 + j + 1 for j in range(6)] for i in range(6)]
NUMERIC_EQUATIONS = [
    "sin(x) + y**2 - 1.5",
    "x**2 + y**2 - 1",
]
NUMERIC_VARIABLES = ["x", "y"]
NUMERIC_INITIAL_GUESS = [0.5, 0.5]


class SkipBenchmark(RuntimeError):
    """Raised when a benchmark case should be skipped."""


@dataclass
class BenchmarkCase:
    name: str
    func: Callable[[], object]
    warmup_runs: int = 1
    iterations: int = 5

    def overridden(
        self,
        *,
        warmup_runs: int | None = None,
        iterations: int | None = None,
    ) -> BenchmarkCase:
        return BenchmarkCase(
            name=self.name,
            func=self.func,
            warmup_runs=self.warmup_runs if warmup_runs is None else warmup_runs,
            iterations=self.iterations if iterations is None else iterations,
        )


@dataclass
class BenchmarkResult:
    name: str
    status: str
    durations: list[float]
    message: str | None = None


def _benchmark_latex_integral() -> None:
    integrate_latex_with_steps(LATEX_INTEGRAL_EXPR, "x")


def _benchmark_latex_solve() -> None:
    solve_equation_latex_with_steps(LATEX_SOLVE_EXPR, "x")


def _benchmark_series() -> None:
    series_expr(SERIES_EXPRESSION, "x", "0", 20)


def _benchmark_matrix_inverse() -> None:
    matrix_inverse(MATRIX_6X6)


def _benchmark_hessian() -> None:
    hessian_expr(HESSIAN_EXPRESSION, ["x", "y", "z", "w"])


def _benchmark_numeric_solver() -> None:
    if operations_module.optimize is None:
        raise SkipBenchmark(
            "SciPy is not available; skipping numeric solver benchmark.",
        )
    solve_system_numeric(
        NUMERIC_EQUATIONS,
        NUMERIC_VARIABLES,
        NUMERIC_INITIAL_GUESS,
        max_iterations=200,
        tolerance=1e-12,
    )


CASES: list[BenchmarkCase] = [
    BenchmarkCase(
        "latex_integral_complex",
        _benchmark_latex_integral,
        warmup_runs=1,
        iterations=3,
    ),
    BenchmarkCase(
        "latex_solve_polynomial",
        _benchmark_latex_solve,
        warmup_runs=1,
        iterations=3,
    ),
    BenchmarkCase("series_high_order", _benchmark_series, warmup_runs=1, iterations=3),
    BenchmarkCase(
        "matrix_inverse_6x6",
        _benchmark_matrix_inverse,
        warmup_runs=1,
        iterations=5,
    ),
    BenchmarkCase(
        "hessian_mixed_terms",
        _benchmark_hessian,
        warmup_runs=1,
        iterations=3,
    ),
    BenchmarkCase(
        "numeric_solver_non_linear",
        _benchmark_numeric_solver,
        warmup_runs=1,
        iterations=3,
    ),
]


def _run_case(case: BenchmarkCase) -> BenchmarkResult:
    try:
        for _ in range(case.warmup_runs):
            case.func()
    except SkipBenchmark as exc:
        return BenchmarkResult(case.name, "skipped", [], str(exc))
    except Exception as exc:  # pragma: no cover - defensive
        return BenchmarkResult(case.name, "error", [], f"Warmup failed: {exc}")

    durations: list[float] = []
    for _ in range(case.iterations):
        try:
            start = time.perf_counter()
            case.func()
            end = time.perf_counter()
        except SkipBenchmark as exc:
            return BenchmarkResult(case.name, "skipped", durations, str(exc))
        except Exception as exc:  # pragma: no cover - defensive
            return BenchmarkResult(case.name, "error", durations, str(exc))
        durations.append(end - start)

    return BenchmarkResult(case.name, "ok", durations)


def _format_duration_stats(durations: Iterable[float]) -> tuple[str, str, str, str]:
    ms = [value * 1000 for value in durations]
    average = statistics.mean(ms)
    best = min(ms)
    worst = max(ms)
    stdev = statistics.stdev(ms) if len(ms) > 1 else 0.0
    return (
        f"{average:10.2f}",
        f"{best:10.2f}",
        f"{worst:10.2f}",
        f"{stdev:10.2f}",
    )


def _print_summary(results: list[BenchmarkResult]) -> None:
    header = (
        f"{'Benchmark':<32} {'Avg (ms)':>12} {'Best (ms)':>12} "
        f"{'Worst (ms)':>12} {'Std Dev':>12} Status"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        if result.status == "ok":
            avg, best, worst, stdev = _format_duration_stats(result.durations)
            print(f"{result.name:<32} {avg} {best} {worst} {stdev} OK")
        elif result.status == "skipped":
            reason = result.message or ""
            print(
                f"{result.name:<32} {'-':>12} {'-':>12} {'-':>12} {'-':>12} Skipped - {reason}",
            )
        else:
            message = result.message or "Unknown error"
            print(
                f"{result.name:<32} {'-':>12} {'-':>12} {'-':>12} {'-':>12} Error - {message}",
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark heavy Scolx Math operations for performance insights.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        help="Override the number of iterations for each benchmark case.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        help="Override the number of warmup runs before timing.",
    )
    parser.add_argument(
        "--case",
        action="append",
        help="Name of a benchmark case to run (can be provided multiple times).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    selected_cases = CASES

    if args.case:
        requested = set(args.case)
        known = {case.name for case in CASES}
        unknown = requested - known
        if unknown:
            print(
                f"Unknown benchmark case(s): {', '.join(sorted(unknown))}",
                file=sys.stderr,
            )
            return 1
        selected_cases = [case for case in CASES if case.name in requested]

    overrides_applied = [
        case.overridden(warmup_runs=args.warmup, iterations=args.iterations)
        for case in selected_cases
    ]

    results = [_run_case(case) for case in overrides_applied]

    print("\nScolx Math heavy-operations benchmark\n")
    _print_summary(results)

    if any(result.status == "error" for result in results):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
