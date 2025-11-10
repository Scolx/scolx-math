"""Utility functions for threadpool management and performance optimization."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import ParamSpec, TypeVar

from fastapi.concurrency import run_in_threadpool

T = TypeVar("T")
P = ParamSpec("P")

# Global threadpool executor for CPU-bound operations
# Using a smaller number of workers to avoid overhead for mathematical operations
_threadpool_executor: ThreadPoolExecutor | None = None


def get_threadpool_executor() -> ThreadPoolExecutor:
    """Get or create a threadpool executor for CPU-bound operations."""
    global _threadpool_executor
    if _threadpool_executor is None:
        # Use number of CPU cores for optimal performance
        import os

        max_workers = min(32, (os.cpu_count() or 1) + 4)
        _threadpool_executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="math_op",
        )
    return _threadpool_executor


def run_cpu_bound(func: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    """Execute a CPU-bound function in a threadpool."""
    # For synchronous execution, directly use the threadpool executor
    executor = get_threadpool_executor()
    if kwargs:

        def wrapper() -> T:
            return func(*args, **kwargs)

        future = executor.submit(wrapper)
    else:
        future = executor.submit(func, *args)
    return future.result()


async def run_cpu_bound_async(
    func: Callable[P, T],
    *args: P.args,
    **kwargs: P.kwargs,
) -> T:
    """Execute a CPU-bound function in a threadpool asynchronously."""
    if kwargs:
        # If kwargs are provided, wrap in a lambda to avoid issues with run_in_threadpool
        def wrapper() -> T:
            return func(*args, **kwargs)

        return await run_in_threadpool(wrapper)
    return await run_in_threadpool(func, *args)


def cleanup_threadpool() -> None:
    """Clean up the threadpool executor."""
    global _threadpool_executor
    if _threadpool_executor is not None:
        _threadpool_executor.shutdown(wait=True)
        _threadpool_executor = None
