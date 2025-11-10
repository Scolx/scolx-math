"""Utility functions for threadpool management and performance optimization."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeVar

from fastapi.concurrency import run_in_threadpool

T = TypeVar("T")

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
            max_workers=max_workers, thread_name_prefix="math_op"
        )
    return _threadpool_executor


def run_cpu_bound(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Execute a CPU-bound function in a threadpool."""
    if kwargs:
        # If kwargs are provided, wrap in a lambda to avoid issues with run_in_threadpool
        def wrapper():
            return func(*args, **kwargs)

        return asyncio.run(run_in_threadpool(wrapper))
    else:
        return asyncio.run(run_in_threadpool(func, *args))


async def run_cpu_bound_async(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Execute a CPU-bound function in a threadpool asynchronously."""
    if kwargs:
        # If kwargs are provided, wrap in a lambda to avoid issues with run_in_threadpool
        def wrapper():
            return func(*args, **kwargs)

        return await run_in_threadpool(wrapper)
    else:
        return await run_in_threadpool(func, *args)


def cleanup_threadpool():
    """Clean up the threadpool executor."""
    global _threadpool_executor
    if _threadpool_executor is not None:
        _threadpool_executor.shutdown(wait=True)
        _threadpool_executor = None
