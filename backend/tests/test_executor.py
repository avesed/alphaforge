"""Tests for app.executor module.

Tests the ThreadPoolExecutor (run_qlib_quick) and ProcessPoolExecutor
(run_qlib_background) execution wrappers.

ProcessPoolExecutor tests are skipped because they can conflict with
aiosqlite's in-memory database and event loop in the test environment.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from app.executor import run_qlib_quick, run_qlib_background


# ---------------------------------------------------------------------------
# Helpers -- simple callables for executor tests
# ---------------------------------------------------------------------------

def _add(a: int, b: int) -> int:
    return a + b


def _slow_func(seconds: float) -> str:
    time.sleep(seconds)
    return "done"


def _raise_error() -> None:
    raise ValueError("intentional error for testing")


def _multiply(x: int, y: int = 2) -> int:
    return x * y


# ---------------------------------------------------------------------------
# run_qlib_quick (ThreadPoolExecutor)
# ---------------------------------------------------------------------------


class TestRunQlibQuick:

    async def test_simple_callable(self):
        """run_qlib_quick executes a simple sync function."""
        result = await run_qlib_quick(_add, 3, 7)
        assert result == 10

    async def test_with_kwargs(self):
        """run_qlib_quick passes kwargs correctly."""
        result = await run_qlib_quick(_multiply, 5, y=3)
        assert result == 15

    async def test_timeout_propagation(self):
        """run_qlib_quick raises TimeoutError (asyncio) on timeout."""
        with pytest.raises((TimeoutError, asyncio.TimeoutError)):
            await run_qlib_quick(_slow_func, 10.0, timeout=0.1)

    async def test_exception_propagation(self):
        """run_qlib_quick propagates exceptions from the callable."""
        with pytest.raises(ValueError, match="intentional error"):
            await run_qlib_quick(_raise_error)

    async def test_returns_correct_type(self):
        """run_qlib_quick preserves return type."""
        result = await run_qlib_quick(_add, 0, 0)
        assert isinstance(result, int)
        assert result == 0

    async def test_no_args_callable(self):
        """run_qlib_quick works with no-arg callables."""
        def _get_value() -> str:
            return "hello"

        result = await run_qlib_quick(_get_value)
        assert result == "hello"


# ---------------------------------------------------------------------------
# run_qlib_background (ProcessPoolExecutor) -- limited testing
# ---------------------------------------------------------------------------


class TestRunQlibBackground:

    async def test_simple_callable(self):
        """run_qlib_background executes a simple picklable function."""
        result = await run_qlib_background(_add, 10, 20)
        assert result == 30

    async def test_exception_propagation(self):
        """run_qlib_background propagates exceptions from the callable."""
        with pytest.raises(ValueError, match="intentional error"):
            await run_qlib_background(_raise_error)
