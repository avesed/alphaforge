"""Dual execution paths for Qlib operations.

Qlib is NOT thread-safe (GitHub #842). All Qlib calls must be serialized.
We use two separate executors:

- ThreadPoolExecutor(max_workers=1): For quick queries (expression evaluation,
  factor computation). These complete in <15 seconds and run in the same process
  as the FastAPI server, sharing the Qlib global state.

- ProcessPoolExecutor(max_workers=1): For long-running tasks (backtests up to
  30 minutes, full market data syncs). These run in a separate subprocess with
  their own Qlib init, preventing them from blocking quick queries.

Both executors have max_workers=1 to ensure serialization.
"""
import asyncio
import logging
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Default timeouts (seconds)
QUICK_TIMEOUT = 60  # Expression/factor queries
BACKGROUND_TIMEOUT = 7200  # Backtests/data sync (2 hours, full US rebuild ~80 min)

# Quick path: expression/factor queries, serialized in one thread (<15s)
_thread_executor = ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="qlib-quick"
)
assert _thread_executor._max_workers == 1, "Qlib thread executor must be single-threaded"

# Background path: backtests/data sync, independent subprocess (up to 30min)
_process_executor = ProcessPoolExecutor(max_workers=1)
assert _process_executor._max_workers == 1, "Qlib process executor must be single-worker"


async def run_qlib_quick(
    func: Callable[..., T],
    *args: Any,
    timeout: float = QUICK_TIMEOUT,
    **kwargs: Any,
) -> T:
    """Execute a quick Qlib operation in the dedicated thread.

    Use for: expression evaluation, factor computation, factor analysis.
    Expected duration: <15 seconds. Hard timeout: 60s by default.
    """
    func_name = getattr(func, "__name__", str(func))
    logger.info("qlib-quick: submitting %s", func_name)
    start = time.monotonic()

    loop = asyncio.get_running_loop()
    if kwargs:
        call = partial(func, *args, **kwargs)
    else:
        call = partial(func, *args) if args else func

    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(_thread_executor, call),
            timeout=timeout,
        )
        elapsed = time.monotonic() - start
        logger.info("qlib-quick: %s completed in %.2fs", func_name, elapsed)
        return result
    except asyncio.TimeoutError:
        elapsed = time.monotonic() - start
        logger.error(
            "qlib-quick: %s timed out after %.2fs (limit: %ds)",
            func_name, elapsed, timeout,
        )
        raise
    except Exception:
        elapsed = time.monotonic() - start
        logger.error(
            "qlib-quick: %s failed after %.2fs", func_name, elapsed, exc_info=True,
        )
        raise


async def run_qlib_background(
    func: Callable[..., T],
    *args: Any,
    timeout: float = BACKGROUND_TIMEOUT,
    **kwargs: Any,
) -> T:
    """Execute a long-running Qlib operation in a separate process.

    Use for: backtests, full market data syncs.
    Expected duration: up to 30 minutes.

    Note: The subprocess has its own Qlib init, independent of the main process.
    """
    func_name = getattr(func, "__name__", str(func))
    logger.info("qlib-background: submitting %s", func_name)
    start = time.monotonic()

    loop = asyncio.get_running_loop()
    if kwargs:
        call = partial(func, *args, **kwargs)
    else:
        call = partial(func, *args) if args else func

    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(_process_executor, call),
            timeout=timeout,
        )
        elapsed = time.monotonic() - start
        logger.info("qlib-background: %s completed in %.2fs", func_name, elapsed)
        return result
    except asyncio.TimeoutError:
        elapsed = time.monotonic() - start
        logger.error(
            "qlib-background: %s timed out after %.2fs (limit: %ds)",
            func_name, elapsed, timeout,
        )
        raise
    except Exception:
        elapsed = time.monotonic() - start
        logger.error(
            "qlib-background: %s failed after %.2fs", func_name, elapsed, exc_info=True,
        )
        raise


def shutdown_executors() -> None:
    """Gracefully shut down both executors. Call on app shutdown."""
    logger.info("Shutting down Qlib executors...")
    _thread_executor.shutdown(wait=True, cancel_futures=True)
    _process_executor.shutdown(wait=True, cancel_futures=True)
    logger.info("Qlib executors shut down")
