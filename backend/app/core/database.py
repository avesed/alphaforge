"""Asyncpg connection pool for high-performance bulk operations.

Used for direct SQL queries (ML data pipelines, batch operations).
For ORM operations (auth, admin CRUD), use app.core.orm instead.
"""

from __future__ import annotations

import logging
from typing import Optional

import asyncpg

from app.config import get_settings

logger = logging.getLogger(__name__)

_pool: Optional[asyncpg.Pool] = None


def _build_dsn() -> str:
    """Convert DATABASE_URL to a plain postgresql:// DSN for asyncpg.

    SQLAlchemy-style URLs use ``postgresql+asyncpg://...`` and may carry
    query parameters (e.g. ``?sslmode=require``).  asyncpg expects a bare
    ``postgresql://`` scheme with no driver suffix.
    """
    db_url = get_settings().DATABASE_URL
    if not db_url:
        raise RuntimeError("DATABASE_URL is not configured")
    dsn = db_url.replace("+asyncpg", "").split("?")[0]
    return dsn


async def init_db_pool() -> None:
    """Create the asyncpg connection pool. Called once during startup."""
    global _pool
    if _pool is not None:
        logger.warning("Database pool already initialized, skipping")
        return

    settings = get_settings()
    dsn = _build_dsn()

    logger.info(
        "Initializing asyncpg pool (min=%d, max=%d, cmd_timeout=%ds)",
        settings.DATABASE_POOL_MIN_SIZE,
        settings.DATABASE_POOL_MAX_SIZE,
        settings.DATABASE_COMMAND_TIMEOUT,
    )

    _pool = await asyncpg.create_pool(
        dsn,
        min_size=settings.DATABASE_POOL_MIN_SIZE,
        max_size=settings.DATABASE_POOL_MAX_SIZE,
        command_timeout=settings.DATABASE_COMMAND_TIMEOUT,
    )

    logger.info("Asyncpg pool created successfully")


async def close_db_pool() -> None:
    """Close the asyncpg connection pool. Called during shutdown."""
    global _pool
    if _pool is not None:
        logger.info("Closing asyncpg pool...")
        await _pool.close()
        _pool = None
        logger.info("Asyncpg pool closed")


def get_db_pool() -> asyncpg.Pool:
    """Return the active connection pool.

    Raises:
        RuntimeError: If the pool has not been initialized via ``init_db_pool()``.
    """
    if _pool is None:
        raise RuntimeError(
            "Database pool is not initialized. "
            "Call init_db_pool() during application startup."
        )
    return _pool
