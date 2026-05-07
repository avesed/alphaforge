"""Redis cache wrapper for AlphaForge.

Uses Redis DB 0 (own instance). All operations are best-effort:
cache misses or errors never block the request.
"""

from __future__ import annotations

import json
import logging
import random
from typing import Any, Optional

import redis.asyncio as aioredis

from app.config import get_settings

logger = logging.getLogger(__name__)

_redis_client: Optional[aioredis.Redis] = None


async def get_redis() -> aioredis.Redis:
    """Get or create the shared Redis client."""
    global _redis_client
    if _redis_client is None:
        settings = get_settings()
        _redis_client = aioredis.from_url(
            settings.REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True,
        )
    return _redis_client


async def close_redis() -> None:
    """Close the Redis connection. Call on application shutdown."""
    global _redis_client
    if _redis_client is not None:
        await _redis_client.close()
        _redis_client = None
        logger.info("Redis connection closed")


async def cache_get(key: str) -> Optional[Any]:
    """Get value from cache, returns None on miss or error."""
    try:
        r = await get_redis()
        data = await r.get(key)
        if data is not None:
            return json.loads(data)
        logger.debug("Cache miss: %s", key)
    except json.JSONDecodeError:
        logger.warning("Corrupted cache entry, deleting: %s", key)
        await cache_delete(key)
    except Exception as e:
        logger.warning("Cache read error for %s: %s", key, e)
    return None


async def cache_set(key: str, value: Any, ttl: int = 300) -> None:
    """Set value in cache with TTL in seconds. Best-effort, never raises."""
    try:
        r = await get_redis()
        await r.setex(key, ttl, json.dumps(value, default=str))
    except Exception as e:
        logger.warning("Cache write error for %s: %s", key, e)


async def cache_delete(key: str) -> None:
    """Delete a cache key. Best-effort, never raises."""
    try:
        r = await get_redis()
        await r.delete(key)
    except Exception as e:
        logger.warning("Cache delete error for %s: %s", key, e)


def jittered_ttl(base: int, jitter: int = 60) -> int:
    """Return base TTL + random jitter to prevent cache stampede."""
    return base + random.randint(0, jitter)
