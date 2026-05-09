"""Tests for app.core.redis -- cache helpers and jittered_ttl.

~10 tests covering:
- cache_get / cache_set / cache_delete with fakeredis
- JSON round-trip (dict -> set -> get -> dict)
- Cache miss returns None
- TTL expiry
- jittered_ttl returns value within expected range
"""

import asyncio
from unittest.mock import patch

import pytest

from app.core.redis import cache_delete, cache_get, cache_set, jittered_ttl


# ---------------------------------------------------------------------------
# cache_set / cache_get / cache_delete
# ---------------------------------------------------------------------------


class TestCacheSetGet:
    async def test_set_and_get_string(self, test_settings, fake_redis):
        with patch("app.core.redis.get_redis", return_value=fake_redis):
            await cache_set("af:test:str", "hello", ttl=60)
            result = await cache_get("af:test:str")
        assert result == "hello"

    async def test_json_round_trip_dict(self, test_settings, fake_redis):
        data = {"name": "test", "count": 42, "nested": {"a": 1}}
        with patch("app.core.redis.get_redis", return_value=fake_redis):
            await cache_set("af:test:dict", data, ttl=60)
            result = await cache_get("af:test:dict")
        assert result == data

    async def test_json_round_trip_list(self, test_settings, fake_redis):
        data = [1, 2, 3, "four"]
        with patch("app.core.redis.get_redis", return_value=fake_redis):
            await cache_set("af:test:list", data, ttl=60)
            result = await cache_get("af:test:list")
        assert result == data

    async def test_cache_miss_returns_none(self, test_settings, fake_redis):
        with patch("app.core.redis.get_redis", return_value=fake_redis):
            result = await cache_get("af:test:nonexistent")
        assert result is None

    async def test_set_overwrites_previous_value(self, test_settings, fake_redis):
        with patch("app.core.redis.get_redis", return_value=fake_redis):
            await cache_set("af:test:overwrite", "old", ttl=60)
            await cache_set("af:test:overwrite", "new", ttl=60)
            result = await cache_get("af:test:overwrite")
        assert result == "new"


class TestCacheDelete:
    async def test_delete_existing_key(self, test_settings, fake_redis):
        with patch("app.core.redis.get_redis", return_value=fake_redis):
            await cache_set("af:test:del", "value", ttl=60)
            await cache_delete("af:test:del")
            result = await cache_get("af:test:del")
        assert result is None

    async def test_delete_nonexistent_key_is_noop(self, test_settings, fake_redis):
        """Deleting a key that does not exist should not raise."""
        with patch("app.core.redis.get_redis", return_value=fake_redis):
            await cache_delete("af:test:never-existed")
            # No exception raised


class TestCacheTTLExpiry:
    async def test_key_expires_after_ttl(self, test_settings, fake_redis):
        """Set with short TTL, verify expiry via fakeredis."""
        with patch("app.core.redis.get_redis", return_value=fake_redis):
            await cache_set("af:test:ttl", "ephemeral", ttl=1)
            assert await cache_get("af:test:ttl") == "ephemeral"

        await asyncio.sleep(1.1)

        with patch("app.core.redis.get_redis", return_value=fake_redis):
            result = await cache_get("af:test:ttl")
        assert result is None


# ---------------------------------------------------------------------------
# jittered_ttl
# ---------------------------------------------------------------------------


class TestJitteredTtl:
    def test_result_within_range(self):
        base, jitter = 300, 60
        result = jittered_ttl(base, jitter)
        assert base <= result <= base + jitter

    def test_default_jitter_is_60(self):
        result = jittered_ttl(100)
        assert 100 <= result <= 160

    def test_zero_jitter_returns_base(self):
        assert jittered_ttl(500, jitter=0) == 500
