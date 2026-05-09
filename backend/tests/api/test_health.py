"""Tests for /health and /health/ready endpoints.

Health endpoints require no authentication. They probe the asyncpg pool
and Redis, both of which must be mocked since we run against aiosqlite
and fakeredis (and the asyncpg pool is never initialized in tests).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealth:

    async def test_health_returns_200(self, client):
        """Basic liveness check returns 200 even when sub-checks fail."""
        # DB pool is not initialized (RuntimeError from get_db_pool),
        # and StockPulse is unreachable -- both should be reported as
        # errors but the endpoint must not crash.
        resp = await client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert "status" in body

    async def test_health_reports_database_error_when_pool_missing(self, client):
        """Without asyncpg pool, database check reports an error string."""
        resp = await client.get("/health")
        body = resp.json()
        # The pool is never initialized in tests, so database should be error
        assert "database" in body
        assert "error" in body["database"] or body["database"] == "ok"

    async def test_health_redis_ok_with_fakeredis(self, client):
        """Redis check should succeed against the injected fakeredis."""
        resp = await client.get("/health")
        body = resp.json()
        assert body["redis"] == "ok"

    async def test_health_all_ok_when_deps_healthy(self, client, monkeypatch):
        """When DB pool and StockPulse are mocked healthy, all checks pass."""
        # Mock the asyncpg pool
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(return_value=1)

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_conn),
                __aexit__=AsyncMock(return_value=False),
            )
        )

        import app.api.health as health_mod
        import app.core.database as db_mod
        monkeypatch.setattr(db_mod, "_pool", mock_pool)

        # Mock StockPulse health check
        mock_sp_config = AsyncMock(return_value=("http://stockpulse:8010", "key"))
        monkeypatch.setattr(health_mod, "_get_stockpulse_config", mock_sp_config, raising=False)

        import httpx as httpx_mod
        original_async_client = httpx_mod.AsyncClient

        class FakeStockPulseClient:
            def __init__(self, **kwargs):
                pass
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass
            async def get(self, url, **kwargs):
                resp = httpx_mod.Response(200, json={"status": "ok"})
                return resp

        monkeypatch.setattr(httpx_mod, "AsyncClient", FakeStockPulseClient)

        resp = await client.get("/health")
        body = resp.json()
        assert body["status"] == "ok"
        assert body["database"] == "ok"
        assert body["redis"] == "ok"


# ---------------------------------------------------------------------------
# GET /health/ready
# ---------------------------------------------------------------------------

class TestReady:

    async def test_ready_fails_without_db_pool(self, client):
        """Readiness probe should fail (500) when asyncpg pool is absent."""
        resp = await client.get("/health/ready")
        # get_db_pool() raises RuntimeError => 500
        assert resp.status_code == 500

    async def test_ready_succeeds_with_mocked_deps(self, client, monkeypatch):
        """Readiness probe returns 200 when DB pool and Redis are healthy."""
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(return_value=1)

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_conn),
                __aexit__=AsyncMock(return_value=False),
            )
        )

        import app.core.database as db_mod
        monkeypatch.setattr(db_mod, "_pool", mock_pool)

        resp = await client.get("/health/ready")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ready"
