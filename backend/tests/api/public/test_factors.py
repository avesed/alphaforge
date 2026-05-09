"""Tests for /api/v1/factors endpoints.

Validates factor computation and summary endpoints with auth
and mocked Qlib executor.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest


# ---------------------------------------------------------------------------
# GET /api/v1/factors/{symbol}
# ---------------------------------------------------------------------------


class TestGetFactors:

    async def test_get_factors_success(self, apikey_client, api_consumer):
        """Factor computation returns canned result when executor is mocked."""
        canned = {
            "symbol": "AAPL",
            "market": "us",
            "alpha_type": "alpha158",
            "mode": "single",
            "factor_count": 3,
            "dates": ["2024-01-01", "2024-01-02"],
            "factors": {
                "ma5": [150.0, 151.0],
                "std20": [2.5, 2.6],
                "rsi14": [0.55, 0.60],
            },
            "top_factors": [
                {"name": "ma5", "z_score": 1.2},
            ],
        }
        with patch(
            "app.api.public.factors.run_qlib_quick",
            new_callable=AsyncMock,
            return_value=canned,
        ):
            resp = await apikey_client.get(
                "/api/v1/factors/AAPL",
                params={"market": "us"},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["symbol"] == "AAPL"
        assert body["factor_count"] == 3
        assert "ma5" in body["factors"]

    async def test_get_factors_timeout(self, apikey_client, api_consumer):
        """Factor computation returns 504 on TimeoutError."""
        with patch(
            "app.api.public.factors.run_qlib_quick",
            new_callable=AsyncMock,
            side_effect=TimeoutError("timed out"),
        ):
            resp = await apikey_client.get(
                "/api/v1/factors/AAPL",
                params={"market": "us"},
            )
        assert resp.status_code == 504

    async def test_get_factors_requires_auth(self, client):
        """Factor endpoint without auth returns 401."""
        resp = await client.get("/api/v1/factors/AAPL")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# GET /api/v1/factors/{symbol}/summary
# ---------------------------------------------------------------------------


class TestGetFactorSummary:

    async def test_get_summary_success(self, apikey_client, api_consumer):
        """Factor summary returns top-10 compact result."""
        canned = {
            "symbol": "AAPL",
            "market": "us",
            "latest_date": "2024-01-02",
            "top_factors": [
                {"name": "ma5", "value": 150.0, "z_score": 1.2},
                {"name": "rsi14", "value": 0.6, "z_score": 0.8},
            ],
            "mode": "single",
        }
        with patch(
            "app.api.public.factors.run_qlib_quick",
            new_callable=AsyncMock,
            return_value=canned,
        ):
            resp = await apikey_client.get(
                "/api/v1/factors/AAPL/summary",
                params={"market": "us"},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["symbol"] == "AAPL"
        assert len(body["top_factors"]) == 2

    async def test_get_summary_timeout(self, apikey_client, api_consumer):
        """Factor summary returns 504 on TimeoutError."""
        with patch(
            "app.api.public.factors.run_qlib_quick",
            new_callable=AsyncMock,
            side_effect=TimeoutError("timed out"),
        ):
            resp = await apikey_client.get(
                "/api/v1/factors/AAPL/summary",
                params={"market": "us"},
            )
        assert resp.status_code == 504

    async def test_get_summary_with_jwt(self, auth_client, admin_user):
        """Factor summary accepts admin JWT Bearer auth."""
        canned = {
            "symbol": "TSLA",
            "market": "us",
            "latest_date": "2024-01-02",
            "top_factors": [],
            "mode": "single",
        }
        with patch(
            "app.api.public.factors.run_qlib_quick",
            new_callable=AsyncMock,
            return_value=canned,
        ):
            resp = await auth_client.get(
                "/api/v1/factors/TSLA/summary",
                params={"market": "us"},
            )
        assert resp.status_code == 200
        assert resp.json()["symbol"] == "TSLA"
