"""Tests for /api/v1/expression endpoints.

Validates expression evaluation, batch, and validation endpoints
with X-API-Key and JWT Bearer auth.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest


# ---------------------------------------------------------------------------
# POST /api/v1/expression/validate
# ---------------------------------------------------------------------------


class TestValidateExpression:

    async def test_valid_expression(self, apikey_client, api_consumer):
        """Valid Qlib expression returns valid=true with operators."""
        resp = await apikey_client.post(
            "/api/v1/expression/validate",
            json={"expression": "Corr($close, $volume, 20)"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["valid"] is True
        assert body["error"] is None
        assert "Corr" in body["operators_used"]

    async def test_invalid_expression_unknown_operator(self, apikey_client, api_consumer):
        """Expression with unknown operator returns valid=false."""
        resp = await apikey_client.post(
            "/api/v1/expression/validate",
            json={"expression": "BadOp($close, 10)"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["valid"] is False
        assert "Unknown operators" in body["error"]

    async def test_invalid_expression_empty(self, apikey_client, api_consumer):
        """Empty expression returns valid=false."""
        resp = await apikey_client.post(
            "/api/v1/expression/validate",
            json={"expression": "   "},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["valid"] is False
        assert "Empty" in body["error"]

    async def test_validate_requires_auth(self, client):
        """Validate endpoint without auth returns 401."""
        resp = await client.post(
            "/api/v1/expression/validate",
            json={"expression": "Mean($close, 5)"},
        )
        assert resp.status_code == 401

    async def test_validate_with_jwt_bearer(self, auth_client, admin_user):
        """Validate endpoint accepts admin JWT Bearer token."""
        resp = await auth_client.post(
            "/api/v1/expression/validate",
            json={"expression": "Mean($close, 5)"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["valid"] is True


# ---------------------------------------------------------------------------
# POST /api/v1/expression/evaluate
# ---------------------------------------------------------------------------


class TestEvaluateExpression:

    async def test_evaluate_success(self, apikey_client, api_consumer):
        """Evaluate returns canned result when executor is mocked."""
        canned = {
            "symbol": "AAPL",
            "expression": "Mean($close, 5)",
            "series": [{"date": "2024-01-01", "value": 150.0}],
            "latest_value": 150.0,
            "count": 1,
        }
        with patch(
            "app.api.public.expression.run_qlib_quick",
            new_callable=AsyncMock,
            return_value=canned,
        ):
            resp = await apikey_client.post(
                "/api/v1/expression/evaluate",
                json={
                    "symbol": "AAPL",
                    "expression": "Mean($close, 5)",
                    "market": "us",
                },
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["symbol"] == "AAPL"
        assert body["latest_value"] == 150.0
        assert len(body["series"]) == 1

    async def test_evaluate_timeout(self, apikey_client, api_consumer):
        """Evaluate returns 504 on TimeoutError."""
        with patch(
            "app.api.public.expression.run_qlib_quick",
            new_callable=AsyncMock,
            side_effect=TimeoutError("timed out"),
        ):
            resp = await apikey_client.post(
                "/api/v1/expression/evaluate",
                json={
                    "symbol": "AAPL",
                    "expression": "Mean($close, 5)",
                    "market": "us",
                },
            )
        assert resp.status_code == 504

    async def test_evaluate_requires_auth(self, client):
        """Evaluate without auth returns 401."""
        resp = await client.post(
            "/api/v1/expression/evaluate",
            json={
                "symbol": "AAPL",
                "expression": "Mean($close, 5)",
                "market": "us",
            },
        )
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# POST /api/v1/expression/batch
# ---------------------------------------------------------------------------


class TestBatchExpression:

    async def test_batch_success(self, apikey_client, api_consumer):
        """Batch evaluate returns results for multiple symbols."""
        canned = {
            "expression": "Mean($close, 5)",
            "results": {"AAPL": 150.0, "MSFT": 380.0},
            "date": "2024-01-01",
        }
        with patch(
            "app.api.public.expression.run_qlib_quick",
            new_callable=AsyncMock,
            return_value=canned,
        ):
            resp = await apikey_client.post(
                "/api/v1/expression/batch",
                json={
                    "symbols": ["AAPL", "MSFT"],
                    "expression": "Mean($close, 5)",
                    "market": "us",
                },
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["results"]["AAPL"] == 150.0
        assert body["results"]["MSFT"] == 380.0

    async def test_batch_timeout(self, apikey_client, api_consumer):
        """Batch evaluate returns 504 on TimeoutError."""
        with patch(
            "app.api.public.expression.run_qlib_quick",
            new_callable=AsyncMock,
            side_effect=TimeoutError("timed out"),
        ):
            resp = await apikey_client.post(
                "/api/v1/expression/batch",
                json={
                    "symbols": ["AAPL"],
                    "expression": "Mean($close, 5)",
                    "market": "us",
                },
            )
        assert resp.status_code == 504
