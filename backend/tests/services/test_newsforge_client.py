"""Tests for NewsForgeClient.

Uses respx to mock httpx requests without hitting real NewsForge.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from app.services.newsforge_client import NewsForgeClient


@pytest.fixture
async def newsforge_client():
    """Create a NewsForgeClient with pre-configured base URL and key."""
    client = NewsForgeClient()
    client._base_url = "http://test-newsforge:8080"
    client._api_key = "test-nf-key"
    client._client = httpx.AsyncClient(
        base_url="http://test-newsforge:8080",
        headers={"X-API-Key": "test-nf-key"},
        timeout=10.0,
    )
    yield client
    await client._client.aclose()


# ---------------------------------------------------------------------------
# get_sentiment_batch
# ---------------------------------------------------------------------------


class TestGetSentimentBatch:

    @respx.mock
    async def test_success_returns_data(self, newsforge_client):
        """Successful request returns list of sentiment dicts."""
        payload = {
            "data": [
                {
                    "symbol": "AAPL",
                    "date": "2024-01-01",
                    "sentiment_avg": 0.65,
                    "article_count": 12,
                    "bullish_ratio": 0.7,
                    "content_score_avg": 0.8,
                },
                {
                    "symbol": "MSFT",
                    "date": "2024-01-01",
                    "sentiment_avg": 0.55,
                    "article_count": 8,
                    "bullish_ratio": 0.6,
                    "content_score_avg": 0.75,
                },
            ]
        }
        respx.post("http://test-newsforge:8080/api/internal/sentiment/ml-batch").mock(
            return_value=httpx.Response(200, json=payload)
        )

        result = await newsforge_client.get_sentiment_batch(
            symbols=["AAPL", "MSFT"],
            market="us",
            start_date="2024-01-01",
            end_date="2024-01-31",
        )
        assert len(result) == 2
        assert result[0]["symbol"] == "AAPL"
        assert result[0]["sentiment_avg"] == 0.65

    @respx.mock
    async def test_http_error_returns_empty(self, newsforge_client):
        """HTTP 500 returns empty list (graceful degradation)."""
        respx.post("http://test-newsforge:8080/api/internal/sentiment/ml-batch").mock(
            return_value=httpx.Response(500, text="Internal Server Error")
        )

        result = await newsforge_client.get_sentiment_batch(
            symbols=["AAPL"],
            market="us",
            start_date="2024-01-01",
            end_date="2024-01-31",
        )
        assert result == []

    @respx.mock
    async def test_timeout_returns_empty(self, newsforge_client):
        """Connection timeout returns empty list."""
        respx.post("http://test-newsforge:8080/api/internal/sentiment/ml-batch").mock(
            side_effect=httpx.ConnectTimeout("timed out")
        )

        result = await newsforge_client.get_sentiment_batch(
            symbols=["AAPL"],
            market="us",
            start_date="2024-01-01",
            end_date="2024-01-31",
        )
        assert result == []

    async def test_empty_symbols_returns_empty(self, newsforge_client):
        """Empty symbols list returns empty without making a request."""
        result = await newsforge_client.get_sentiment_batch(
            symbols=[],
            market="us",
            start_date="2024-01-01",
            end_date="2024-01-31",
        )
        assert result == []

    @respx.mock
    async def test_missing_data_key_returns_empty(self, newsforge_client):
        """Response without 'data' key returns empty list."""
        respx.post("http://test-newsforge:8080/api/internal/sentiment/ml-batch").mock(
            return_value=httpx.Response(200, json={"status": "ok"})
        )

        result = await newsforge_client.get_sentiment_batch(
            symbols=["AAPL"],
            market="us",
            start_date="2024-01-01",
            end_date="2024-01-31",
        )
        assert result == []


# ---------------------------------------------------------------------------
# health_check
# ---------------------------------------------------------------------------


class TestHealthCheck:

    @respx.mock
    async def test_health_check_success(self, newsforge_client):
        """Health check returns True on 200."""
        respx.get("http://test-newsforge:8080/api/v1/health").mock(
            return_value=httpx.Response(200)
        )

        result = await newsforge_client.health_check()
        assert result is True

    @respx.mock
    async def test_health_check_failure(self, newsforge_client):
        """Health check returns False on connection error."""
        respx.get("http://test-newsforge:8080/api/v1/health").mock(
            side_effect=httpx.ConnectError("refused")
        )

        result = await newsforge_client.health_check()
        assert result is False

    @respx.mock
    async def test_health_check_non_200(self, newsforge_client):
        """Health check returns False on non-200 status."""
        respx.get("http://test-newsforge:8080/api/v1/health").mock(
            return_value=httpx.Response(503)
        )

        result = await newsforge_client.health_check()
        assert result is False
