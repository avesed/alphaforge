"""HTTP client for fetching sentiment data from NewsForge.

Async-only — used by feature_service to build sentiment features
for LightGBM training/inference.

Authenticates via X-API-Key header against NewsForge's internal API.
Configuration: NEWSFORGE_URL + NEWSFORGE_API_KEY env vars,
overridable via system_settings DB (admin UI).
"""

from __future__ import annotations

import logging

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)


async def _get_newsforge_config() -> tuple[str, str]:
    """Read NewsForge URL and API key from system_settings DB, with env fallback."""
    settings = get_settings()
    base_url = settings.NEWSFORGE_URL
    api_key = settings.NEWSFORGE_API_KEY

    try:
        from sqlalchemy import select

        from app.core.orm import get_session_factory
        from app.models.system_setting import SystemSetting

        factory = get_session_factory()
        async with factory() as session:
            url_result = await session.execute(
                select(SystemSetting).where(SystemSetting.key == "newsforge_url")
            )
            url_setting = url_result.scalar_one_or_none()
            if url_setting and url_setting.value:
                base_url = url_setting.value

            key_result = await session.execute(
                select(SystemSetting).where(SystemSetting.key == "newsforge_api_key")
            )
            key_setting = key_result.scalar_one_or_none()
            if key_setting and key_setting.value:
                api_key = key_setting.value
    except Exception as e:
        logger.debug(
            "Could not read NewsForge config from DB, using env vars: %s", e,
        )

    return base_url.rstrip("/"), api_key


class NewsForgeClient:
    """Async HTTP client for NewsForge sentiment data."""

    def __init__(self) -> None:
        self._base_url: str | None = None
        self._api_key: str | None = None
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._base_url, self._api_key = await _get_newsforge_config()
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers={"X-API-Key": self._api_key},
                timeout=120.0,
            )
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def get_sentiment_batch(
        self,
        symbols: list[str],
        market: str,
        start_date: str,
        end_date: str,
    ) -> list[dict]:
        """Fetch daily per-symbol sentiment from NewsForge ml-batch endpoint.

        Returns list of dicts with keys:
            symbol, date, sentiment_avg, article_count,
            bullish_ratio, content_score_avg
        """
        if not symbols:
            return []

        client = await self._get_client()
        payload = {
            "symbols": symbols,
            "market": market,
            "start_date": start_date,
            "end_date": end_date,
        }

        try:
            resp = await client.post(
                "/api/internal/sentiment/ml-batch",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json().get("data", [])
            logger.info(
                "NewsForge sentiment: %d rows for %d symbols (%s to %s)",
                len(data), len(symbols), start_date, end_date,
            )
            return data
        except httpx.HTTPStatusError as e:
            logger.warning(
                "NewsForge sentiment HTTP %d: %s",
                e.response.status_code, e.response.text[:200],
            )
            return []
        except Exception as e:
            logger.warning("NewsForge sentiment request failed: %s", e)
            return []

    async def health_check(self) -> bool:
        """Check if NewsForge API is reachable."""
        if not self._api_key:
            base_url, api_key = await _get_newsforge_config()
            if not api_key:
                return False
        try:
            client = await self._get_client()
            resp = await client.get("/api/v1/health")
            return resp.status_code == 200
        except Exception as e:
            logger.debug("NewsForge health check failed: %s", e)
            return False


# ------------------------------------------------------------------
# Module-level singleton
# ------------------------------------------------------------------

_async_client: NewsForgeClient | None = None


async def get_newsforge_client() -> NewsForgeClient:
    global _async_client
    if _async_client is None:
        _async_client = NewsForgeClient()
    return _async_client


async def close_newsforge_client() -> None:
    global _async_client
    if _async_client is not None:
        await _async_client.close()
        _async_client = None
