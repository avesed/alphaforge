"""Lightweight httpx client for ML agents to call ai-gateway.

Provides a singleton async HTTP client that sends chat completion requests
to ai-gateway with proper provider routing headers. Designed for ML agent
workflows that need structured JSON responses from LLMs.

AlphaForge adaptation:
- LLM provider config is read from the local DB (system_settings table)
  via get_db_pool(), replacing WebStock's SettingsCache.
- Usage recording is a no-op placeholder (no llm_usage_records table yet).
"""

import asyncio
import json
import logging
import time
from typing import Any

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)


class MLAgentError(Exception):
    """Raised when LLM call fails."""

    pass


# Strong references to fire-and-forget tasks (prevent GC before completion)
_background_tasks: set[asyncio.Task[None]] = set()


async def _get_prediction_llm_config() -> dict[str, Any]:
    """Read prediction LLM config from system_settings table.

    Returns dict with keys: provider_id (int or None), model (str or None).
    """
    try:
        from app.core.database import get_db_pool

        pool = get_db_pool()
        async with pool.acquire(timeout=10) as conn:
            row = await conn.fetchrow(
                "SELECT prediction_provider_id, prediction_model "
                "FROM system_settings WHERE id = 1"
            )
            if row:
                return {
                    "provider_id": row["prediction_provider_id"],
                    "model": row["prediction_model"],
                }
    except Exception as e:
        logger.debug("Failed to read prediction LLM config from DB: %s", e)
    return {"provider_id": None, "model": None}


class MLAgentClient:
    """AI Gateway client for ML agent LLM calls.

    Lightweight httpx wrapper that sends chat completion requests
    to ai-gateway with proper provider routing headers.
    """

    _client: httpx.AsyncClient | None = None
    _TIMEOUT = 60.0  # seconds
    _RETRY_BACKOFF = 2.0  # seconds

    @classmethod
    def _get_client(cls) -> httpx.AsyncClient:
        """Lazy singleton httpx client."""
        if cls._client is None or cls._client.is_closed:
            cls._client = httpx.AsyncClient(
                timeout=httpx.Timeout(cls._TIMEOUT, connect=10.0),
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=3),
            )
        return cls._client

    @classmethod
    async def close(cls) -> None:
        """Close client on shutdown."""
        if cls._client and not cls._client.is_closed:
            await cls._client.aclose()
            cls._client = None

    async def chat_json(
        self,
        system_prompt: str,
        user_content: str,
        temperature: float = 0.1,
        max_tokens: int = 2000,
    ) -> dict[str, Any]:
        """Send chat request, return parsed JSON response.

        Uses response_format={"type": "json_object"} to force JSON output.
        Retries once with exponential backoff on transient errors.

        Args:
            system_prompt: System message content.
            user_content: User message content.
            temperature: Sampling temperature (default 0.1 for deterministic).
            max_tokens: Maximum tokens in response.

        Returns:
            Parsed JSON dict from the LLM response.

        Raises:
            MLAgentError: On any failure (network, parse, upstream error).
        """
        settings = get_settings()

        # Resolve prediction LLM provider from DB
        llm_config = await _get_prediction_llm_config()
        if not llm_config["provider_id"]:
            raise MLAgentError(
                "Prediction LLM provider not configured. "
                "Set prediction_provider_id in system_settings."
            )

        # Build upstream URL
        gateway_url = settings.AI_GATEWAY_URL.rstrip("/")
        upstream_url = f"{gateway_url}/v1/chat/completions"

        # Build headers
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "X-Provider-Id": str(llm_config["provider_id"]),
        }

        # Build request body
        body: dict[str, Any] = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "response_format": {"type": "json_object"},
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if llm_config["model"]:
            body["model"] = llm_config["model"]

        client = self._get_client()
        last_error: Exception | None = None

        for attempt in range(2):  # 1 initial + 1 retry
            if attempt > 0:
                logger.info(
                    "Retrying LLM request after %.1fs backoff (attempt %d)",
                    self._RETRY_BACKOFF,
                    attempt + 1,
                )
                await asyncio.sleep(self._RETRY_BACKOFF)

            t0 = time.monotonic()
            try:
                resp = await client.post(
                    upstream_url,
                    json=body,
                    headers=headers,
                )
                elapsed_ms = (time.monotonic() - t0) * 1000

                if resp.status_code != 200:
                    detail = resp.text[:500]
                    logger.warning(
                        "AI gateway returned %d (%.0fms): %s",
                        resp.status_code,
                        elapsed_ms,
                        detail,
                    )
                    last_error = MLAgentError(
                        f"AI gateway returned HTTP {resp.status_code}: {detail}"
                    )
                    # Don't retry on 4xx client errors
                    if 400 <= resp.status_code < 500:
                        raise last_error
                    continue

                # Parse upstream response
                resp_json = resp.json()
                content_str = resp_json["choices"][0]["message"]["content"]
                result = json.loads(content_str)

                resp_model = resp_json.get("model", "unknown")
                usage = resp_json.get("usage", {})
                logger.info(
                    "LLM call OK (%.0fms, model=%s, tokens=%s)",
                    elapsed_ms,
                    resp_model,
                    usage.get("total_tokens", "?"),
                )

                return result

            except (httpx.TimeoutException, httpx.ConnectError) as e:
                elapsed_ms = (time.monotonic() - t0) * 1000
                logger.warning(
                    "LLM request failed (%.0fms, attempt %d): %s: %s",
                    elapsed_ms,
                    attempt + 1,
                    type(e).__name__,
                    e,
                )
                last_error = e
                continue

            except (KeyError, IndexError, json.JSONDecodeError) as e:
                elapsed_ms = (time.monotonic() - t0) * 1000
                logger.warning(
                    "Failed to parse LLM response (%.0fms): %s: %s",
                    elapsed_ms,
                    type(e).__name__,
                    e,
                )
                raise MLAgentError(f"Failed to parse LLM response: {e}") from e

            except MLAgentError:
                raise

            except Exception as e:
                elapsed_ms = (time.monotonic() - t0) * 1000
                logger.warning(
                    "Unexpected error in LLM call (%.0fms): %s: %s",
                    elapsed_ms,
                    type(e).__name__,
                    e,
                )
                raise MLAgentError(f"Unexpected error: {e}") from e

        # All retries exhausted
        raise MLAgentError(
            f"LLM request failed after 2 attempts: {last_error}"
        ) from last_error

    async def is_available(self) -> bool:
        """Check if LLM is available (provider configured + gateway reachable).

        Returns:
            True if provider is configured and gateway responds to health check.
        """
        try:
            llm_config = await _get_prediction_llm_config()
            if not llm_config["provider_id"]:
                return False

            settings = get_settings()
            gateway_url = settings.AI_GATEWAY_URL.rstrip("/")
            client = self._get_client()

            resp = await client.get(
                f"{gateway_url}/health",
                timeout=httpx.Timeout(5.0, connect=3.0),
            )
            return resp.status_code == 200
        except Exception as e:
            logger.debug("LLM availability check failed: %s: %s", type(e).__name__, e)
            return False


# Singleton instance
ml_llm_client = MLAgentClient()
