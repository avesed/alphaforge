"""LLM proxy for RD-Agent subprocess.

Transparent proxy to the AI Gateway. RD-Agent runs as a subprocess
on the same container and needs an OpenAI-compatible endpoint for
its LLM calls. This endpoint forwards requests to the AI Gateway
and streams responses back.

This endpoint does NOT require API key auth -- it is called by the
RD-Agent subprocess on localhost. The AI Gateway handles its own
authentication with the upstream LLM providers.
"""

import logging

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from app.config import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["llm-proxy"])

# Reusable async client for proxying -- created lazily
_proxy_client: httpx.AsyncClient | None = None


async def _get_proxy_client() -> httpx.AsyncClient:
    global _proxy_client
    if _proxy_client is None:
        _proxy_client = httpx.AsyncClient(
            timeout=httpx.Timeout(300.0, connect=10.0),
        )
    return _proxy_client


@router.post("/v1/llm/chat/completions")
async def llm_chat_completions(request: Request):
    """Transparent proxy to AI Gateway chat completions.

    Forwards the full request body to the AI Gateway's OpenAI-compatible
    endpoint. Supports both streaming and non-streaming modes.

    No authentication required -- called by RD-Agent subprocess
    on localhost only.
    """
    settings = get_settings()
    gateway_url = settings.AI_GATEWAY_URL.rstrip("/")
    target_url = f"{gateway_url}/v1/chat/completions"

    try:
        body = await request.body()
    except Exception as e:
        logger.error("Failed to read request body: %s", e)
        return {"error": {"message": "Failed to read request body", "type": "invalid_request"}}

    # Check if streaming is requested
    import json
    try:
        body_json = json.loads(body)
        is_streaming = body_json.get("stream", False)
    except (json.JSONDecodeError, UnicodeDecodeError):
        is_streaming = False

    # Forward headers that matter
    forward_headers: dict[str, str] = {
        "Content-Type": "application/json",
    }
    # Pass through provider routing header if present
    provider_id = request.headers.get("X-Provider-Id")
    if provider_id:
        forward_headers["X-Provider-Id"] = provider_id

    client = await _get_proxy_client()

    if is_streaming:
        # Stream response back
        async def _stream_generator():
            try:
                async with client.stream(
                    "POST",
                    target_url,
                    content=body,
                    headers=forward_headers,
                ) as resp:
                    if resp.status_code != 200:
                        error_body = await resp.aread()
                        logger.error(
                            "AI Gateway returned %d: %s",
                            resp.status_code,
                            error_body[:500],
                        )
                        yield error_body
                        return

                    async for chunk in resp.aiter_bytes():
                        yield chunk
            except httpx.ConnectError:
                logger.error("Cannot connect to AI Gateway at %s", gateway_url)
                yield b'data: {"error": "AI Gateway unavailable"}\n\n'
            except Exception as e:
                logger.error("Streaming proxy error: %s", e, exc_info=True)
                yield f'data: {{"error": "{str(e)}"}}\n\n'.encode()

        return StreamingResponse(
            _stream_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        # Non-streaming: forward and return
        try:
            resp = await client.post(
                target_url,
                content=body,
                headers=forward_headers,
            )
        except httpx.ConnectError:
            logger.error("Cannot connect to AI Gateway at %s", gateway_url)
            return {"error": {"message": "AI Gateway unavailable", "type": "server_error"}}
        except Exception as e:
            logger.error("Proxy request failed: %s", e, exc_info=True)
            return {"error": {"message": str(e), "type": "server_error"}}

        # Return raw response to preserve OpenAI format
        return StreamingResponse(
            iter([resp.content]),
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type", "application/json"),
        )
