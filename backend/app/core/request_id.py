"""Request ID middleware.

Pure ASGI middleware (not BaseHTTPMiddleware) to preserve SSE streaming.
Extracts X-Request-ID from incoming headers or generates a short UUID.
The value is stored in a ContextVar for structured logging.
"""

from __future__ import annotations

import uuid
from contextvars import ContextVar

from starlette.types import ASGIApp, Receive, Scope, Send

request_id_var: ContextVar[str] = ContextVar("request_id", default="")


class RequestIdMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        req_id = headers.get(b"x-request-id", b"").decode() or uuid.uuid4().hex[:8]
        token = request_id_var.set(req_id)
        try:
            await self.app(scope, receive, send)
        finally:
            request_id_var.reset(token)
