"""Schemas for API consumer management."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from app.schemas.base import CamelModel


class ConsumerCreateRequest(CamelModel):
    name: str
    description: str | None = None
    rate_limit: int = 100
    allowed_endpoints: list[str] | None = None


class ConsumerUpdateRequest(CamelModel):
    name: str | None = None
    description: str | None = None
    rate_limit: int | None = None
    is_active: bool | None = None


class ConsumerResponse(CamelModel):
    id: UUID
    name: str
    api_key_prefix: str
    description: str | None = None
    is_active: bool
    rate_limit: int
    allowed_endpoints: list[str] | None = None
    last_used_at: datetime | None = None
    created_at: datetime


class ConsumerWithKeyResponse(ConsumerResponse):
    """Returned only on creation -- includes the raw API key (shown once)."""
    raw_api_key: str
