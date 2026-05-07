"""Admin API consumer management endpoints.

Consumers are machine clients (e.g. WebStock) that authenticate via
X-API-Key. The raw key is shown exactly once on creation -- only the
SHA-256 hash and an 8-char prefix are stored.
"""

from __future__ import annotations

import hashlib
import logging
import secrets
from datetime import datetime, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import require_admin
from app.core.orm import get_db
from app.models.api_consumer import ApiConsumer
from app.models.user import User
from app.schemas.consumer import (
    ConsumerCreateRequest,
    ConsumerResponse,
    ConsumerUpdateRequest,
    ConsumerWithKeyResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/consumers", tags=["admin-consumers"])


@router.get("", response_model=list[ConsumerResponse])
async def list_consumers(
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """List all API consumers."""
    result = await db.execute(
        select(ApiConsumer).order_by(ApiConsumer.created_at.desc())
    )
    return result.scalars().all()


@router.post("", response_model=ConsumerWithKeyResponse, status_code=status.HTTP_201_CREATED)
async def create_consumer(
    body: ConsumerCreateRequest,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Create a new API consumer. Returns the raw API key ONCE."""
    existing = await db.execute(
        select(ApiConsumer).where(ApiConsumer.name == body.name)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Consumer with name '{body.name}' already exists",
        )

    raw_key = secrets.token_urlsafe(32)
    key_prefix = raw_key[:8]
    key_hash = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    consumer = ApiConsumer(
        name=body.name,
        api_key=key_hash,
        api_key_prefix=key_prefix,
        description=body.description,
        rate_limit=body.rate_limit,
        allowed_endpoints=body.allowed_endpoints,
    )
    db.add(consumer)
    await db.flush()

    logger.info("Admin %s created consumer '%s' (prefix=%s)", admin.email, body.name, key_prefix)

    return ConsumerWithKeyResponse(
        id=consumer.id,
        name=consumer.name,
        api_key_prefix=key_prefix,
        description=consumer.description,
        is_active=consumer.is_active,
        rate_limit=consumer.rate_limit,
        allowed_endpoints=consumer.allowed_endpoints,
        last_used_at=consumer.last_used_at,
        created_at=consumer.created_at,
        raw_api_key=raw_key,
    )


@router.put("/{consumer_id}", response_model=ConsumerResponse)
async def update_consumer(
    consumer_id: UUID,
    body: ConsumerUpdateRequest,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Update consumer name, description, rate_limit, or is_active."""
    result = await db.execute(
        select(ApiConsumer).where(ApiConsumer.id == consumer_id)
    )
    consumer = result.scalar_one_or_none()
    if consumer is None:
        raise HTTPException(status_code=404, detail="Consumer not found")

    if body.name is not None:
        # Check uniqueness if name is changing
        if body.name != consumer.name:
            dup = await db.execute(
                select(ApiConsumer).where(ApiConsumer.name == body.name)
            )
            if dup.scalar_one_or_none():
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Consumer with name '{body.name}' already exists",
                )
        consumer.name = body.name
    if body.description is not None:
        consumer.description = body.description
    if body.rate_limit is not None:
        consumer.rate_limit = body.rate_limit
    if body.is_active is not None:
        consumer.is_active = body.is_active

    consumer.updated_at = datetime.now(timezone.utc)

    logger.info("Admin %s updated consumer %s", admin.email, consumer_id)
    return consumer


@router.delete("/{consumer_id}", status_code=status.HTTP_204_NO_CONTENT)
async def deactivate_consumer(
    consumer_id: UUID,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Deactivate an API consumer (soft delete)."""
    result = await db.execute(
        select(ApiConsumer).where(ApiConsumer.id == consumer_id)
    )
    consumer = result.scalar_one_or_none()
    if consumer is None:
        raise HTTPException(status_code=404, detail="Consumer not found")

    consumer.is_active = False
    consumer.updated_at = datetime.now(timezone.utc)
    logger.info("Admin %s deactivated consumer %s", admin.email, consumer_id)
