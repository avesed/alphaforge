"""Admin dashboard statistics endpoint."""

from __future__ import annotations

import logging

import httpx
from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.auth import require_admin
from app.core.orm import get_db
from app.models.api_consumer import ApiConsumer
from app.models.user import User
from app.schemas.base import CamelModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/stats", tags=["admin-stats"])


class DashboardStats(CamelModel):
    user_count: int
    consumer_count: int
    scheduler_is_leader: bool
    scheduler_job_count: int
    stockpulse_connected: bool
    redis_connected: bool


@router.get("", response_model=DashboardStats)
async def get_stats(
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Return basic dashboard statistics."""

    # DB counts
    user_result = await db.execute(select(func.count(User.id)))
    user_count = user_result.scalar_one()

    consumer_result = await db.execute(select(func.count(ApiConsumer.id)))
    consumer_count = consumer_result.scalar_one()

    # Scheduler status
    from app.core.scheduler import get_scheduler, is_leader

    scheduler = get_scheduler()
    scheduler_is_leader = is_leader()
    scheduler_job_count = len(scheduler.get_jobs()) if scheduler is not None else 0

    # StockPulse health check
    stockpulse_connected = False
    try:
        from app.services.stockpulse_client import _get_stockpulse_config
        sp_url, sp_key = await _get_stockpulse_config()
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(
                f"{sp_url.rstrip('/')}/health",
                headers={"X-API-Key": sp_key} if sp_key else {},
            )
            stockpulse_connected = resp.status_code == 200
    except Exception:
        pass

    # Redis health check
    redis_connected = False
    try:
        from app.core.redis import get_redis
        r = await get_redis()
        pong = await r.ping()
        redis_connected = bool(pong)
    except Exception:
        pass

    return DashboardStats(
        user_count=user_count,
        consumer_count=consumer_count,
        scheduler_is_leader=scheduler_is_leader,
        scheduler_job_count=scheduler_job_count,
        stockpulse_connected=stockpulse_connected,
        redis_connected=redis_connected,
    )
