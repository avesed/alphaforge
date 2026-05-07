from fastapi import APIRouter

from app.core.redis import get_redis

router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    checks: dict[str, str] = {"status": "ok"}

    # Check local DB
    try:
        from app.core.database import get_db_pool
        pool = get_db_pool()
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        checks["database"] = "ok"
    except Exception as e:
        checks["database"] = f"error: {e}"
        checks["status"] = "degraded"

    # Check Redis
    try:
        r = await get_redis()
        await r.ping()
        checks["redis"] = "ok"
    except Exception as e:
        checks["redis"] = f"error: {e}"
        checks["status"] = "degraded"

    # Check StockPulse connectivity
    try:
        from app.services.stockpulse_client import _get_stockpulse_config
        import httpx
        sp_url, sp_key = await _get_stockpulse_config()
        if sp_url:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(
                    f"{sp_url.rstrip('/')}/health",
                    headers={"X-API-Key": sp_key} if sp_key else {},
                )
                checks["stockpulse"] = "ok" if resp.status_code == 200 else f"status {resp.status_code}"
        else:
            checks["stockpulse"] = "not configured"
    except Exception as e:
        checks["stockpulse"] = f"error: {e}"
        checks["status"] = "degraded"

    return checks


@router.get("/health/ready")
async def ready():
    from app.core.database import get_db_pool
    pool = get_db_pool()
    async with pool.acquire() as conn:
        await conn.fetchval("SELECT 1")

    r = await get_redis()
    await r.ping()

    return {"status": "ready"}
