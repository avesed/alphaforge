"""Admin system settings endpoints (key-value store)."""

from __future__ import annotations

import logging

import httpx
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.auth import require_admin
from app.core.orm import get_db
from app.models.system_setting import SystemSetting
from app.models.user import User
from app.schemas.settings import SettingResponse, SettingUpdate
from app.services.newsforge_client import close_newsforge_client
from app.services.stockpulse_client import close_stockpulse_async_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/settings", tags=["admin-settings"])

# Keys that cannot be read or modified via the admin API
_PROTECTED_KEYS = frozenset({"jwt_secret_key"})


@router.get("", response_model=dict[str, str])
async def get_all_settings(
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Get all settings as a key-value dict."""
    result = await db.execute(
        select(SystemSetting).order_by(SystemSetting.key)
    )
    settings = result.scalars().all()
    return {s.key: s.value for s in settings if s.key not in _PROTECTED_KEYS}


@router.put("", response_model=dict[str, str])
async def bulk_update_settings(
    body: dict[str, str],
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Bulk update settings. Accepts a dict of key-value pairs."""
    protected = _PROTECTED_KEYS & body.keys()
    if protected:
        raise HTTPException(
            status_code=403,
            detail=f"Protected settings cannot be modified: {', '.join(sorted(protected))}",
        )

    for key, value in body.items():
        result = await db.execute(
            select(SystemSetting).where(SystemSetting.key == key)
        )
        setting = result.scalar_one_or_none()
        if setting is not None:
            setting.value = value
        else:
            db.add(SystemSetting(key=key, value=value))

    logger.info("Admin %s bulk-updated %d settings", admin.email, len(body))

    # Reset async clients if connection settings changed
    _stockpulse_keys = {"stockpulse_url", "stockpulse_api_key"}
    if _stockpulse_keys & body.keys():
        await close_stockpulse_async_client()
        logger.info("StockPulse async client reset due to settings change")

    _newsforge_keys = {"newsforge_url", "newsforge_api_key"}
    if _newsforge_keys & body.keys():
        await close_newsforge_client()
        logger.info("NewsForge client reset due to settings change")

    # Return the full settings dict after update
    result = await db.execute(
        select(SystemSetting).order_by(SystemSetting.key)
    )
    settings = result.scalars().all()
    return {s.key: s.value for s in settings if s.key not in _PROTECTED_KEYS}


@router.get("/stockpulse/test")
async def test_stockpulse_connection(
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Test connectivity to the StockPulse API."""
    settings = get_settings()

    # Read from system_settings DB, fallback to env vars
    url_result = await db.execute(
        select(SystemSetting).where(SystemSetting.key == "stockpulse_url")
    )
    url_setting = url_result.scalar_one_or_none()
    url = url_setting.value if url_setting else settings.STOCKPULSE_URL

    key_result = await db.execute(
        select(SystemSetting).where(SystemSetting.key == "stockpulse_api_key")
    )
    key_setting = key_result.scalar_one_or_none()
    api_key = key_setting.value if key_setting else settings.STOCKPULSE_API_KEY

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{url.rstrip('/')}/health",
                headers={"X-API-Key": api_key},
            )
            if resp.status_code == 200:
                return {"connected": True, "error": None}
            return {
                "connected": False,
                "error": f"HTTP {resp.status_code}: {resp.text[:200]}",
            }
    except Exception as e:
        return {"connected": False, "error": f"{type(e).__name__}: {e}"}


@router.get("/newsforge/test")
async def test_newsforge_connection(
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Test connectivity to the NewsForge API."""
    settings = get_settings()

    url_result = await db.execute(
        select(SystemSetting).where(SystemSetting.key == "newsforge_url")
    )
    url_setting = url_result.scalar_one_or_none()
    url = url_setting.value if url_setting else settings.NEWSFORGE_URL

    key_result = await db.execute(
        select(SystemSetting).where(SystemSetting.key == "newsforge_api_key")
    )
    key_setting = key_result.scalar_one_or_none()
    api_key = key_setting.value if key_setting else settings.NEWSFORGE_API_KEY

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{url.rstrip('/')}/api/v1/health",
                headers={"X-API-Key": api_key},
            )
            if resp.status_code == 200:
                return {"connected": True, "error": None}
            return {
                "connected": False,
                "error": f"HTTP {resp.status_code}: {resp.text[:200]}",
            }
    except Exception as e:
        return {"connected": False, "error": f"{type(e).__name__}: {e}"}


@router.get("/{key}", response_model=SettingResponse)
async def get_setting(
    key: str,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Get a single setting by key."""
    if key in _PROTECTED_KEYS:
        raise HTTPException(status_code=403, detail="This setting is protected")

    result = await db.execute(
        select(SystemSetting).where(SystemSetting.key == key)
    )
    setting = result.scalar_one_or_none()
    if setting is None:
        raise HTTPException(status_code=404, detail="Setting not found")

    return setting


@router.put("/{key}", response_model=SettingResponse)
async def upsert_setting(
    key: str,
    body: SettingUpdate,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Create or update a single setting."""
    if key in _PROTECTED_KEYS:
        raise HTTPException(status_code=403, detail="This setting cannot be modified")

    result = await db.execute(
        select(SystemSetting).where(SystemSetting.key == key)
    )
    setting = result.scalar_one_or_none()
    if setting is not None:
        setting.value = body.value
    else:
        setting = SystemSetting(key=key, value=body.value)
        db.add(setting)
        await db.flush()

    logger.info("Admin %s updated setting '%s'", admin.email, key)
    return setting
