"""Data synchronization API endpoints.

Endpoints:
    POST /data/sync/{market}       - Trigger market data sync (runs in ProcessPoolExecutor)
    GET  /data/status              - Get sync status for all markets
    POST /data/sync/{market}/trigger - Non-blocking sync trigger
    GET  /data/sync/progress       - Get sync progress for all markets
"""
import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException, Query

from app.config import get_settings
from app.context import QlibContext
from app.core.auth import get_api_consumer
from app.executor import run_qlib_background
from app.models.api_consumer import ApiConsumer
from app.schemas.qlib import SyncRequest, SyncStatusResponse
from app.services.data_sync import DataSyncService, get_sync_progress

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/data", tags=["data"])


@router.post("/sync/{market}")
async def sync_market(
    market: str,
    req: SyncRequest,
    consumer: ApiConsumer = Depends(get_api_consumer),
):
    """Trigger data synchronization for a market.

    Downloads EOD data from StockPulse and converts to Qlib .bin format.
    Runs in ProcessPoolExecutor (can take minutes for full markets).

    Supported markets: us, hk, cn, metal.
    """
    # Validate market code
    valid_markets = {"us", "hk", "cn", "sh", "sz", "metal"}
    if market not in valid_markets:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid market: {market}. Valid: {sorted(valid_markets)}",
        )

    logger.info(
        "POST /data/sync/%s symbols=%s update_only=%s consumer=%s",
        market,
        f"{len(req.symbols)} symbols" if req.symbols else "full market",
        req.update_only,
        consumer.name,
    )

    try:
        from app.services.stockpulse_client import _get_stockpulse_config
        sp_url, sp_key = await _get_stockpulse_config()

        result = await run_qlib_background(
            DataSyncService.sync_market,
            market,
            symbols=req.symbols,
            update_only=req.update_only,
            sp_url=sp_url,
            sp_key=sp_key,
        )
        # Reset QlibContext so next factor/backtest request re-reads
        # updated calendar + instruments from disk
        QlibContext.reset()
        logger.info("QlibContext reset after sync for market=%s", market)
        return result
    except TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=f"Data sync for {market} timed out (limit: 30 minutes)",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Data sync failed for %s: %s", market, e, exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Data sync failed: {e}"
        )


@router.get("/status", response_model=SyncStatusResponse)
async def get_data_status(
    consumer: ApiConsumer = Depends(get_api_consumer),
):
    """Get data synchronization status for all markets.

    Returns per-market information: last_sync timestamp, symbol count,
    date range, and whether data exists on disk.
    """
    logger.info("GET /data/status")

    try:
        settings = get_settings()
        markets = DataSyncService.get_sync_status(settings.QLIB_DATA_DIR)
        return SyncStatusResponse(markets=markets)
    except Exception as e:
        logger.error("Failed to get sync status: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to get sync status: {e}"
        )


@router.post("/sync/{market}/trigger")
async def trigger_sync(
    market: str,
    req: SyncRequest = None,
    consumer: ApiConsumer = Depends(get_api_consumer),
):
    """Non-blocking sync trigger. Returns immediately, sync runs in background."""
    valid_markets = {"us", "hk", "cn", "sh", "sz", "metal"}
    if market not in valid_markets:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid market: {market}. Valid: {sorted(valid_markets)}",
        )

    # Default request when called without body
    if req is None:
        req = SyncRequest(market=market, update_only=True)

    # Check if already running via progress file
    settings = get_settings()
    progress = get_sync_progress(settings.QLIB_DATA_DIR)
    if market in progress and progress[market].get("status") == "syncing":
        raise HTTPException(
            status_code=409,
            detail=f"Sync already running for market={market}",
        )

    logger.info(
        "POST /data/sync/%s/trigger update_only=%s",
        market, req.update_only,
    )

    async def _run_sync():
        try:
            await run_qlib_background(
                DataSyncService.sync_market,
                market,
                symbols=req.symbols,
                update_only=req.update_only,
            )
            QlibContext.reset()
            logger.info("Background sync complete for market=%s, QlibContext reset", market)
        except Exception as e:
            logger.error("Background sync failed for market=%s: %s", market, e, exc_info=True)
            # Safety net: clear progress on crash to avoid permanently blocking this market
            from app.services.data_sync import _clear_sync_progress
            _clear_sync_progress(settings.QLIB_DATA_DIR, market)

    asyncio.ensure_future(_run_sync())
    return {"message": f"Sync started for market={market}", "status": "started"}


@router.get("/sync/progress")
async def get_progress(
    consumer: ApiConsumer = Depends(get_api_consumer),
):
    """Get sync progress for all markets."""
    settings = get_settings()
    progress = get_sync_progress(settings.QLIB_DATA_DIR)
    # Return null for markets with no active progress
    all_markets = {"us", "hk", "cn", "metal"}
    result = {m: progress.get(m) for m in all_markets}
    return {"markets": result}
