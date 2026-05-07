"""Backtest API endpoints.

Endpoints:
    POST   /backtests              - Create a new backtest (async, returns task_id)
    GET    /backtests              - List all backtest tasks
    GET    /backtests/{task_id}    - Get backtest status + results
    POST   /backtests/{task_id}/cancel - Cancel a running backtest
    DELETE /backtests/{task_id}    - Delete a completed/failed backtest
"""
import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException

from app.core.auth import get_api_consumer
from app.models.api_consumer import ApiConsumer
from app.schemas.qlib import (
    BacktestCreateRequest,
    BacktestListResponse,
    BacktestStatusResponse,
)
from app.services.backtest_service import BacktestService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/backtests", tags=["backtests"])


@router.post("", response_model=BacktestStatusResponse, status_code=201)
async def create_backtest(
    req: BacktestCreateRequest,
    consumer: ApiConsumer = Depends(get_api_consumer),
):
    """Create a new backtest task.

    Launches the backtest in ProcessPoolExecutor (non-blocking). Returns
    immediately with task_id and status=pending. Poll GET /backtests/{task_id}
    for progress and results.

    Strategy types:
    - topk: Buy top K stocks by factor score, drop N worst at each rebalance
    - signal: Buy/sell based on expression threshold crossing
    - long_short: Long top decile, short bottom decile
    """
    logger.info(
        "POST /backtests name=%s strategy=%s symbols=%d dates=%s~%s consumer=%s",
        req.name,
        req.strategy_type.value,
        len(req.symbols),
        req.start_date,
        req.end_date,
        consumer.name,
    )

    # Validate date range
    if req.end_date <= req.start_date:
        raise HTTPException(
            status_code=400,
            detail="end_date must be after start_date",
        )

    # Build config dict for the service
    config = {
        "name": req.name,
        "market": req.market.value,
        "symbols": req.symbols,
        "start_date": req.start_date.isoformat(),
        "end_date": req.end_date.isoformat(),
        "strategy_type": req.strategy_type.value,
        "strategy_config": req.strategy_config,
        "execution_config": req.execution_config,
    }

    try:
        task_id = await BacktestService.create(config)
    except RuntimeError as e:
        raise HTTPException(status_code=429, detail=str(e))
    except Exception as e:
        logger.error("Failed to create backtest: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to create backtest: {e}"
        )

    status = BacktestService.get_status(task_id)
    if status is None:
        raise HTTPException(status_code=500, detail="Task created but not found")

    return status


@router.get("", response_model=BacktestListResponse)
async def list_backtests(
    consumer: ApiConsumer = Depends(get_api_consumer),
):
    """List all backtest tasks, newest first.

    Returns all tasks regardless of status (pending, running, completed,
    failed, cancelled). Tasks are stored in-memory and cleared on restart.
    """
    logger.info("GET /backtests")

    tasks = BacktestService.list_tasks()
    return BacktestListResponse(tasks=tasks, total=len(tasks))


@router.get("/{task_id}", response_model=BacktestStatusResponse)
async def get_backtest_status(
    task_id: str,
    consumer: ApiConsumer = Depends(get_api_consumer),
):
    """Get backtest task status and results.

    Returns current progress (0-100) while running, and full results
    (equity curve, risk metrics, trades) once completed.
    """
    logger.info("GET /backtests/%s", task_id)

    status = BacktestService.get_status(task_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Backtest task not found: {task_id}")

    return status


@router.post("/{task_id}/cancel")
async def cancel_backtest(
    task_id: str,
    consumer: ApiConsumer = Depends(get_api_consumer),
):
    """Cancel a running or pending backtest.

    Only tasks with status pending or running can be cancelled.
    """
    logger.info("POST /backtests/%s/cancel", task_id)

    status = BacktestService.get_status(task_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Backtest task not found: {task_id}")

    if not BacktestService.cancel(task_id):
        raise HTTPException(
            status_code=409,
            detail=f"Cannot cancel task in status: {status['status']}",
        )

    return {"task_id": task_id, "status": "cancelled"}


@router.delete("/{task_id}")
async def delete_backtest(
    task_id: str,
    consumer: ApiConsumer = Depends(get_api_consumer),
):
    """Delete a completed, failed, or cancelled backtest.

    Running or pending tasks must be cancelled first.
    """
    logger.info("DELETE /backtests/%s", task_id)

    status = BacktestService.get_status(task_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Backtest task not found: {task_id}")

    if not BacktestService.delete(task_id):
        raise HTTPException(
            status_code=409,
            detail=f"Cannot delete task in status: {status['status']}. Cancel it first.",
        )

    return {"task_id": task_id, "deleted": True}
