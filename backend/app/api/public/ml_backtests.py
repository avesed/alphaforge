"""ML Backtest API endpoints (X-API-Key auth).

Historical cutoff backtesting with optional LLM agent loop.
Separate from Qlib backtests -- these use the LightGBM ML pipeline.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Query

from app.core.auth import get_api_consumer
from app.models.api_consumer import ApiConsumer
from app.schemas.backtest import (
    BacktestDetail,
    BacktestListResponse,
    BacktestRequest,
    BacktestStartResponse,
    BacktestSummary,
    BacktestTaskStatus,
)
from app.services.ml_backtest_service import backtest_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predictions", tags=["ml-backtests"])


@router.post("/{market}/backtest")
async def start_backtest(
    market: str,
    request: BacktestRequest,
    consumer: ApiConsumer = Depends(get_api_consumer),
):
    """Start a historical cutoff backtest (static or rolling).

    Non-blocking: returns task_id + backtest_id immediately.
    Poll /predictions/backtests/tasks/{task_id} for progress.
    """
    market = market.lower()
    if market not in ("us", "hk", "cn"):
        raise HTTPException(400, f"Unsupported market: {market}")

    try:
        task_id, backtest_id = await backtest_service.start_backtest(
            market=market,
            cutoff_date=request.cutoff_date,
            validation_days=request.validation_days,
            forward_days=request.forward_days,
            config_override=request.config_override,
            use_llm_agents=request.use_llm_agents,
            max_iterations=request.max_iterations,
            backtest_type=request.backtest_type,
            retrain_interval=request.retrain_interval,
        )
    except RuntimeError as e:
        raise HTTPException(429, str(e))

    return BacktestStartResponse(
        task_id=task_id,
        backtest_id=backtest_id,
        market=market,
        status="pending",
    )


@router.post("/{market}/agent-backtest")
async def start_agent_backtest(
    market: str,
    request: BacktestRequest,
    consumer: ApiConsumer = Depends(get_api_consumer),
):
    """Start an agent-mode backtest with LLM-driven config optimization.

    Same as regular backtest but forces use_llm_agents=True and
    allows up to max_iterations rounds of Profiler -> Strategist ->
    Train -> Evaluate iteration.
    """
    market = market.lower()
    if market not in ("us", "hk", "cn"):
        raise HTTPException(400, f"Unsupported market: {market}")

    try:
        task_id, backtest_id = await backtest_service.start_backtest(
            market=market,
            cutoff_date=request.cutoff_date,
            validation_days=request.validation_days,
            forward_days=request.forward_days,
            config_override=request.config_override,
            use_llm_agents=True,
            max_iterations=max(request.max_iterations, 2),
            backtest_type=request.backtest_type,
            retrain_interval=request.retrain_interval,
        )
    except RuntimeError as e:
        raise HTTPException(429, str(e))

    return BacktestStartResponse(
        task_id=task_id,
        backtest_id=backtest_id,
        market=market,
        status="pending",
    )


@router.get("/backtests/tasks/{task_id}")
async def get_backtest_task_status(
    task_id: str,
    consumer: ApiConsumer = Depends(get_api_consumer),
):
    """Get real-time backtest task progress with structured observability.

    Returns phase, iteration progress, elapsed time, and per-iteration
    metric snapshots.
    """
    task = backtest_service.get_task(task_id)
    if not task:
        raise HTTPException(404, f"Backtest task not found: {task_id}")
    return task


@router.get("/{market}/backtests")
async def list_backtests(
    market: str,
    limit: int = Query(50, ge=1, le=200),
    consumer: ApiConsumer = Depends(get_api_consumer),
):
    """List historical backtests for a market."""
    market = market.lower()
    if market not in ("us", "hk", "cn"):
        raise HTTPException(400, f"Unsupported market: {market}")

    backtests = await backtest_service.list_backtests(market=market, limit=limit)
    return BacktestListResponse(
        backtests=[
            BacktestSummary(
                id=str(b["id"]),
                market=b["market"],
                cutoff_date=b["cutoff_date"],
                validation_days=b["validation_days"],
                forward_days=b["forward_days"],
                status=b["status"],
                train_ic=b.get("train_ic"),
                train_icir=b.get("train_icir"),
                val_ic=b.get("val_ic"),
                val_icir=b.get("val_icir"),
                val_direction_accuracy=b.get("val_direction_accuracy"),
                val_spread=b.get("val_spread"),
                agent_iteration=b.get("agent_iteration"),
                duration_seconds=b.get("duration_seconds"),
                created_at=b.get("created_at"),
                completed_at=b.get("completed_at"),
            )
            for b in backtests
        ],
        total=len(backtests),
    )


@router.get("/backtests/{backtest_id}")
async def get_backtest_detail(
    backtest_id: str,
    consumer: ApiConsumer = Depends(get_api_consumer),
):
    """Get full backtest detail including validation metrics and results."""
    result = await backtest_service.get_backtest(backtest_id)
    if not result:
        raise HTTPException(404, f"Backtest not found: {backtest_id}")

    return BacktestDetail(
        id=str(result["id"]),
        market=result["market"],
        cutoff_date=result["cutoff_date"],
        validation_days=result["validation_days"],
        forward_days=result["forward_days"],
        status=result["status"],
        config_override=result.get("config_override"),
        effective_config=result.get("effective_config", {}),
        train_ic=result.get("train_ic"),
        train_icir=result.get("train_icir"),
        train_ndcg=result.get("train_ndcg"),
        fold_ics=result.get("fold_ics"),
        ensemble_size=result.get("ensemble_size"),
        feature_count=result.get("feature_count"),
        symbol_count=result.get("symbol_count"),
        val_ic=result.get("val_ic"),
        val_icir=result.get("val_icir"),
        val_direction_accuracy=result.get("val_direction_accuracy"),
        val_spread=result.get("val_spread"),
        val_q1_return=result.get("val_q1_return"),
        val_q5_return=result.get("val_q5_return"),
        val_hit_rate=result.get("val_hit_rate"),
        val_max_drawdown=result.get("val_max_drawdown"),
        results=result.get("results", {}),
        error=result.get("error"),
        agent_run_id=result.get("agent_run_id"),
        agent_iteration=result.get("agent_iteration"),
        duration_seconds=result.get("duration_seconds"),
        created_at=result.get("created_at"),
        completed_at=result.get("completed_at"),
    )


@router.delete("/backtests/{backtest_id}")
async def delete_backtest(
    backtest_id: str,
    consumer: ApiConsumer = Depends(get_api_consumer),
):
    """Delete a backtest record."""
    success = await backtest_service.delete_backtest(backtest_id)
    if not success:
        raise HTTPException(404, f"Backtest not found: {backtest_id}")
    return {"deleted": True, "backtest_id": backtest_id}
