"""Prediction API endpoints (X-API-Key auth).

ML prediction workflows: trigger training/inference, retrieve results,
query models, accuracy metrics, and backfill actual returns.
All prediction data is stored locally via PredictionStore.
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from app.core.auth import get_api_consumer
from app.models.api_consumer import ApiConsumer
from app.schemas.prediction import (
    ModelQualityUpdateRequest,
    PredictionRunRequest,
    PredictionRunResponse,
    PredictionTaskStatus,
)
from app.services.prediction_service import prediction_service
from app.services.prediction_store import prediction_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predictions", tags=["predictions"])

# ---------------------------------------------------------------------------
# In-memory background task tracking
# ---------------------------------------------------------------------------

_tasks: dict[str, dict[str, Any]] = {}


def _new_task_id() -> str:
    return uuid.uuid4().hex[:12]


def _sanitize_error(msg: str) -> str:
    """Strip internal URLs from error messages."""
    for token in ("http://stockpulse:", "http://ai-gateway:", "http://redis:"):
        if token in msg:
            idx = msg.find(token)
            end = msg.find(" ", idx)
            msg = msg[:idx] + "[internal]" + (msg[end:] if end > 0 else "")
    return msg


# ---------------------------------------------------------------------------
# Prediction run / task management
# ---------------------------------------------------------------------------


@router.post("/{market}/run", response_model=PredictionRunResponse)
async def run_prediction(
    market: str,
    request: PredictionRunRequest = PredictionRunRequest(),
    consumer: ApiConsumer = Depends(get_api_consumer),
):
    """Trigger a prediction run (train + predict) for the given market.

    Non-blocking: returns task_id immediately. Poll /tasks/{task_id} for status.
    """
    market = market.lower()
    if market not in ("us", "hk", "cn"):
        raise HTTPException(400, f"Unsupported market: {market}")

    try:
        task_id = await prediction_service.run_prediction(
            market=market,
            force_retrain=request.force_retrain,
            forward_days=request.forward_days,
        )
    except RuntimeError as e:
        raise HTTPException(429, str(e))

    return PredictionRunResponse(task_id=task_id, market=market, status="pending")


@router.get("/tasks/{task_id}", response_model=PredictionTaskStatus)
async def get_task_status(
    task_id: str,
    consumer: ApiConsumer = Depends(get_api_consumer),
):
    """Get the status of a prediction task."""
    task = prediction_service.get_task(task_id)
    if not task:
        raise HTTPException(404, f"Task not found: {task_id}")
    return PredictionTaskStatus(
        task_id=task["task_id"],
        status=task["status"],
        progress=task.get("progress"),
        message=task.get("message"),
    )


# ---------------------------------------------------------------------------
# Proxy endpoints -- data lives in StockPulse
# ---------------------------------------------------------------------------


@router.get("/{market}/latest")
async def get_latest_predictions(
    market: str,
    top_n: int = Query(50, ge=1, le=500),
    symbol: Optional[str] = Query(None),
    forward_days: Optional[int] = Query(None, ge=0, le=60),
    consumer: ApiConsumer = Depends(get_api_consumer),
):
    """Get the latest prediction results for a market.

    When forward_days is specified, only return predictions for that horizon.
    forward_days=0 returns combined multi-horizon signal.
    """
    market = market.lower()
    results = await prediction_service.get_latest_predictions(
        market=market, top_n=top_n, symbol=symbol, forward_days=forward_days,
    )
    return {"market": market, "count": len(results), "predictions": results}


@router.get("/models")
async def list_models(
    market: Optional[str] = Query(None),
    consumer: ApiConsumer = Depends(get_api_consumer),
):
    """List available prediction models with metrics."""
    models = await prediction_service.get_models(market=market)
    return {"models": models}


@router.get("/models/{model_id}/feature-importance")
async def get_feature_importance(
    model_id: str,
    consumer: ApiConsumer = Depends(get_api_consumer),
):
    """Get feature importance for a specific model.

    Reads top-30 from StockPulse model metadata and full importance
    from the local model file on disk (if available).
    """
    import json
    import os

    # Validate UUID format
    try:
        uuid.UUID(model_id)
    except ValueError:
        raise HTTPException(400, f"Invalid model ID: {model_id}")

    # Get model metadata from local PredictionStore
    try:
        models = await prediction_store.get_models()
    except Exception as e:
        logger.error("Failed to query models: %s", e)
        raise HTTPException(500, "Failed to query model metadata")

    model_row = None
    for m in models:
        if str(m.get("id")) == model_id:
            model_row = m
            break

    if model_row is None:
        raise HTTPException(404, f"Model not found: {model_id}")

    # Extract top-30 from training_config or feature_importance
    training_config = model_row.get("training_config") or {}
    if isinstance(training_config, str):
        training_config = json.loads(training_config)
    top30 = training_config.get("feature_importance_top30", {})

    # Feature importance stored directly on model row
    stored_importance = model_row.get("feature_importance")
    if not top30 and stored_importance:
        # Build top-30 from the full stored importance
        sorted_items = sorted(
            stored_importance.items(), key=lambda x: x[1], reverse=True,
        )[:30]
        top30 = dict(sorted_items)

    # Try to load full importance from local disk
    full_importance = stored_importance
    model_path = model_row.get("file_path")
    if model_path and not full_importance:
        features_path = os.path.join(os.path.dirname(model_path), "features.json")

        def _read_features():
            if not os.path.exists(features_path):
                return None
            with open(features_path) as f:
                return json.load(f).get("feature_importance")

        try:
            full_importance = await asyncio.to_thread(_read_features)
        except Exception as e:
            logger.warning("Failed to read features.json: %s", e)

    return {
        "model_id": model_id,
        "market": model_row.get("market"),
        "model_date": model_row.get("model_date"),
        "feature_count": model_row.get("feature_count"),
        "top30": top30,
        "full": full_importance,
    }


@router.put("/models/{model_id}/quality")
async def update_model_quality(
    model_id: str,
    request: ModelQualityUpdateRequest,
    consumer: ApiConsumer = Depends(get_api_consumer),
):
    """Admin override: mark model as approved/rejected.

    Updates the quality_passed flag in StockPulse's prediction_models table.
    """
    try:
        uuid.UUID(model_id)
    except ValueError:
        raise HTTPException(400, f"Invalid model ID: {model_id}")

    try:
        quality = "approved" if request.quality_passed else "rejected"
        await prediction_store.update_model_quality(model_id, quality)
    except Exception as e:
        logger.error("Failed to update model quality for %s: %s", model_id, e, exc_info=True)
        raise HTTPException(500, "Failed to update model quality")

    return {"model_id": model_id, "quality_passed": request.quality_passed}


@router.get("/{market}/history")
async def get_prediction_history(
    market: str,
    days: int = Query(30, ge=1, le=365),
    consumer: ApiConsumer = Depends(get_api_consumer),
):
    """Get historical prediction results with actual returns."""
    market = market.lower()
    history = await prediction_service.get_prediction_history(market=market, days=days)
    return {"market": market, "days": days, "count": len(history), "predictions": history}


@router.get("/{market}/accuracy")
async def get_accuracy(
    market: str,
    days: int = Query(30, ge=1, le=365),
    consumer: ApiConsumer = Depends(get_api_consumer),
):
    """Get prediction accuracy summary (direction accuracy, IC, ICIR).

    Proxied via StockPulse performance data -- computes IC/ICIR from
    predictions that have backfilled actual returns.
    """
    market = market.lower()
    if market not in ("us", "hk", "cn"):
        raise HTTPException(400, f"Unsupported market: {market}")

    try:
        data = await prediction_store.get_performance_metrics(market, days)
    except Exception as e:
        logger.error("Failed to fetch accuracy data for %s: %s", market, e)
        raise HTTPException(500, "Failed to fetch accuracy data")

    # Compute summary IC/ICIR from raw performance data
    import numpy as np
    from scipy.stats import spearmanr

    if not data:
        return {
            "market": market,
            "days": days,
            "total_predictions": 0,
            "direction_accuracy": None,
            "ic": None,
            "icir": None,
        }

    # Filter to predictions with actual returns
    with_returns = [d for d in data if d.get("actual_return") is not None]
    if not with_returns:
        return {
            "market": market,
            "days": days,
            "total_predictions": len(data),
            "direction_accuracy": None,
            "ic": None,
            "icir": None,
        }

    scores = np.array([d.get("rank_score", 0) or 0 for d in with_returns])
    returns = np.array([d["actual_return"] for d in with_returns])

    # Direction accuracy based on up_probability (>0.5 = up, <0.5 = down)
    predicted_dirs = np.array([
        1 if (d.get("up_probability") or 0.5) > 0.5 else
        (-1 if (d.get("up_probability") or 0.5) < 0.5 else 0)
        for d in with_returns
    ])
    actual_dirs = np.sign(returns)
    non_zero = actual_dirs != 0
    dir_acc = None
    if non_zero.sum() > 0:
        dir_acc = float((predicted_dirs[non_zero] == actual_dirs[non_zero]).mean())

    # IC: rank correlation between scores and actual returns, per date
    from collections import defaultdict
    date_groups: dict[str, list[dict]] = defaultdict(list)
    for d in with_returns:
        date_groups[d.get("prediction_date", "")].append(d)

    daily_ics = []
    for dt, group in date_groups.items():
        if len(group) < 5:
            continue
        s = np.array([g.get("rank_score", 0) or 0 for g in group])
        r = np.array([g["actual_return"] for g in group])
        ic_val, _ = spearmanr(s, r)
        if not np.isnan(ic_val):
            daily_ics.append(ic_val)

    ic = float(np.mean(daily_ics)) if daily_ics else None
    icir = float(np.mean(daily_ics) / np.std(daily_ics)) if len(daily_ics) > 1 and np.std(daily_ics) > 0 else None

    return {
        "market": market,
        "days": days,
        "total_predictions": len(data),
        "with_returns": len(with_returns),
        "direction_accuracy": round(dir_acc, 4) if dir_acc is not None else None,
        "ic": round(ic, 6) if ic is not None else None,
        "icir": round(icir, 4) if icir is not None else None,
        "daily_ic_count": len(daily_ics),
    }


@router.get("/{market}/performance")
async def get_performance_metrics(
    market: str,
    days: int = Query(90, ge=7, le=365),
    consumer: ApiConsumer = Depends(get_api_consumer),
):
    """Get model performance metrics over time (IC trend, hit rate, spread).

    Proxied from StockPulse -- returns raw performance data for the
    specified lookback period.
    """
    market = market.lower()
    if market not in ("us", "hk", "cn"):
        raise HTTPException(400, f"Unsupported market: {market}")

    try:
        data = await prediction_store.get_performance_metrics(market, days)
    except Exception as e:
        logger.error("Failed to fetch performance metrics for %s: %s", market, e)
        raise HTTPException(500, "Failed to fetch performance metrics")

    return {"market": market, "days": days, "count": len(data), "data": data}


@router.get("/{market}/ic-decay")
async def get_ic_decay(
    market: str,
    days: int = Query(90, ge=7, le=365),
    consumer: ApiConsumer = Depends(get_api_consumer),
):
    """IC at multiple forward horizons (alpha decay curve).

    Computes rank IC for each forward horizon from historical predictions
    that have been backfilled with actual returns.
    """
    market = market.lower()
    if market not in ("us", "hk", "cn"):
        raise HTTPException(400, f"Unsupported market: {market}")

    try:
        data = await prediction_store.get_performance_metrics(market, days)
    except Exception as e:
        logger.error("Failed to fetch IC decay data for %s: %s", market, e)
        raise HTTPException(500, "Failed to fetch IC decay data")

    if not data:
        return {"market": market, "horizons": []}

    # Group by forward_days and compute IC per horizon
    import numpy as np
    from collections import defaultdict
    from scipy.stats import spearmanr

    horizon_groups: dict[int, list[dict]] = defaultdict(list)
    for d in data:
        if d.get("actual_return") is not None and d.get("forward_days") is not None:
            horizon_groups[d["forward_days"]].append(d)

    horizons = []
    for fd in sorted(horizon_groups.keys()):
        group = horizon_groups[fd]
        if len(group) < 10:
            continue
        scores = np.array([g.get("rank_score", 0) or 0 for g in group])
        returns = np.array([g["actual_return"] for g in group])
        ic_val, _ = spearmanr(scores, returns)
        if not np.isnan(ic_val):
            horizons.append({
                "forward_days": fd,
                "ic": round(float(ic_val), 6),
                "n_predictions": len(group),
            })

    return {"market": market, "horizons": horizons}


@router.get("/{market}/turnover")
async def get_turnover_metrics(
    market: str,
    days: int = Query(90, ge=7, le=365),
    top_n: int = Query(20, ge=5, le=100),
    consumer: ApiConsumer = Depends(get_api_consumer),
):
    """Prediction rank stability: rank autocorrelation and top-N retention.

    Measures how much the top-N predicted symbols change between
    consecutive prediction dates.
    """
    market = market.lower()
    if market not in ("us", "hk", "cn"):
        raise HTTPException(400, f"Unsupported market: {market}")

    try:
        data = await prediction_store.get_performance_metrics(market, days)
    except Exception as e:
        logger.error("Failed to fetch turnover data for %s: %s", market, e)
        raise HTTPException(500, "Failed to fetch turnover data")

    if not data:
        return {"market": market, "turnover": []}

    # Group by prediction_date, rank by rank_score, compute retention
    from collections import defaultdict

    date_groups: dict[str, list[dict]] = defaultdict(list)
    for d in data:
        pd_str = d.get("prediction_date", "")
        if pd_str:
            date_groups[pd_str].append(d)

    sorted_dates = sorted(date_groups.keys())
    turnover_points = []
    prev_top: set[str] | None = None

    for dt in sorted_dates:
        group = sorted(date_groups[dt], key=lambda x: x.get("rank_score", 0) or 0, reverse=True)
        current_top = {g["symbol"] for g in group[:top_n]}
        if prev_top is not None and len(prev_top) > 0:
            retention = len(current_top & prev_top) / max(len(prev_top), 1)
            turnover_points.append({
                "date": dt,
                "retention": round(retention, 4),
                "turnover": round(1 - retention, 4),
                "top_n": top_n,
            })
        prev_top = current_top

    return {"market": market, "top_n": top_n, "turnover": turnover_points}


@router.get("/{market}/attribution")
async def get_return_attribution(
    market: str,
    days: int = Query(90, ge=7, le=365),
    top_n: int = Query(20, ge=5, le=100),
    consumer: ApiConsumer = Depends(get_api_consumer),
):
    """Return attribution decomposition: sector, size, and alpha components.

    Proxied from StockPulse performance data. Returns a breakdown of
    how prediction returns decompose across sectors and market cap tiers.
    """
    market = market.lower()
    if market not in ("us", "hk", "cn"):
        raise HTTPException(400, f"Unsupported market: {market}")

    try:
        data = await prediction_store.get_performance_metrics(market, days)
    except Exception as e:
        logger.error("Failed to fetch attribution data for %s: %s", market, e)
        raise HTTPException(500, "Failed to fetch attribution data")

    if not data:
        return {"market": market, "attribution": {}}

    # Filter to predictions with actual returns, take top_n per date
    from collections import defaultdict
    import numpy as np

    date_groups: dict[str, list[dict]] = defaultdict(list)
    for d in data:
        if d.get("actual_return") is not None:
            date_groups[d.get("prediction_date", "")].append(d)

    sector_returns: dict[str, list[float]] = defaultdict(list)
    all_returns: list[float] = []

    for dt, group in date_groups.items():
        top = sorted(group, key=lambda x: x.get("rank_score", 0) or 0, reverse=True)[:top_n]
        for g in top:
            ret = g.get("actual_return", 0)
            all_returns.append(ret)
            sector = g.get("sector") or "Unknown"
            sector_returns[sector].append(ret)

    sector_summary = {}
    for sector, rets in sector_returns.items():
        sector_summary[sector] = {
            "count": len(rets),
            "mean_return": round(float(np.mean(rets)), 6) if rets else 0,
        }

    return {
        "market": market,
        "days": days,
        "top_n": top_n,
        "total_predictions": len(all_returns),
        "mean_return": round(float(np.mean(all_returns)), 6) if all_returns else None,
        "sector_attribution": sector_summary,
    }


@router.get("/{market}/prediction-dates")
async def get_prediction_dates(
    market: str,
    n_dates: int = Query(2, ge=1, le=10),
    forward_days: int = Query(5, ge=0, le=60),
    consumer: ApiConsumer = Depends(get_api_consumer),
):
    """Get predictions for the last N prediction dates (for holdings change).

    Returns grouped predictions to allow the caller to compare ranking
    changes across consecutive prediction dates.
    """
    market = market.lower()
    if market not in ("us", "hk", "cn"):
        raise HTTPException(400, f"Unsupported market: {market}")

    try:
        data = await prediction_store.get_prediction_history(market, days=n_dates * 7)
    except Exception as e:
        logger.error("Failed to fetch prediction dates for %s: %s", market, e)
        raise HTTPException(500, "Failed to fetch prediction dates")

    if not data:
        return {"market": market, "dates": []}

    # prediction_history returns date+count summaries, not full prediction rows.
    # Return the date list for the caller to subsequently query per-date.
    sorted_dates = sorted(
        [d["prediction_date"] for d in data if d.get("prediction_date")],
        reverse=True,
    )[:n_dates]

    result_dates = []
    for dt in sorted(sorted_dates):
        result_dates.append({
            "date": dt,
            "count": next(
                (d["count"] for d in data if d.get("prediction_date") == dt), 0,
            ),
        })

    return {"market": market, "forward_days": forward_days, "dates": result_dates}


@router.post("/backfill-returns")
async def backfill_returns(
    consumer: ApiConsumer = Depends(get_api_consumer),
):
    """Backfill actual returns for past predictions whose forward period has elapsed."""
    result = await prediction_service.backfill_returns()
    return result
