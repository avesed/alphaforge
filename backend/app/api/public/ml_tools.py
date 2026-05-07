"""ML Tools API -- decomposed backtest steps (X-API-Key auth).

Five endpoints that expose profile, train, task-status, validate, and
deploy as independent REST calls. The ML Agent in the backend container
chains these to run iterative config optimization without LLM coupling
in AlphaForge.

Additionally exposes rolling-backtest as a single background task.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException

from app.core.auth import get_api_consumer
from app.models.api_consumer import ApiConsumer
from app.schemas.ml_tools import (
    DeployRequest,
    DeployResponse,
    ProfileRequest,
    ProfileResponse,
    RollingBacktestRequest,
    RollingBacktestResponse,
    TrainRequest,
    TrainResponse,
    TrainTaskStatus,
    ValidateRequest,
    ValidateResponse,
)
from app.services.ml_tools_service import ml_tools_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ml-tools", tags=["ml-tools"])


@router.post("/profile", response_model=ProfileResponse)
async def profile(
    request: ProfileRequest,
    consumer: ApiConsumer = Depends(get_api_consumer),
):
    """Profile the feature matrix for a market.

    Returns data statistics (NaN rates, return distribution, sector
    breakdown) and the current MarketConfig as a baseline for the
    ML Agent to reason about config adjustments.
    """
    try:
        result = await ml_tools_service.profile(
            market=request.market,
            cutoff_date=request.cutoff_date,
            validation_days=request.validation_days,
            forward_days=request.forward_days,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Profile failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Profile failed: {e}")

    return ProfileResponse(**result)


@router.post("/train", response_model=TrainResponse)
async def train(
    request: TrainRequest,
    consumer: ApiConsumer = Depends(get_api_consumer),
):
    """Submit a training task with the given config.

    Non-blocking: returns task_id immediately. Poll
    GET /ml-tools/tasks/{task_id} for status and results.
    """
    try:
        task_id = await ml_tools_service.submit_training(
            market=request.market,
            cutoff_date=request.cutoff_date,
            forward_days=request.forward_days,
            config=request.config,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=429, detail=str(e))
    except Exception as e:
        logger.error("Train submission failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Train submission failed: {e}"
        )

    return TrainResponse(task_id=task_id, status="submitted")


@router.get("/tasks/{task_id}", response_model=TrainTaskStatus)
async def get_task(
    task_id: str,
    consumer: ApiConsumer = Depends(get_api_consumer),
):
    """Get the status of a training task.

    Returns status, progress percentage, and results (once completed)
    or error message (on failure).
    """
    result = ml_tools_service.get_task(task_id)
    if result is None:
        raise HTTPException(
            status_code=404, detail=f"Task not found: {task_id}"
        )

    return TrainTaskStatus(**result)


@router.post("/validate", response_model=ValidateResponse)
async def validate(
    request: ValidateRequest,
    consumer: ApiConsumer = Depends(get_api_consumer),
):
    """Run out-of-sample validation using the trained models.

    Requires a completed training task. Performs multi-day inference
    on the validation window (dates after cutoff_date) and computes
    IC, ICIR, quintile spread, direction accuracy, and hit rate.
    """
    try:
        result = await ml_tools_service.validate(
            task_id=request.task_id,
            cutoff_date=request.cutoff_date,
            validation_days=request.validation_days,
            forward_days=request.forward_days,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Validate failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Validation failed: {e}"
        )

    return ValidateResponse(**result)


@router.post("/rolling-backtest", response_model=RollingBacktestResponse)
async def rolling_backtest(
    request: RollingBacktestRequest,
    consumer: ApiConsumer = Depends(get_api_consumer),
):
    """Submit a rolling retrain backtest task.

    Non-blocking: returns task_id immediately. Poll
    GET /ml-tools/tasks/{task_id} for status and results.

    Rolling backtest retrains the model every retrain_interval trading
    days during the validation window, simulating production behavior.
    """
    try:
        task_id = await ml_tools_service.submit_rolling_backtest(
            market=request.market,
            cutoff_date=request.cutoff_date,
            validation_days=request.validation_days,
            forward_days=request.forward_days,
            retrain_interval=request.retrain_interval,
            config=request.config,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=429, detail=str(e))
    except Exception as e:
        logger.error(
            "Rolling backtest submission failed: %s", e, exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail=f"Rolling backtest submission failed: {e}",
        )

    return RollingBacktestResponse(task_id=task_id, status="submitted")


@router.post("/deploy", response_model=DeployResponse)
async def deploy(
    request: DeployRequest,
    consumer: ApiConsumer = Depends(get_api_consumer),
):
    """Persist a validated backtest result to the database.

    Writes the effective config and metrics as a completed backtest
    record in ml_backtests, making it visible through the standard
    backtest list/detail endpoints.
    """
    try:
        result = await ml_tools_service.deploy(
            market=request.market,
            backtest_id=request.backtest_id,
            effective_config=request.effective_config,
            iteration=request.iteration,
            val_ic=request.val_ic,
            train_ic=request.train_ic,
            train_icir=request.train_icir,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Deploy failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Deploy failed: {e}")

    return DeployResponse(**result)
