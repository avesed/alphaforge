"""Expression engine API endpoints.

Endpoints:
    POST /expression/evaluate  - Evaluate a Qlib expression for a single symbol
    POST /expression/batch     - Evaluate across multiple symbols (cross-sectional)
    POST /expression/validate  - Validate expression syntax without Qlib call
"""
import logging

from fastapi import APIRouter, Depends, HTTPException

from app.core.auth import get_api_consumer
from app.executor import run_qlib_quick
from app.models.api_consumer import ApiConsumer
from app.schemas.qlib import (
    ExpressionBatchRequest,
    ExpressionBatchResult,
    ExpressionEvaluateRequest,
    ExpressionResult,
    ExpressionValidateRequest,
    ValidationResult,
)
from app.services.expression_engine import ExpressionEngine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/expression", tags=["expression"])


@router.post("/evaluate", response_model=ExpressionResult)
async def evaluate_expression(
    req: ExpressionEvaluateRequest,
    consumer: ApiConsumer = Depends(get_api_consumer),
):
    """Evaluate a Qlib expression for a single stock.

    Returns time-series data points for the expression.
    Runs in ThreadPoolExecutor (typical: 1-10s).
    """
    logger.info(
        "POST /expression/evaluate symbol=%s expression=%s market=%s",
        req.symbol,
        req.expression[:80],
        req.market.value,
    )

    try:
        result = await run_qlib_quick(
            ExpressionEngine.evaluate,
            req.symbol,
            req.expression,
            req.market.value,
            start_date=str(req.start_date) if req.start_date else None,
            end_date=str(req.end_date) if req.end_date else None,
            period=req.period,
        )
        if "error" in result and result["error"]:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except HTTPException:
        raise
    except TimeoutError:
        raise HTTPException(
            status_code=504, detail="Expression evaluation timed out"
        )
    except Exception as e:
        logger.error(
            "Expression evaluation failed for %s: %s",
            req.symbol,
            e,
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail="Internal expression evaluation error"
        )


@router.post("/batch", response_model=ExpressionBatchResult)
async def evaluate_batch(
    req: ExpressionBatchRequest,
    consumer: ApiConsumer = Depends(get_api_consumer),
):
    """Evaluate a Qlib expression across multiple symbols (cross-sectional).

    Returns the latest value for each symbol on the target date.
    Runs in ThreadPoolExecutor (typical: 2-15s).
    """
    logger.info(
        "POST /expression/batch symbols=%d expression=%s market=%s",
        len(req.symbols),
        req.expression[:80],
        req.market.value,
    )

    try:
        result = await run_qlib_quick(
            ExpressionEngine.evaluate_batch,
            req.symbols,
            req.expression,
            req.market.value,
            target_date=str(req.target_date) if req.target_date else None,
            timeout=120,  # Batch operations get more time
        )
        if "error" in result and result["error"]:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except HTTPException:
        raise
    except TimeoutError:
        raise HTTPException(
            status_code=504, detail="Batch expression evaluation timed out"
        )
    except Exception as e:
        logger.error(
            "Batch expression evaluation failed: %s", e, exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail="Internal batch expression evaluation error",
        )


@router.post("/validate", response_model=ValidationResult)
async def validate_expression(
    req: ExpressionValidateRequest,
    consumer: ApiConsumer = Depends(get_api_consumer),
):
    """Validate a Qlib expression without executing it.

    Pure string validation -- checks operator whitelist, variable names,
    bracket matching, and dangerous patterns. No Qlib call needed.
    """
    logger.info(
        "POST /expression/validate expression=%s", req.expression[:80]
    )

    is_valid, error, operators = ExpressionEngine.validate(req.expression)
    return {
        "valid": is_valid,
        "error": error if not is_valid else None,
        "operators_used": operators,
    }
