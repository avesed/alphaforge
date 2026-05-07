"""Factor computation and analysis API endpoints.

Endpoints:
    GET  /factors/{symbol}         - Full Alpha158 factor computation
    GET  /factors/{symbol}/summary - Top-10 factor summary (LLM-optimized)
    POST /factors/ic               - IC/ICIR computation for a universe
    POST /factors/cs-rank          - Cross-sectional percentile ranking
"""
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from app.core.auth import get_api_consumer
from app.executor import run_qlib_quick
from app.models.api_consumer import ApiConsumer
from app.schemas.qlib import (
    CSRankRequest,
    CSRankResult,
    FactorResult,
    FactorSummary,
    ICRequest,
    ICResult,
    MarketCode,
)
from app.services.factor_analysis import FactorAnalysisService
from app.services.factor_service import FactorService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/factors", tags=["factors"])


@router.get("/{symbol}", response_model=FactorResult)
async def get_factors(
    symbol: str,
    market: MarketCode = Query(MarketCode.US, description="Market code"),
    alpha_type: str = Query("alpha158", description="Factor set type"),
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    consumer: ApiConsumer = Depends(get_api_consumer),
):
    """Compute Alpha158 factors for a single stock.

    Returns time-series factor values and top factors ranked by z-score.
    Runs in ThreadPoolExecutor (typical: 2-10s).
    """
    logger.info(
        "GET /factors/%s market=%s alpha_type=%s consumer=%s",
        symbol, market.value, alpha_type, consumer.name,
    )

    try:
        result = await run_qlib_quick(
            FactorService.compute_factors,
            symbol,
            market.value,
            alpha_type,
            start_date=start_date,
            end_date=end_date,
        )
        return result
    except TimeoutError:
        raise HTTPException(
            status_code=504, detail="Factor computation timed out"
        )
    except Exception as e:
        logger.error("Factor computation failed for %s: %s", symbol, e, exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Factor computation failed: {e}"
        )


@router.get("/{symbol}/summary", response_model=FactorSummary)
async def get_factor_summary(
    symbol: str,
    market: MarketCode = Query(MarketCode.US, description="Market code"),
    consumer: ApiConsumer = Depends(get_api_consumer),
):
    """Get top-10 factor summary for a stock.

    Optimized for LLM agents: returns only the most significant factors
    by z-score with compact output.
    Runs in ThreadPoolExecutor (typical: 2-10s).
    """
    logger.info("GET /factors/%s/summary market=%s", symbol, market.value)

    try:
        result = await run_qlib_quick(
            FactorService.get_factor_summary,
            symbol,
            market.value,
        )
        return result
    except TimeoutError:
        raise HTTPException(
            status_code=504, detail="Factor summary computation timed out"
        )
    except Exception as e:
        logger.error(
            "Factor summary failed for %s: %s", symbol, e, exc_info=True
        )
        raise HTTPException(
            status_code=500, detail=f"Factor summary failed: {e}"
        )


@router.post("/ic", response_model=ICResult)
async def compute_ic(
    req: ICRequest,
    consumer: ApiConsumer = Depends(get_api_consumer),
):
    """Compute Information Coefficient (IC) and ICIR for a factor universe.

    Evaluates the predictive power of Alpha158 factors by computing
    rank IC (Spearman correlation) between factor values and forward returns.

    Requires at least 2 symbols. Runs in ThreadPoolExecutor (typical: 5-30s).
    """
    logger.info(
        "POST /factors/ic universe=%d market=%s forward_days=%d",
        len(req.universe), req.market.value, req.forward_days,
    )

    try:
        result = await run_qlib_quick(
            FactorAnalysisService.compute_ic,
            req.universe,
            req.market.value,
            factor_names=req.factor_names if req.factor_names else None,
            start_date=req.start_date.isoformat() if req.start_date else None,
            end_date=req.end_date.isoformat() if req.end_date else None,
            forward_days=req.forward_days,
        )
        return result
    except TimeoutError:
        raise HTTPException(
            status_code=504, detail="IC computation timed out"
        )
    except Exception as e:
        logger.error("IC computation failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"IC computation failed: {e}"
        )


@router.post("/cs-rank", response_model=CSRankResult)
async def compute_cs_rank(
    req: CSRankRequest,
    consumer: ApiConsumer = Depends(get_api_consumer),
):
    """Compute cross-sectional percentile rank for an expression.

    Evaluates a Qlib expression across all provided symbols on a single
    date and returns the percentile rank (0.0 = lowest, 1.0 = highest).

    Requires at least 2 symbols. Runs in ThreadPoolExecutor (typical: 2-10s).
    """
    logger.info(
        "POST /factors/cs-rank expression=%s symbols=%d market=%s",
        req.expression, len(req.symbols), req.market.value,
    )

    try:
        result = await run_qlib_quick(
            FactorAnalysisService.compute_cs_rank,
            req.expression,
            req.symbols,
            req.market.value,
            target_date=req.target_date.isoformat() if req.target_date else None,
        )
        return result
    except TimeoutError:
        raise HTTPException(
            status_code=504, detail="CS rank computation timed out"
        )
    except Exception as e:
        logger.error("CS rank computation failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"CS rank computation failed: {e}"
        )
