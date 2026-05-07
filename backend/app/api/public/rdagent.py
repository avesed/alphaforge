"""RD-Agent API endpoints (X-API-Key auth).

Controls the RD-Agent automated factor research workflow:
start/stop research loops, monitor progress, and manage
discovered factors (list, enable/disable).
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from app.core.auth import get_api_consumer
from app.models.api_consumer import ApiConsumer
from app.schemas.prediction import RDAgentStartRequest
from app.services.factor_registry import factor_registry
from app.services.rdagent_runner import rdagent_runner

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rdagent", tags=["rdagent"])


class FactorToggleRequest(BaseModel):
    is_active: bool


@router.post("/{market}/start")
async def start_rdagent(
    market: str,
    request: RDAgentStartRequest = RDAgentStartRequest(),
    consumer: ApiConsumer = Depends(get_api_consumer),
):
    """Start an RD-Agent research loop for the given market.

    Launches the automated factor discovery process which iteratively
    proposes, implements, and evaluates new quantitative factors.
    Only one loop per market can run concurrently (Redis lock enforced).
    """
    market = market.lower()
    if market not in ("us", "hk", "cn"):
        raise HTTPException(400, f"Unsupported market: {market}")

    result = await rdagent_runner.start(
        market=market,
        universe_id=str(request.universe_id) if request.universe_id else None,
        max_rounds=request.max_rounds,
    )

    if "error" in result:
        raise HTTPException(409, result["error"])

    return result


@router.get("/{market}/status")
async def get_rdagent_status(
    market: str,
    consumer: ApiConsumer = Depends(get_api_consumer),
):
    """Get the current status of an RD-Agent research loop.

    Returns round progress, discovered factor count, and
    log tail for the active or most recent session.
    """
    market = market.lower()
    return await rdagent_runner.get_status(market)


@router.post("/{market}/stop")
async def stop_rdagent(
    market: str,
    consumer: ApiConsumer = Depends(get_api_consumer),
):
    """Stop a running RD-Agent research loop.

    Gracefully terminates the current research round and
    persists any discovered factors.
    """
    market = market.lower()
    result = await rdagent_runner.stop(market)

    if "error" in result:
        raise HTTPException(404, result["error"])

    return result


@router.get("/factors")
async def list_factors(
    market: Optional[str] = Query(None, description="Filter by market"),
    consumer: ApiConsumer = Depends(get_api_consumer),
):
    """List all factors discovered by RD-Agent.

    Returns factor definitions with their IC/ICIR metrics
    and activation status.
    """
    factors = await factor_registry.get_all_factors(market=market)
    return {"count": len(factors), "factors": factors}


@router.put("/factors/{factor_id}")
async def update_factor(
    factor_id: str,
    request: FactorToggleRequest,
    consumer: ApiConsumer = Depends(get_api_consumer),
):
    """Update a discovered factor's activation status.

    Used by admins to enable/disable factors for inclusion
    in the production prediction pipeline. Active factors are
    automatically included in the feature matrix during
    the next LightGBM training run.
    """
    success = await factor_registry.toggle_factor(factor_id, request.is_active)
    if not success:
        raise HTTPException(404, f"Factor not found: {factor_id}")

    return {"id": factor_id, "is_active": request.is_active}
