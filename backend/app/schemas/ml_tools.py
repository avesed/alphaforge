"""Schemas for ML Tools API endpoints.

These endpoints decompose the monolithic backtest loop into independent,
callable steps for consumption by the ML Agent in the backend container.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------


class ProfileRequest(BaseModel):
    """POST /ml-tools/profile request body."""

    market: str = Field(..., pattern=r"^(us|cn|hk)$")
    cutoff_date: date
    validation_days: int = Field(60, ge=10, le=250)
    forward_days: int = Field(5, ge=1, le=30)


class ProfileResponse(BaseModel):
    """Profile result: data statistics + current MarketConfig baseline."""

    market: str
    universe_size: int
    n_trading_days: int
    date_range: List[str]  # [start, end]
    feature_nan_rates: Dict[str, float]
    median_nan_rate: float
    sparse_features: List[str]
    sector_distribution: Dict[str, int]
    min_sector_size: int
    return_stats: Dict[str, float]
    baseline_config: Dict[str, Any]  # Current MarketConfig as dict


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------


class TrainRequest(BaseModel):
    """POST /ml-tools/train request body."""

    market: str = Field(..., pattern=r"^(us|cn|hk)$")
    cutoff_date: date
    forward_days: int = Field(5, ge=1, le=30)
    config: Dict[str, Any]  # Full MarketConfig fields (applied as override)


class TrainResponse(BaseModel):
    """Immediate response from POST /ml-tools/train."""

    task_id: str
    status: str = "submitted"


class TrainTaskStatus(BaseModel):
    """Response from GET /ml-tools/tasks/{task_id}."""

    task_id: str
    status: str  # submitted, training, completed, failed
    progress: float = 0.0
    status_detail: str = ""  # human-readable sub-step description
    result: Optional[Dict[str, Any]] = None  # ic, icir, fold_ics, etc.
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------


class ValidateRequest(BaseModel):
    """POST /ml-tools/validate request body."""

    task_id: str
    cutoff_date: date
    validation_days: int = Field(60, ge=10, le=250)
    forward_days: int = Field(5, ge=1, le=30)


class ValidateResponse(BaseModel):
    """Validation metrics computed on out-of-sample data."""

    val_ic: float
    val_icir: float
    val_spread: float
    val_direction_accuracy: float
    quintile_returns: Dict[str, float]
    val_hit_rate: float
    ic_curve: List[float]
    val_max_drawdown: Optional[float] = None


# ---------------------------------------------------------------------------
# Rolling Backtest
# ---------------------------------------------------------------------------


class RollingBacktestRequest(BaseModel):
    """POST /ml-tools/rolling-backtest request body."""

    market: str = Field(..., pattern=r"^(us|cn|hk)$")
    cutoff_date: date
    validation_days: int = Field(60, ge=10, le=250)
    forward_days: int = Field(5, ge=1, le=30)
    retrain_interval: int = Field(5, ge=1, le=20)
    config: Dict[str, Any]  # Config override (applied to MarketConfig baseline)


class RollingBacktestResponse(BaseModel):
    """Immediate response from POST /ml-tools/rolling-backtest."""

    task_id: str
    status: str = "submitted"


# ---------------------------------------------------------------------------
# Deploy
# ---------------------------------------------------------------------------


class DeployRequest(BaseModel):
    """POST /ml-tools/deploy request body."""

    market: str = Field(..., pattern=r"^(us|cn|hk)$")
    backtest_id: str
    effective_config: Dict[str, Any]
    iteration: int = Field(1, ge=1)
    val_ic: float
    train_ic: Optional[float] = None
    train_icir: Optional[float] = None


class DeployResponse(BaseModel):
    """Response from POST /ml-tools/deploy."""

    status: str = "ok"
