"""Request/response schemas for ML backtest API."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class BacktestRequest(BaseModel):
    """POST /predictions/{market}/backtest request body."""

    cutoff_date: date
    validation_days: int = Field(default=60, ge=10, le=250)
    forward_days: int = Field(default=5, ge=1, le=30)
    config_override: dict[str, Any] | None = None
    use_llm_agents: bool = False
    max_iterations: int = Field(default=3, ge=1, le=10)
    backtest_type: Literal["static", "rolling"] = "static"
    retrain_interval: int = Field(default=5, ge=1, le=20)


class BacktestStartResponse(BaseModel):
    """Response from POST /predictions/{market}/backtest."""

    task_id: str
    backtest_id: str
    market: str
    status: str = "pending"


class BacktestTaskStatus(BaseModel):
    """Response from GET /predictions/backtests/tasks/{task_id}."""

    task_id: str
    backtest_id: str
    market: str
    status: str  # pending/running/completed/failed
    progress: float = 0.0
    message: str = ""
    current_phase: str = ""  # profiling/training/inference/evaluating
    current_iteration: int = 0
    max_iterations: int = 1
    iterations: list[dict[str, Any]] = Field(default_factory=list)
    current_retrain: int | None = None
    total_retrains: int | None = None
    elapsed_seconds: float = 0.0
    created_at: datetime | None = None
    completed_at: datetime | None = None


class BacktestSummary(BaseModel):
    """Summary for backtest list responses."""

    id: str
    market: str
    cutoff_date: date
    validation_days: int
    forward_days: int
    status: str
    train_ic: float | None = None
    train_icir: float | None = None
    val_ic: float | None = None
    val_icir: float | None = None
    val_direction_accuracy: float | None = None
    val_spread: float | None = None
    agent_iteration: int | None = None
    duration_seconds: float | None = None
    created_at: datetime | None = None
    completed_at: datetime | None = None


class BacktestDetail(BacktestSummary):
    """Full backtest detail response."""

    config_override: dict[str, Any] | None = None
    effective_config: dict[str, Any] = Field(default_factory=dict)
    train_ndcg: float | None = None
    fold_ics: list[float] | None = None
    ensemble_size: int | None = None
    feature_count: int | None = None
    symbol_count: int | None = None
    val_q1_return: float | None = None
    val_q5_return: float | None = None
    val_hit_rate: float | None = None
    val_max_drawdown: float | None = None
    results: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    agent_run_id: str | None = None


class BacktestListResponse(BaseModel):
    """Response from GET /predictions/{market}/backtests."""

    backtests: list[BacktestSummary]
    total: int
