"""Pydantic schemas for Qlib-related API endpoints."""
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# === Enums ===


class MarketCode(str, Enum):
    US = "us"
    HK = "hk"
    SH = "sh"
    SZ = "sz"
    CN = "cn"
    METAL = "metal"


class AlphaType(str, Enum):
    ALPHA158 = "alpha158"
    ALPHA360 = "alpha360"


class StrategyType(str, Enum):
    TOPK = "topk"
    SIGNAL = "signal"
    LONG_SHORT = "long_short"


class BacktestStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SyncStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# === Expression Engine ===


class ExpressionEvaluateRequest(BaseModel):
    symbol: str = Field(
        ..., max_length=20, pattern=r"^[A-Za-z0-9.=\-]+$",
        description="Stock symbol (e.g., AAPL, 600000.SS)",
    )
    expression: str = Field(
        ..., max_length=500,
        description="Qlib expression (e.g., Corr($close, $volume, 20))",
    )
    market: MarketCode = MarketCode.US
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    period: str = Field("3mo", description="Fallback period if dates not specified (1mo, 3mo, 6mo, 1y, 2y, 5y)")


class ExpressionBatchRequest(BaseModel):
    symbols: List[str] = Field(..., min_length=1, max_length=500)
    expression: str = Field(..., max_length=500)
    market: MarketCode = MarketCode.US
    target_date: Optional[date] = None


class ExpressionValidateRequest(BaseModel):
    expression: str = Field(..., max_length=500)


class ExpressionResult(BaseModel):
    symbol: str
    expression: str
    series: List[Dict[str, Any]] = Field(
        default_factory=list, description="[{date, value}]"
    )
    latest_value: Optional[float] = None
    count: int = 0


class ExpressionBatchResult(BaseModel):
    expression: str
    results: Dict[str, Optional[float]] = Field(
        default_factory=dict, description="{symbol: value}"
    )
    date: Optional[str] = None


class ValidationResult(BaseModel):
    valid: bool
    error: Optional[str] = None
    operators_used: List[str] = Field(default_factory=list)


# === Factors ===


class FactorRequest(BaseModel):
    symbol: str
    market: MarketCode = MarketCode.US
    alpha_type: AlphaType = AlphaType.ALPHA158
    start_date: Optional[date] = None
    end_date: Optional[date] = None


class FactorResult(BaseModel):
    symbol: str
    market: str
    alpha_type: str
    mode: str = Field(
        description="'single' (time-series only) or 'market' (with CS features)"
    )
    factor_count: int = 0
    dates: List[str] = Field(default_factory=list)
    factors: Dict[str, List[Optional[float]]] = Field(default_factory=dict)
    top_factors: List[Dict[str, Any]] = Field(
        default_factory=list, description="Top factors by absolute z-score"
    )


class FactorSummary(BaseModel):
    symbol: str
    market: str
    latest_date: Optional[str] = None
    top_factors: List[Dict[str, Any]] = Field(default_factory=list)
    mode: str = "single"


class ICRequest(BaseModel):
    universe: List[str] = Field(..., min_length=2)
    factor_names: List[str] = Field(
        default_factory=list, description="Empty = all Alpha158"
    )
    market: MarketCode = MarketCode.US
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    forward_days: int = Field(5, ge=1, le=20)


class ICResult(BaseModel):
    factor_ic: Dict[str, float] = Field(default_factory=dict)
    factor_icir: Dict[str, float] = Field(default_factory=dict)
    ic_series: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)


class CSRankRequest(BaseModel):
    expression: str
    symbols: List[str] = Field(..., min_length=2)
    market: MarketCode = MarketCode.US
    target_date: Optional[date] = None


class CSRankResult(BaseModel):
    expression: str
    date: Optional[str] = None
    rankings: Dict[str, float] = Field(
        default_factory=dict, description="{symbol: percentile_rank}"
    )


# === Backtests ===


class BacktestCreateRequest(BaseModel):
    name: str = Field(..., max_length=200, description="Human-readable name for this backtest")
    market: MarketCode = MarketCode.US
    symbols: List[str] = Field(..., min_length=1, max_length=500, description="Stock symbols to include")
    start_date: date = Field(..., description="Backtest start date (YYYY-MM-DD)")
    end_date: date = Field(..., description="Backtest end date (YYYY-MM-DD)")
    strategy_type: StrategyType = StrategyType.TOPK
    strategy_config: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Strategy-specific config. "
            "TopK: {k, n_drop, score_expression, rebalance_days}. "
            "Signal: {expression, buy_threshold, sell_threshold, max_positions, rebalance_days}. "
            "LongShort: {score_expression, long_pct, short_pct, rebalance_days}."
        ),
    )
    execution_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            "slippage": 0.0005,
            "commission": 0.0015,
            "limit_threshold": None,  # None = no limit, 0.095 = A-share 10%
        },
        description="Execution parameters: slippage, commission, limit_threshold",
    )


class BacktestStatusResponse(BaseModel):
    task_id: str
    name: Optional[str] = None
    status: BacktestStatus = BacktestStatus.PENDING
    progress: int = Field(0, ge=0, le=100)
    current_date: Optional[str] = None
    current_return: Optional[float] = None
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class BacktestListResponse(BaseModel):
    tasks: List[BacktestStatusResponse] = Field(default_factory=list)
    total: int = 0


# === Data Sync ===


class SyncRequest(BaseModel):
    market: MarketCode = MarketCode.US
    symbols: Optional[List[str]] = None  # None = full market
    start_date: Optional[date] = None
    update_only: bool = True  # Only download new data


class SyncStatusResponse(BaseModel):
    markets: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Per-market status: {market: {last_sync, symbol_count, date_range, status}}",
    )


# === Common ===


class ErrorResponse(BaseModel):
    detail: str
    code: Optional[str] = None
