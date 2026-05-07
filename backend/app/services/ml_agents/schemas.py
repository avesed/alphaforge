"""Pydantic models for the ML agent system.

Three schemas form the agent pipeline:

1. **DataProfile** -- output of the Profiler agent. Captures market data
   characteristics (NaN rates, sector distribution, return statistics)
   that inform training strategy decisions.

2. **TrainingConfig** -- output of the Strategist agent. A Pydantic
   equivalent of the frozen ``MarketConfig`` dataclass, enriched with
   LLM reasoning.  Numeric fields are clamped to safe ranges via a
   model validator to guard against hallucinated values.

3. **EvaluationResult** -- output of the Evaluator agent. Decides
   whether a trained model should be deployed, retried with adjustments,
   or rejected outright.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from app.services.market_config import MarketConfig


# ---------------------------------------------------------------------------
# 1. Profiler output
# ---------------------------------------------------------------------------


class DataProfile(BaseModel):
    """Market data profile produced by the Profiler agent."""

    market: str
    universe_size: int
    n_trading_days: int
    date_range: tuple[str, str]

    # Feature quality
    feature_nan_rates: dict[str, float]
    median_nan_rate: float
    sparse_features: list[str]  # NaN > 70%

    # Universe composition
    sector_distribution: dict[str, int]  # sector -> stock count
    min_sector_size: int

    # Return distribution
    return_stats: dict[str, float]  # mean, std, skew, kurt

    # Optional context
    recent_model_ics: list[float] | None = None
    regime_analysis: str = ""
    data_quality_warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 2. Strategist output
# ---------------------------------------------------------------------------


# Clamp ranges -- (min, max) inclusive.
_CLAMP_RANGES: dict[str, tuple[float, float]] = {
    "nan_threshold": (0.5, 0.95),
    "ffill_limit": (15, 120),
    "min_ic_threshold": (0.001, 0.05),
    "min_icir_threshold": (0.01, 0.50),
    "num_boost_round": (100, 5000),
    "early_stopping_rounds": (20, 500),
    "confidence": (0.0, 1.0),
}


class TrainingConfig(BaseModel):
    """Training configuration produced by the Strategist agent.

    Mirrors every field of ``MarketConfig`` so it can be converted back
    via ``to_market_config()`` for consumption by existing training code.
    """

    # -- Training mode --
    use_temporal_sort: bool
    # -- Label engineering --
    use_sector_neutral_labels: bool
    use_balanced_quintiles: bool
    # -- Feature engineering --
    use_sector_rank: bool
    use_interactions: bool
    nan_threshold: float  # [0.5, 0.95]
    ffill_limit: int  # [15, 120]
    # -- Quality gate --
    min_ic_threshold: float  # [0.001, 0.05]
    min_icir_threshold: float  # [0.01, 0.50]
    # -- LightGBM hyperparameters --
    lgb_overrides: dict[str, Any] = Field(default_factory=dict)
    num_boost_round: int  # [100, 5000]
    early_stopping_rounds: int  # [20, 500]
    # -- LLM metadata --
    reasoning: str = ""

    @model_validator(mode="after")
    def clamp_values(self) -> TrainingConfig:
        """Clamp numeric fields to safe ranges.

        LLMs occasionally hallucinate extreme values (e.g.
        ``num_boost_round=999999``).  This validator silently clamps
        every numeric field to its documented safe range.
        """
        for field_name, (lo, hi) in _CLAMP_RANGES.items():
            if field_name == "confidence":
                # confidence belongs to EvaluationResult, not here
                continue
            raw = getattr(self, field_name)
            clamped = type(raw)(max(lo, min(hi, raw)))
            if clamped != raw:
                object.__setattr__(self, field_name, clamped)
        return self

    def to_market_config(self) -> MarketConfig:
        """Convert to a frozen ``MarketConfig`` for existing code paths."""
        return MarketConfig(
            use_temporal_sort=self.use_temporal_sort,
            use_sector_neutral_labels=self.use_sector_neutral_labels,
            use_balanced_quintiles=self.use_balanced_quintiles,
            use_sector_rank=self.use_sector_rank,
            use_interactions=self.use_interactions,
            nan_threshold=self.nan_threshold,
            ffill_limit=self.ffill_limit,
            min_ic_threshold=self.min_ic_threshold,
            min_icir_threshold=self.min_icir_threshold,
            lgb_overrides=dict(self.lgb_overrides),
            num_boost_round=self.num_boost_round,
            early_stopping_rounds=self.early_stopping_rounds,
        )


# ---------------------------------------------------------------------------
# 3. Evaluator output
# ---------------------------------------------------------------------------


class EvaluationResult(BaseModel):
    """Model evaluation produced by the Evaluator agent."""

    decision: Literal["deploy", "retry", "reject"]
    reasoning: str
    suggested_adjustments: dict[str, Any] | None = None  # for retry
    confidence: float  # 0-1

    @model_validator(mode="after")
    def clamp_confidence(self) -> EvaluationResult:
        """Ensure confidence stays within [0, 1]."""
        if self.confidence < 0.0:
            object.__setattr__(self, "confidence", 0.0)
        elif self.confidence > 1.0:
            object.__setattr__(self, "confidence", 1.0)
        return self
