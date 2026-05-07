"""Pydantic schemas for prediction-related API requests and responses.

These models define the wire format for ML prediction workflows
including training runs, inference results, task tracking, and
universe management.
"""

from datetime import date
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field


class PredictionRunRequest(BaseModel):
    """Request to trigger a prediction run."""

    force_retrain: bool = Field(
        False,
        description="Force model retraining even if a recent model exists",
    )
    forward_days: int = Field(
        5,
        ge=1,
        le=60,
        description="Number of trading days to predict forward",
    )


class PredictionResult(BaseModel):
    """Single stock prediction result."""

    symbol: str
    predicted_score: float = Field(
        description="Raw model prediction score (higher = stronger buy signal)",
    )
    percentile_rank: float = Field(
        ge=0.0,
        le=1.0,
        description="Percentile rank within the universe (0.0-1.0, where 1.0 is the highest)",
    )
    predicted_direction: str = Field(
        description="Predicted direction: up, down, or sideways",
    )


class PredictionRunResponse(BaseModel):
    """Response after triggering a prediction run."""

    task_id: str
    market: str
    status: str = Field(description="Initial status: pending")


class PredictionTaskStatus(BaseModel):
    """Status of a long-running prediction task."""

    task_id: str
    status: str = Field(
        description="Task status: pending, training, predicting, completed, or failed",
    )
    progress: Optional[float] = Field(
        None,
        ge=0.0,
        le=100.0,
        description="Progress percentage (0-100)",
    )
    message: Optional[str] = Field(
        None,
        description="Human-readable status message",
    )


class ModelInfo(BaseModel):
    """Metadata for a trained prediction model."""

    id: UUID
    market: str
    model_date: date = Field(description="Date the model was trained on")
    feature_count: int = Field(ge=0, description="Number of input features")
    symbol_count: int = Field(ge=0, description="Number of symbols in training universe")
    ic: Optional[float] = Field(None, description="Information Coefficient")
    icir: Optional[float] = Field(None, description="IC Information Ratio")
    ndcg: Optional[float] = Field(None, description="Normalized Discounted Cumulative Gain")


class UniverseRequest(BaseModel):
    """Request to create or update a prediction universe."""

    name: str = Field(max_length=100, description="Universe display name")
    market: str = Field(max_length=10, description="Market code: us, hk, cn")
    universe_type: str = Field(
        "custom",
        description="Universe type: 'index' (track an index) or 'custom' (explicit symbols)",
    )
    index_code: Optional[str] = Field(
        None,
        max_length=20,
        description="Index code for 'index' type (e.g., SPY, HSI, CSI300)",
    )
    symbols: Optional[list[str]] = Field(
        None,
        description="Explicit symbol list for 'custom' type",
    )


class ModelQualityUpdateRequest(BaseModel):
    """Request to update model quality status."""

    quality_passed: bool = Field(description="Whether the model passed quality gate")


class RDAgentStartRequest(BaseModel):
    """Request to start an RD-Agent research loop."""

    universe_id: Optional[UUID] = Field(
        None,
        description="Target universe ID (uses default if not specified)",
    )
    max_rounds: int = Field(
        30,
        ge=1,
        le=200,
        description="Maximum number of research rounds",
    )
