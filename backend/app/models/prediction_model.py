from datetime import date, datetime, timezone
from typing import Any

from sqlalchemy import Date, DateTime, Float, Integer, String
from sqlalchemy.dialects.postgresql import JSON
from sqlalchemy.orm import Mapped, mapped_column

from app.core.orm import Base


class PredictionModel(Base):
    __tablename__ = "prediction_models"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    market: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    model_date: Mapped[date] = mapped_column(Date, nullable=False)
    model_type: Mapped[str] = mapped_column(String(20), nullable=False, default="lgbm_rank")
    forward_days: Mapped[int] = mapped_column(Integer, nullable=False, default=5)
    ic: Mapped[float | None] = mapped_column(Float, nullable=True)
    icir: Mapped[float | None] = mapped_column(Float, nullable=True)
    ndcg: Mapped[float | None] = mapped_column(Float, nullable=True)
    fold_ics: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    best_iterations: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    feature_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    symbol_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ensemble_size: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    quality: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
    training_config: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    feature_importance: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    file_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )
