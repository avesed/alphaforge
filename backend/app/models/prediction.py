from datetime import date, datetime, timezone

from sqlalchemy import (
    BigInteger,
    Date,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.core.orm import Base


class StockPrediction(Base):
    __tablename__ = "stock_predictions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    market: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    prediction_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    forward_days: Mapped[int] = mapped_column(Integer, nullable=False, default=5)
    rank_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    percentile_rank: Mapped[float | None] = mapped_column(Float, nullable=True)
    up_probability: Mapped[float | None] = mapped_column(Float, nullable=True)
    actual_return: Mapped[float | None] = mapped_column(Float, nullable=True)
    model_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )

    __table_args__ = (
        Index("ix_pred_market_date", "market", "prediction_date"),
        UniqueConstraint(
            "market", "symbol", "prediction_date", "forward_days",
            name="uq_prediction",
        ),
    )
