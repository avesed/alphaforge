"""Local database storage for predictions, models, and discovered factors.

Replaces StockPulse API calls for ML data write/read operations.
Uses SQLAlchemy async ORM with the session factory from app.core.orm.

All methods are async and safe for concurrent use -- each call creates
its own session via get_session_factory().
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any

from sqlalchemy import and_, delete, desc, func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.core.orm import get_session_factory
from app.models.discovered_factor import DiscoveredFactor
from app.models.prediction import StockPrediction
from app.models.prediction_model import PredictionModel

logger = logging.getLogger(__name__)


def _date_from_value(v: Any) -> date | None:
    """Coerce a string, datetime, or date to a date object."""
    if v is None:
        return None
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    if isinstance(v, str):
        return date.fromisoformat(v)
    return v


def _prediction_to_dict(row: StockPrediction) -> dict[str, Any]:
    """Convert a StockPrediction ORM instance to a plain dict."""
    return {
        "id": row.id,
        "market": row.market,
        "symbol": row.symbol,
        "prediction_date": row.prediction_date.isoformat() if row.prediction_date else None,
        "forward_days": row.forward_days,
        "rank_score": row.rank_score,
        "percentile_rank": row.percentile_rank,
        "up_probability": row.up_probability,
        "actual_return": row.actual_return,
        "model_id": row.model_id,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


def _model_to_dict(row: PredictionModel) -> dict[str, Any]:
    """Convert a PredictionModel ORM instance to a plain dict."""
    return {
        "id": row.id,
        "market": row.market,
        "model_date": row.model_date.isoformat() if row.model_date else None,
        "model_type": row.model_type,
        "forward_days": row.forward_days,
        "ic": row.ic,
        "icir": row.icir,
        "ndcg": row.ndcg,
        "fold_ics": row.fold_ics,
        "best_iterations": row.best_iterations,
        "feature_count": row.feature_count,
        "symbol_count": row.symbol_count,
        "ensemble_size": row.ensemble_size,
        "quality": row.quality,
        "training_config": row.training_config,
        "feature_importance": row.feature_importance,
        "file_path": row.file_path,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


def _factor_to_dict(row: DiscoveredFactor) -> dict[str, Any]:
    """Convert a DiscoveredFactor ORM instance to a plain dict."""
    return {
        "id": row.id,
        "market": row.market,
        "name": row.name,
        "expression": row.expression,
        "ic": row.ic,
        "icir": row.icir,
        "source": row.source,
        "is_active": row.is_active,
        "metadata": row.metadata_,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
    }


class PredictionStore:
    """Local database storage for predictions, models, and factors.

    Uses get_session_factory() from app.core.orm for all DB access.
    Bulk operations use PostgreSQL ON CONFLICT for upserts.
    """

    # ------------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------------

    async def write_predictions(self, predictions: list[dict]) -> int:
        """Batch upsert predictions. Returns count written.

        Each dict should contain: market, symbol, prediction_date,
        forward_days, rank_score, percentile_rank, model_id.
        Optional: up_probability, actual_return.

        Uses ON CONFLICT (market, symbol, prediction_date, forward_days)
        DO UPDATE to overwrite scores when re-running predictions.
        """
        if not predictions:
            return 0

        factory = get_session_factory()
        written = 0

        async with factory() as session:
            try:
                # Process in batches of 500 to avoid excessive memory use
                # Deduplicate by unique key before insert
                seen: set[tuple] = set()
                deduped: list[dict] = []
                for p in predictions:
                    key = (
                        p["market"], p["symbol"],
                        str(p["prediction_date"])[:10],
                        p.get("forward_days", 5),
                    )
                    if key not in seen:
                        seen.add(key)
                        deduped.append(p)

                batch_size = 500
                for i in range(0, len(deduped), batch_size):
                    batch = deduped[i : i + batch_size]
                    values = []
                    for p in batch:
                        values.append({
                            "market": p["market"],
                            "symbol": p["symbol"],
                            "prediction_date": _date_from_value(p["prediction_date"]),
                            "forward_days": p.get("forward_days", 5),
                            "rank_score": p.get("rank_score"),
                            "percentile_rank": p.get("percentile_rank"),
                            "up_probability": p.get("up_probability"),
                            "actual_return": p.get("actual_return"),
                            "model_id": p.get("model_id"),
                        })

                    stmt = pg_insert(StockPrediction).values(values)
                    stmt = stmt.on_conflict_do_update(
                        constraint="uq_prediction",
                        set_={
                            "rank_score": stmt.excluded.rank_score,
                            "percentile_rank": stmt.excluded.percentile_rank,
                            "up_probability": stmt.excluded.up_probability,
                            "model_id": stmt.excluded.model_id,
                        },
                    )
                    result = await session.execute(stmt)
                    written += result.rowcount

                await session.commit()
                logger.info("Wrote %d predictions (input=%d)", written, len(predictions))
            except Exception:
                await session.rollback()
                raise

        return written

    async def get_latest_predictions(
        self,
        market: str,
        top_n: int = 100,
        symbol: str | None = None,
        forward_days: int | None = None,
    ) -> list[dict]:
        """Get latest predictions for a market, sorted by rank_score desc.

        Finds the most recent prediction_date for the given market,
        then returns up to top_n predictions from that date.
        Filters by forward_days and/or symbol when provided.
        """
        factory = get_session_factory()

        async with factory() as session:
            # Find latest prediction date for this market
            date_filters = [StockPrediction.market == market]
            if forward_days is not None:
                date_filters.append(StockPrediction.forward_days == forward_days)

            latest_date_stmt = (
                select(func.max(StockPrediction.prediction_date))
                .where(and_(*date_filters))
            )
            result = await session.execute(latest_date_stmt)
            latest_date = result.scalar()

            if latest_date is None:
                return []

            filters = [
                StockPrediction.market == market,
                StockPrediction.prediction_date == latest_date,
            ]
            if forward_days is not None:
                filters.append(StockPrediction.forward_days == forward_days)
            if symbol is not None:
                filters.append(StockPrediction.symbol == symbol)

            stmt = (
                select(StockPrediction)
                .where(and_(*filters))
                .order_by(desc(StockPrediction.rank_score))
                .limit(top_n)
            )
            result = await session.execute(stmt)
            rows = result.scalars().all()

        return [_prediction_to_dict(r) for r in rows]

    async def get_prediction_history(
        self, market: str, days: int = 30
    ) -> list[dict]:
        """Get prediction dates and counts for recent days.

        Returns a list of dicts with prediction_date and count,
        ordered by date descending.
        """
        factory = get_session_factory()
        cutoff = date.today() - timedelta(days=days)

        async with factory() as session:
            stmt = (
                select(
                    StockPrediction.prediction_date,
                    func.count(StockPrediction.id).label("count"),
                )
                .where(
                    and_(
                        StockPrediction.market == market,
                        StockPrediction.prediction_date >= cutoff,
                    )
                )
                .group_by(StockPrediction.prediction_date)
                .order_by(desc(StockPrediction.prediction_date))
            )
            result = await session.execute(stmt)
            rows = result.all()

        return [
            {
                "prediction_date": row.prediction_date.isoformat(),
                "count": row.count,
            }
            for row in rows
        ]

    async def get_predictions_for_backfill(
        self, market: str, days: int = 30
    ) -> list[dict]:
        """Get predictions where actual_return is NULL and date is old enough.

        Returns predictions whose prediction_date is at least forward_days
        old (so actual returns are available) but not older than `days`.
        """
        factory = get_session_factory()
        today = date.today()
        cutoff = today - timedelta(days=days)

        async with factory() as session:
            # Fetch candidates with NULL actual_return, then filter in Python
            # for the date+forward_days eligibility check (avoids complex
            # PostgreSQL interval arithmetic with a dynamic column).
            stmt = (
                select(StockPrediction)
                .where(
                    and_(
                        StockPrediction.market == market,
                        StockPrediction.actual_return.is_(None),
                        StockPrediction.prediction_date >= cutoff,
                    )
                )
                .order_by(StockPrediction.prediction_date)
            )
            result = await session.execute(stmt)
            rows = result.scalars().all()

        # Filter in Python: only include rows where enough time has passed
        eligible = []
        for r in rows:
            if r.prediction_date + timedelta(days=r.forward_days) <= today:
                eligible.append(_prediction_to_dict(r))

        return eligible

    async def backfill_returns(self, updates: list[dict]) -> int:
        """Update actual_return for existing predictions. Returns count updated.

        Each dict must contain: market, symbol, prediction_date,
        forward_days, actual_return.
        """
        if not updates:
            return 0

        factory = get_session_factory()
        updated = 0

        async with factory() as session:
            try:
                for u in updates:
                    stmt = (
                        update(StockPrediction)
                        .where(
                            and_(
                                StockPrediction.market == u["market"],
                                StockPrediction.symbol == u["symbol"],
                                StockPrediction.prediction_date == _date_from_value(u["prediction_date"]),
                                StockPrediction.forward_days == u.get("forward_days", 5),
                            )
                        )
                        .values(actual_return=u["actual_return"])
                    )
                    result = await session.execute(stmt)
                    updated += result.rowcount

                await session.commit()
                logger.info("Backfilled returns for %d predictions", updated)
            except Exception:
                await session.rollback()
                raise

        return updated

    async def update_up_probabilities(
        self,
        market: str,
        prediction_date: str,
        forward_days: int,
        updates: list[dict],
    ) -> int:
        """Batch update up_probability on existing predictions.

        Each dict in updates must contain: symbol, up_probability.
        """
        if not updates:
            return 0

        factory = get_session_factory()
        pred_date = _date_from_value(prediction_date)
        updated = 0

        async with factory() as session:
            try:
                for u in updates:
                    stmt = (
                        update(StockPrediction)
                        .where(
                            and_(
                                StockPrediction.market == market,
                                StockPrediction.symbol == u["symbol"],
                                StockPrediction.prediction_date == pred_date,
                                StockPrediction.forward_days == forward_days,
                            )
                        )
                        .values(up_probability=u["up_probability"])
                    )
                    result = await session.execute(stmt)
                    updated += result.rowcount

                await session.commit()
                logger.info(
                    "Updated up_probability for %d predictions (market=%s, date=%s)",
                    updated, market, prediction_date,
                )
            except Exception:
                await session.rollback()
                raise

        return updated

    async def get_performance_metrics(
        self, market: str, days: int = 90
    ) -> list[dict]:
        """Get predictions with actual_return filled (for accuracy/performance calcs).

        Returns predictions from the last `days` days where actual_return
        is not NULL, ordered by prediction_date desc, rank_score desc.
        """
        factory = get_session_factory()
        cutoff = date.today() - timedelta(days=days)

        async with factory() as session:
            stmt = (
                select(StockPrediction)
                .where(
                    and_(
                        StockPrediction.market == market,
                        StockPrediction.actual_return.isnot(None),
                        StockPrediction.prediction_date >= cutoff,
                    )
                )
                .order_by(
                    desc(StockPrediction.prediction_date),
                    desc(StockPrediction.rank_score),
                )
            )
            result = await session.execute(stmt)
            rows = result.scalars().all()

        return [_prediction_to_dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------

    async def write_model(self, model_data: dict) -> str:
        """Insert or update model metadata. Returns model_id.

        model_data must contain 'id' (the model hash). All other fields
        are optional and will be set from the dict.
        """
        factory = get_session_factory()
        model_id = model_data["id"]

        values = {
            "id": model_id,
            "market": model_data.get("market", ""),
            "model_date": _date_from_value(model_data.get("model_date")),
            "model_type": model_data.get("model_type", "lgbm_rank"),
            "forward_days": model_data.get("forward_days", 5),
            "ic": model_data.get("ic"),
            "icir": model_data.get("icir"),
            "ndcg": model_data.get("ndcg"),
            "fold_ics": model_data.get("fold_ics"),
            "best_iterations": model_data.get("best_iterations"),
            "feature_count": model_data.get("feature_count"),
            "symbol_count": model_data.get("symbol_count"),
            "ensemble_size": model_data.get("ensemble_size", 1),
            "quality": model_data.get("quality", "pending"),
            "training_config": model_data.get("training_config"),
            "feature_importance": model_data.get("feature_importance"),
            "file_path": model_data.get("file_path"),
        }

        async with factory() as session:
            try:
                stmt = pg_insert(PredictionModel).values(values)
                stmt = stmt.on_conflict_do_update(
                    index_elements=["id"],
                    set_={
                        "ic": stmt.excluded.ic,
                        "icir": stmt.excluded.icir,
                        "ndcg": stmt.excluded.ndcg,
                        "fold_ics": stmt.excluded.fold_ics,
                        "best_iterations": stmt.excluded.best_iterations,
                        "feature_count": stmt.excluded.feature_count,
                        "symbol_count": stmt.excluded.symbol_count,
                        "ensemble_size": stmt.excluded.ensemble_size,
                        "quality": stmt.excluded.quality,
                        "training_config": stmt.excluded.training_config,
                        "feature_importance": stmt.excluded.feature_importance,
                        "file_path": stmt.excluded.file_path,
                    },
                )
                await session.execute(stmt)
                await session.commit()
                logger.info("Wrote model: id=%s, market=%s", model_id, values["market"])
            except Exception:
                await session.rollback()
                raise

        return model_id

    async def get_models(self, market: str | None = None) -> list[dict]:
        """List models, optionally filtered by market.

        Returns models ordered by created_at descending.
        """
        factory = get_session_factory()

        async with factory() as session:
            stmt = select(PredictionModel).order_by(desc(PredictionModel.created_at))
            if market is not None:
                stmt = stmt.where(PredictionModel.market == market)

            result = await session.execute(stmt)
            rows = result.scalars().all()

        return [_model_to_dict(r) for r in rows]

    async def update_model_quality(self, model_id: str, quality: str) -> None:
        """Update model quality status (pending/approved/rejected)."""
        factory = get_session_factory()

        async with factory() as session:
            try:
                stmt = (
                    update(PredictionModel)
                    .where(PredictionModel.id == model_id)
                    .values(quality=quality)
                )
                result = await session.execute(stmt)
                await session.commit()

                if result.rowcount == 0:
                    logger.warning("Model not found for quality update: %s", model_id)
                else:
                    logger.info("Updated model quality: id=%s, quality=%s", model_id, quality)
            except Exception:
                await session.rollback()
                raise

    # ------------------------------------------------------------------
    # Factors
    # ------------------------------------------------------------------

    async def write_factors(self, factors: list[dict]) -> int:
        """Batch upsert discovered factors. Returns count written.

        Each dict should contain: market, name, expression.
        Optional: ic, icir, source, is_active, metadata.

        Upserts on (market, name) -- if a factor with the same name
        and market exists, its metrics are updated.
        """
        if not factors:
            return 0

        factory = get_session_factory()
        written = 0

        async with factory() as session:
            try:
                for f in factors:
                    # Check if factor with same name+market exists
                    existing_stmt = (
                        select(DiscoveredFactor)
                        .where(
                            and_(
                                DiscoveredFactor.market == f["market"],
                                DiscoveredFactor.name == f["name"],
                            )
                        )
                    )
                    result = await session.execute(existing_stmt)
                    existing = result.scalar_one_or_none()

                    if existing is not None:
                        # Update existing factor
                        update_stmt = (
                            update(DiscoveredFactor)
                            .where(DiscoveredFactor.id == existing.id)
                            .values(
                                expression=f.get("expression", existing.expression),
                                ic=f.get("ic", existing.ic),
                                icir=f.get("icir", existing.icir),
                                source=f.get("source", existing.source),
                                is_active=f.get("is_active", existing.is_active),
                                metadata_=f.get("metadata", existing.metadata_),
                                updated_at=datetime.now(timezone.utc),
                            )
                        )
                        await session.execute(update_stmt)
                    else:
                        # Insert new factor
                        new_factor = DiscoveredFactor(
                            market=f["market"],
                            name=f["name"],
                            expression=f["expression"],
                            ic=f.get("ic"),
                            icir=f.get("icir"),
                            source=f.get("source", "rdagent"),
                            is_active=f.get("is_active", True),
                            metadata_=f.get("metadata"),
                        )
                        session.add(new_factor)

                    written += 1

                await session.commit()
                logger.info("Wrote %d factors", written)
            except Exception:
                await session.rollback()
                raise

        return written

    async def get_factors(
        self, market: str | None = None, active_only: bool = True
    ) -> list[dict]:
        """List discovered factors.

        Args:
            market: If provided, filter to this market only.
            active_only: If True (default), only return active factors.

        Returns:
            List of factor dicts ordered by abs(ic) descending.
        """
        factory = get_session_factory()

        async with factory() as session:
            stmt = select(DiscoveredFactor)

            conditions = []
            if market is not None:
                conditions.append(DiscoveredFactor.market == market)
            if active_only:
                conditions.append(DiscoveredFactor.is_active.is_(True))

            if conditions:
                stmt = stmt.where(and_(*conditions))

            stmt = stmt.order_by(desc(func.abs(DiscoveredFactor.ic)))
            result = await session.execute(stmt)
            rows = result.scalars().all()

        return [_factor_to_dict(r) for r in rows]


# Module singleton
prediction_store = PredictionStore()
