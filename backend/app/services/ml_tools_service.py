"""ML Tools Service -- decomposed backtest steps for the ML Agent.

Exposes five independent operations (profile, train, validate, deploy,
get_task) that the backend ML Agent orchestrates via REST calls.  Each
operation reuses existing compute logic in prediction_service,
feature_service, and ml_backtest_service without LLM dependencies.

Design:
- Module-level singleton: ``ml_tools_service``
- In-memory task tracking via ``_ToolTask`` dataclass
- asyncio.Lock for concurrent task submission guard
- Trained models cached with TTL for validate step

AlphaForge adaptation:
- Uses get_db_pool() (asyncpg pool) instead of settings_cache.pool.
- Direction model validation is optional (stubbed if direction_service
  is not available).
"""

import asyncio
import dataclasses
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Optional

import numpy as np
import pandas as pd

from app.core.database import get_db_pool
from app.services.market_config import (
    MarketConfig,
    apply_override,
    get_market_config,
)
from app.services.prediction_service import prediction_service

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Training data lookback in calendar days (~2 years of trading days)
_TRAIN_LOOKBACK_DAYS = 730

# Maximum concurrent training tasks (per-market)
_MAX_CONCURRENT_TASKS = 1

# TTL for cached trained models (seconds) -- cleaned up after this period
_MODEL_CACHE_TTL = 3600  # 1 hour


# ---------------------------------------------------------------------------
# Task dataclass
# ---------------------------------------------------------------------------


@dataclass
class _ToolTask:
    """In-memory ML tools task with progress tracking."""

    task_id: str
    market: str
    status: str = "submitted"  # submitted, training, completed, failed
    progress: float = 0.0
    status_detail: str = ""  # human-readable sub-step description
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    # Internal -- not exposed in API response
    _models: Optional[list] = field(default=None, repr=False, compare=False)
    _feature_cols: Optional[list[str]] = field(
        default=None, repr=False, compare=False
    )
    _config: Optional[MarketConfig] = field(
        default=None, repr=False, compare=False
    )
    # Direction model cache (populated during training, used by validate)
    _direction_models: Optional[list] = field(default=None, repr=False, compare=False)
    _direction_feature_cols: Optional[list[str]] = field(
        default=None, repr=False, compare=False
    )
    _direction_calibrator: Optional[object] = field(
        default=None, repr=False, compare=False
    )
    _asyncio_task: Optional[asyncio.Task] = field(
        default=None, repr=False, compare=False
    )
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "progress": round(self.progress, 1),
            "status_detail": self.status_detail,
            "result": self.result,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# SQL for deploy
# ---------------------------------------------------------------------------

_SQL_UPSERT_BACKTEST = """
INSERT INTO ml_backtests (
    id, market, cutoff_date, validation_days, forward_days,
    config_override, effective_config, status, results,
    val_ic, train_ic, train_icir,
    agent_run_id, agent_iteration,
    completed_at
) VALUES (
    $1::uuid, $2, $3, $4, $5,
    $6::jsonb, $7::jsonb, $8, $9::jsonb,
    $10, $11, $12,
    $13, $14,
    NOW()
)
ON CONFLICT (id) DO UPDATE SET
    status = EXCLUDED.status,
    config_override = EXCLUDED.config_override,
    effective_config = EXCLUDED.effective_config,
    results = EXCLUDED.results,
    val_ic = EXCLUDED.val_ic,
    train_ic = EXCLUDED.train_ic,
    train_icir = EXCLUDED.train_icir,
    agent_run_id = EXCLUDED.agent_run_id,
    agent_iteration = EXCLUDED.agent_iteration,
    completed_at = EXCLUDED.completed_at
"""


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class MLToolsService:
    """Decomposed ML backtest steps for agent orchestration."""

    def __init__(self) -> None:
        self._tasks: dict[str, _ToolTask] = {}
        self._training_markets: set[str] = set()
        self._submit_lock = asyncio.Lock()

    async def shutdown(self) -> None:
        """Cancel running training tasks during graceful shutdown."""
        for tid, task in self._tasks.items():
            if task._asyncio_task and not task._asyncio_task.done():
                logger.info("ML tools shutdown: cancelling task %s", tid)
                task._asyncio_task.cancel()
        self._tasks.clear()
        self._training_markets.clear()

    # --- 1. Profile ---

    async def profile(
        self,
        market: str,
        cutoff_date: date,
        validation_days: int = 60,
        forward_days: int = 5,
    ) -> dict[str, Any]:
        """Profile the feature matrix for a market up to the cutoff date.

        Computes NaN rates, return statistics, sector distribution, and
        returns the current MarketConfig as a baseline.  Pure compute --
        no LLM calls.
        """
        # Resolve universe symbols
        symbols = await prediction_service._resolve_symbols(market)
        if not symbols:
            raise RuntimeError(f"No symbols resolved for market={market}")

        # Date range: 2-year lookback to cutoff
        start_date = cutoff_date - timedelta(days=_TRAIN_LOOKBACK_DAYS)
        end_date = cutoff_date

        # Build feature matrix
        from app.services.feature_service import feature_service

        feature_df = await feature_service.build_feature_matrix(
            market=market,
            symbols=symbols,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
        )

        if feature_df.empty:
            raise RuntimeError(
                f"Feature matrix is empty for {market} "
                f"({start_date} to {end_date})"
            )

        # Compute statistics via Profiler._compute_stats (no LLM)
        from app.services.ml_agents.profiler import Profiler

        profiler_instance = Profiler.__new__(Profiler)
        stats = profiler_instance._compute_stats(
            feature_df, market, symbols, None
        )

        # Current baseline config
        cfg = get_market_config(market)
        baseline_config = dataclasses.asdict(cfg)

        return {
            "market": stats["market"],
            "universe_size": stats["universe_size"],
            "n_trading_days": stats["n_trading_days"],
            "date_range": list(stats["date_range"]),
            "feature_nan_rates": stats["feature_nan_rates"],
            "median_nan_rate": stats["median_nan_rate"],
            "sparse_features": stats["sparse_features"],
            "sector_distribution": stats["sector_distribution"],
            "min_sector_size": stats["min_sector_size"],
            "return_stats": stats["return_stats"],
            "baseline_config": baseline_config,
        }

    # --- 2. Submit training ---

    async def submit_training(
        self,
        market: str,
        cutoff_date: date,
        forward_days: int,
        config: dict[str, Any],
    ) -> str:
        """Submit a training task.  Returns task_id immediately.

        The config dict is applied as an override to the market's base
        MarketConfig.  Training runs in a background asyncio.Task.
        """
        async with self._submit_lock:
            if market in self._training_markets:
                raise RuntimeError(
                    f"Training already in progress for market {market}"
                )

            task_id = f"mlt-{uuid.uuid4().hex[:12]}"
            task = _ToolTask(task_id=task_id, market=market)

            # Apply config override
            effective_config = apply_override(market, config)
            task._config = effective_config

            self._tasks[task_id] = task
            self._training_markets.add(market)

        # Launch background training
        try:
            async_task = asyncio.create_task(
                self._run_training(task, market, cutoff_date, forward_days, effective_config)
            )
            task._asyncio_task = async_task
        except Exception:
            self._training_markets.discard(market)
            del self._tasks[task_id]
            raise

        logger.info(
            "ML tools training submitted: task_id=%s, market=%s, "
            "cutoff=%s, fwd=%d",
            task_id,
            market,
            cutoff_date,
            forward_days,
        )
        return task_id

    async def _run_training(
        self,
        task: _ToolTask,
        market: str,
        cutoff_date: date,
        forward_days: int,
        config: MarketConfig,
    ) -> None:
        """Background training coroutine."""
        t0 = time.monotonic()
        try:
            task.status = "training"
            task.progress = 10.0
            task.status_detail = "Resolving symbols"

            # Resolve symbols
            symbols = await prediction_service._resolve_symbols(market)
            if not symbols:
                raise RuntimeError(f"No symbols resolved for market={market}")

            task.progress = 20.0
            task.status_detail = f"Training ensemble ({len(symbols)} symbols)"

            # Train using the existing backtest training pipeline
            train_result = await prediction_service.train_for_backtest(
                market=market,
                symbols=symbols,
                forward_days=forward_days,
                cutoff_date=cutoff_date,
                config=config,
            )

            task.progress = 90.0

            # Cache models for the validate step
            task._models = train_result["models"]
            task._feature_cols = train_result["feature_cols"]

            # Build serializable result (exclude non-serializable objects)
            serializable_result = {
                k: v
                for k, v in train_result.items()
                if k not in ("models", "feature_cols", "calibrator")
            }
            # Ensure numpy types are converted to native Python
            serializable_result = _sanitize_for_json(serializable_result)

            # Train direction model alongside ranking (non-fatal, optional)
            try:
                task.status_detail = "Training direction model"
                from app.services.direction_service import _train_direction_model

                pool = get_db_pool()
                dir_result = await _train_direction_model(
                    market, cutoff_date, forward_days, pool, config,
                )
                if dir_result:
                    serializable_result["direction_auc"] = dir_result.get("auc")
                    serializable_result["direction_brier"] = dir_result.get("brier_score")
                    serializable_result["direction_quality_passed"] = dir_result.get(
                        "quality_passed"
                    )
                    # Cache for validate step
                    task._direction_models = dir_result.get("models")
                    task._direction_feature_cols = dir_result.get("feature_cols")
                    task._direction_calibrator = dir_result.get("calibrator")
                    logger.info(
                        "Direction model trained: AUC=%.4f, Brier=%.4f",
                        dir_result.get("auc", 0.0),
                        dir_result.get("brier_score", 0.0),
                    )
            except ImportError:
                logger.debug(
                    "direction_service not available, skipping direction model"
                )
            except Exception as e:
                logger.warning(
                    "Direction training in ML tools failed (non-fatal): %s", e
                )

            task.result = serializable_result
            task.status = "completed"
            task.progress = 100.0
            task.completed_at = datetime.now()

            duration = time.monotonic() - t0
            logger.info(
                "ML tools training completed: task_id=%s, market=%s, "
                "IC=%.4f, ICIR=%.4f, duration=%.1fs",
                task.task_id,
                market,
                train_result.get("ic", 0.0),
                train_result.get("icir", 0.0),
                duration,
            )

        except Exception as e:
            logger.error(
                "ML tools training failed: task_id=%s, error=%s",
                task.task_id,
                e,
                exc_info=True,
            )
            task.status = "failed"
            task.error = str(e)[:2000]
            task.completed_at = datetime.now()

        finally:
            self._training_markets.discard(market)
            # Schedule model cache cleanup
            try:
                loop = asyncio.get_running_loop()
                loop.call_later(
                    _MODEL_CACHE_TTL,
                    self._cleanup_task_models,
                    task.task_id,
                )
            except RuntimeError as exc:
                logger.debug("Could not schedule model cleanup: %s", exc)

    def _cleanup_task_models(self, task_id: str) -> None:
        """Remove cached models from a completed task after TTL expires."""
        task = self._tasks.get(task_id)
        if task and task._models is not None:
            logger.info(
                "Cleaning up cached models for task %s", task_id
            )
            task._models = None
            task._feature_cols = None
            task._direction_models = None
            task._direction_feature_cols = None
            task._direction_calibrator = None

    # --- 3. Get task ---

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        """Look up a task by ID and return its status."""
        task = self._tasks.get(task_id)
        if task is None:
            return None
        return task.to_dict()

    # --- 4. Validate ---

    async def validate(
        self,
        task_id: str,
        cutoff_date: date,
        validation_days: int,
        forward_days: int,
    ) -> dict[str, Any]:
        """Run multi-day inference on the validation window using trained models.

        Requires a completed training task with cached models.  Reuses
        the validation logic from ml_backtest_service._compute_validation_metrics().
        """
        task = self._tasks.get(task_id)
        if task is None:
            raise ValueError(f"Task not found: {task_id}")
        if task.status != "completed":
            raise ValueError(
                f"Task {task_id} is not completed (status={task.status})"
            )
        if task._models is None or task._feature_cols is None:
            raise ValueError(
                f"Models for task {task_id} have been cleaned up "
                f"(TTL={_MODEL_CACHE_TTL}s expired). Re-run training."
            )

        market = task.market
        models = task._models
        feature_cols = task._feature_cols
        config = task._config or get_market_config(market)

        # Resolve symbols
        symbols = await prediction_service._resolve_symbols(market)
        if not symbols:
            raise RuntimeError(f"No symbols resolved for market={market}")

        # Build feature matrix covering training + validation range
        train_start = cutoff_date - timedelta(days=_TRAIN_LOOKBACK_DAYS)
        val_end = cutoff_date + timedelta(days=int(validation_days * 1.5))

        from app.services.feature_service import feature_service

        full_feature_df = await feature_service.build_feature_matrix(
            market=market,
            symbols=symbols,
            start_date=train_start.isoformat(),
            end_date=val_end.isoformat(),
            config_override=config,
        )

        if full_feature_df.empty:
            raise RuntimeError("Feature matrix is empty for validation")

        full_feature_df["date"] = pd.to_datetime(full_feature_df["date"])

        # Fetch close prices for the full range
        close_df = await prediction_service._fetch_close_prices(
            market, symbols, train_start.isoformat(), val_end.isoformat()
        )
        if close_df.empty:
            raise RuntimeError("Close price data is empty for validation")
        close_df["date"] = pd.to_datetime(close_df["date"])

        # Determine validation trading dates (dates after cutoff)
        cutoff_ts = pd.Timestamp(cutoff_date)
        all_dates = sorted(full_feature_df["date"].unique())
        val_trading_dates = [d for d in all_dates if d > cutoff_ts][
            :validation_days
        ]

        if len(val_trading_dates) < 5:
            raise RuntimeError(
                f"Only {len(val_trading_dates)} validation dates after "
                f"cutoff {cutoff_date} (need at least 5)"
            )

        logger.info(
            "ML tools validate: task=%s, market=%s, %d val dates (%s to %s)",
            task_id,
            market,
            len(val_trading_dates),
            val_trading_dates[0].strftime("%Y-%m-%d"),
            val_trading_dates[-1].strftime("%Y-%m-%d"),
        )

        # Multi-day inference on validation window
        predictions_rows: list[dict[str, Any]] = []
        for d in val_trading_dates:
            day_df = full_feature_df[full_feature_df["date"] == d]
            if day_df.empty:
                continue

            # Align columns -- add missing feature columns as NaN
            missing_cols = [
                c for c in feature_cols if c not in day_df.columns
            ]
            if missing_cols:
                for mc in missing_cols:
                    day_df = day_df.assign(**{mc: np.nan})

            X = day_df[feature_cols].values
            scores = np.mean([m.predict(X) for m in models], axis=0)

            for sym, score in zip(day_df["symbol"].values, scores):
                predictions_rows.append(
                    {
                        "date": d,
                        "symbol": sym,
                        "predicted_score": float(score),
                    }
                )

        predictions_df = pd.DataFrame(predictions_rows)
        if predictions_df.empty:
            raise RuntimeError("No predictions generated for validation window")

        # Compute validation metrics (reuse ml_backtest_service logic)
        from app.services.ml_backtest_service import backtest_service

        val_metrics = backtest_service._compute_validation_metrics(
            predictions_df, close_df, forward_days, val_trading_dates
        )

        # --- Direction model forward validation (optional) ---
        direction_val = {}
        if (
            task._direction_models is not None
            and task._direction_feature_cols is not None
        ):
            try:
                direction_val = await self._validate_direction_forward(
                    task, market, full_feature_df, close_df,
                    val_trading_dates, forward_days, cutoff_date, val_end,
                )
            except Exception as e:
                logger.warning("Direction forward validation failed: %s", e)

        # Extract and format the response
        ic_curve_raw = val_metrics.get("ic_curve", [])
        ic_values = (
            [entry["ic"] for entry in ic_curve_raw]
            if ic_curve_raw and isinstance(ic_curve_raw[0], dict)
            else ic_curve_raw
        )

        quintile_returns_raw = val_metrics.get("quintile_returns", {})
        quintile_returns = {
            str(k): float(v) for k, v in quintile_returns_raw.items()
        }

        result = {
            "val_ic": val_metrics.get("val_ic", 0.0) or 0.0,
            "val_icir": val_metrics.get("val_icir", 0.0) or 0.0,
            "val_spread": val_metrics.get("val_spread", 0.0) or 0.0,
            "val_direction_accuracy": (
                val_metrics.get("val_direction_accuracy", 0.0) or 0.0
            ),
            "quintile_returns": quintile_returns,
            "val_hit_rate": val_metrics.get("val_hit_rate", 0.0) or 0.0,
            "ic_curve": ic_values,
            "val_max_drawdown": val_metrics.get("val_max_drawdown"),
        }
        # Merge direction forward validation metrics
        if direction_val:
            result["val_direction_auc"] = direction_val.get("auc")
            result["val_direction_brier"] = direction_val.get("brier")
            result["val_direction_hit_rate"] = direction_val.get("hit_rate")

        return result

    async def _validate_direction_forward(
        self,
        task: _ToolTask,
        market: str,
        full_feature_df: pd.DataFrame,
        close_df: pd.DataFrame,
        val_trading_dates: list,
        forward_days: int,
        cutoff_date,
        val_end,
    ) -> dict[str, float]:
        """Run direction model inference on validation window, compute AUC/Brier.

        This is the 'forward validation' -- model was trained on data before
        cutoff_date, now we run inference on dates AFTER cutoff and compare
        predicted up_probability against actual binary outcomes.
        """
        try:
            from app.services.direction_service import _apply_calibrator
            from app.services.market_features_service import (
                MARKET_FEATURE_COLUMNS,
                build_market_features,
            )
        except ImportError:
            logger.debug("direction/market_features services not available")
            return {}

        from sklearn.metrics import brier_score_loss, roc_auc_score

        # Build market features for validation window
        pool = get_db_pool()
        mkt_features = await build_market_features(
            market, cutoff_date, val_end, pool,
        )

        dir_models = task._direction_models
        dir_feature_cols = task._direction_feature_cols
        dir_calibrator = task._direction_calibrator

        dir_preds_rows: list[dict] = []
        for d in val_trading_dates:
            day_df = full_feature_df[full_feature_df["date"] == d].copy()
            if day_df.empty:
                continue

            # Merge market features
            if not mkt_features.empty:
                day_df = day_df.merge(mkt_features, on="date", how="left")

            # Align direction feature columns
            for mc in dir_feature_cols:
                if mc not in day_df.columns:
                    day_df[mc] = np.nan

            X_dir = day_df[dir_feature_cols].values
            raw_probs = np.mean(
                [m.predict(X_dir) for m in dir_models], axis=0,
            )

            if dir_calibrator is not None:
                probs = _apply_calibrator(dir_calibrator, raw_probs)
            else:
                probs = raw_probs

            for sym, prob in zip(day_df["symbol"].values, probs):
                dir_preds_rows.append({
                    "date": d, "symbol": str(sym),
                    "up_probability": float(prob),
                })

        if not dir_preds_rows:
            return {}

        dir_df = pd.DataFrame(dir_preds_rows)

        # Compute actual forward returns from close prices
        close_df_copy = close_df.copy()
        close_df_copy = close_df_copy.sort_values(["symbol", "date"])

        merged = dir_df.merge(
            close_df_copy[["date", "symbol", "close"]],
            on=["date", "symbol"], how="left",
        )

        # Build forward close lookup
        forward_rows = []
        for sym, sym_close in close_df_copy.groupby("symbol"):
            sym_close = sym_close.sort_values("date").reset_index(drop=True)
            dates = sym_close["date"].values
            closes = sym_close["close"].values
            for i in range(len(dates)):
                fwd_idx = i + forward_days
                if fwd_idx < len(dates):
                    forward_rows.append({
                        "date": dates[i],
                        "symbol": sym,
                        "forward_close": closes[fwd_idx],
                    })

        if not forward_rows:
            return {}

        fwd_df = pd.DataFrame(forward_rows)
        merged = merged.merge(fwd_df, on=["date", "symbol"], how="inner")
        merged = merged.dropna(subset=["close", "forward_close"])

        if len(merged) < 20:
            return {}

        merged["actual_return"] = (
            merged["forward_close"] - merged["close"]
        ) / merged["close"]
        merged["binary_outcome"] = (merged["actual_return"] > 0).astype(int)

        probs = merged["up_probability"].values
        outcomes = merged["binary_outcome"].values

        try:
            auc = float(roc_auc_score(outcomes, probs))
        except ValueError:
            auc = 0.5
        try:
            brier = float(brier_score_loss(outcomes, probs))
        except ValueError:
            brier = 0.25
        correct = (
            ((probs > 0.5) & (outcomes == 1))
            | ((probs < 0.5) & (outcomes == 0))
        )
        hit_rate = float(correct.mean())

        logger.info(
            "Direction forward validation: market=%s, %d samples, "
            "AUC=%.4f, Brier=%.4f, hit_rate=%.4f",
            market, len(merged), auc, brier, hit_rate,
        )

        return {"auc": round(auc, 6), "brier": round(brier, 6), "hit_rate": round(hit_rate, 4)}

    # --- 5. Rolling Backtest ---

    async def submit_rolling_backtest(
        self,
        market: str,
        cutoff_date: date,
        validation_days: int,
        forward_days: int,
        retrain_interval: int,
        config: dict[str, Any],
    ) -> str:
        """Submit a rolling backtest task.  Returns task_id immediately.

        Rolling backtest retrains the model every retrain_interval trading
        days, simulating production behavior.  Feature matrix and prices are
        built once and sliced per retrain window.
        """
        async with self._submit_lock:
            if market in self._training_markets:
                raise RuntimeError(
                    f"Training/backtest already in progress for market {market}"
                )

            task_id = f"mlt-{uuid.uuid4().hex[:12]}"
            task = _ToolTask(task_id=task_id, market=market)

            effective_config = apply_override(market, config)
            task._config = effective_config

            self._tasks[task_id] = task
            self._training_markets.add(market)

        try:
            async_task = asyncio.create_task(
                self._run_rolling_backtest_task(
                    task, market, cutoff_date, validation_days,
                    forward_days, retrain_interval, effective_config,
                )
            )
            task._asyncio_task = async_task
        except Exception:
            self._training_markets.discard(market)
            del self._tasks[task_id]
            raise

        logger.info(
            "ML tools rolling backtest submitted: task_id=%s, market=%s, "
            "cutoff=%s, val_days=%d, retrain_interval=%d",
            task_id, market, cutoff_date, validation_days, retrain_interval,
        )
        return task_id

    async def _run_rolling_backtest_task(
        self,
        task: _ToolTask,
        market: str,
        cutoff_date: date,
        validation_days: int,
        forward_days: int,
        retrain_interval: int,
        config: MarketConfig,
    ) -> None:
        """Background rolling backtest coroutine.

        Reuses the same algorithm as ml_backtest_service._run_rolling_backtest
        but stores results in the _ToolTask for agent consumption.
        """
        t0 = time.monotonic()
        try:
            task.status = "training"
            task.progress = 1.0

            # Step 1: Resolve symbols
            symbols = await prediction_service._resolve_symbols(market)
            if not symbols:
                raise RuntimeError(f"No symbols resolved for market={market}")
            task.progress = 3.0

            # Step 2: Build full feature matrix (one-time)
            train_start = cutoff_date - timedelta(days=_TRAIN_LOOKBACK_DAYS)
            val_end = cutoff_date + timedelta(days=int(validation_days * 1.5))

            from app.services.feature_service import feature_service

            full_feature_df = await feature_service.build_feature_matrix(
                market=market,
                symbols=symbols,
                start_date=train_start.isoformat(),
                end_date=val_end.isoformat(),
                config_override=config,
            )
            if full_feature_df.empty:
                raise RuntimeError("Feature matrix is empty")
            full_feature_df["date"] = pd.to_datetime(full_feature_df["date"])
            task.progress = 10.0

            # Step 3: Fetch close prices (one-time)
            close_df = await prediction_service._fetch_close_prices(
                market, symbols, train_start.isoformat(), val_end.isoformat(),
            )
            if close_df.empty:
                raise RuntimeError("Close price data is empty")
            close_df["date"] = pd.to_datetime(close_df["date"])

            # Step 4: Determine validation trading dates
            cutoff_ts = pd.Timestamp(cutoff_date)
            all_dates = sorted(full_feature_df["date"].unique())
            val_trading_dates = [
                d for d in all_dates if d > cutoff_ts
            ][:validation_days]

            if len(val_trading_dates) < 5:
                raise RuntimeError(
                    f"Only {len(val_trading_dates)} validation dates "
                    f"after cutoff {cutoff_date} (need at least 5)"
                )

            # Step 5: Split into retrain windows
            retrain_windows: list[list] = []
            for i in range(0, len(val_trading_dates), retrain_interval):
                retrain_windows.append(
                    val_trading_dates[i : i + retrain_interval]
                )

            total_retrains = len(retrain_windows)
            task.progress = 12.0

            logger.info(
                "ML tools rolling backtest %s: %d val dates -> %d retrains "
                "(interval=%d)",
                task.task_id, len(val_trading_dates),
                total_retrains, retrain_interval,
            )

            # Step 6: Rolling retrain loop
            all_predictions: list[dict[str, Any]] = []
            per_retrain_metrics: list[dict[str, Any]] = []

            window_pct = 78.0 / max(total_retrains, 1)

            for win_idx, window_dates in enumerate(retrain_windows):
                retrain_num = win_idx + 1
                window_base = 12.0 + window_pct * win_idx
                task.progress = window_base
                task.status_detail = (
                    f"Window {retrain_num}/{total_retrains}: training"
                )

                retrain_cutoff_ts = window_dates[0] - pd.Timedelta(days=1)
                retrain_cutoff = retrain_cutoff_ts.date()

                train_slice = full_feature_df[
                    full_feature_df["date"] <= retrain_cutoff_ts
                ].copy()

                if len(train_slice) < 100:
                    per_retrain_metrics.append({
                        "retrain_date": str(retrain_cutoff),
                        "train_ic": None, "window_ic": None,
                        "n_dates": len(window_dates), "status": "skipped",
                    })
                    continue

                try:
                    train_result = await prediction_service.train_for_backtest(
                        market=market, symbols=symbols,
                        forward_days=forward_days,
                        cutoff_date=retrain_cutoff,
                        config=config, feature_df=train_slice,
                    )
                except Exception as e:
                    logger.error(
                        "Rolling BT %s retrain %d failed: %s",
                        task.task_id, retrain_num, e,
                    )
                    per_retrain_metrics.append({
                        "retrain_date": str(retrain_cutoff),
                        "train_ic": None, "window_ic": None,
                        "n_dates": len(window_dates),
                        "status": "failed", "error": str(e)[:200],
                    })
                    continue

                models = train_result["models"]
                feature_cols = train_result["feature_cols"]
                train_ic = train_result.get("ic")

                task.progress = window_base + window_pct * 0.6
                task.status_detail = (
                    f"Window {retrain_num}/{total_retrains}: "
                    f"inference ({len(window_dates)} days)"
                )

                # Inference on window dates
                window_predictions: list[dict[str, Any]] = []
                n_window_dates = len(window_dates)
                for day_idx, d in enumerate(window_dates):
                    day_df = full_feature_df[full_feature_df["date"] == d]
                    if day_df.empty:
                        continue
                    missing_cols = [
                        c for c in feature_cols if c not in day_df.columns
                    ]
                    if missing_cols:
                        for mc in missing_cols:
                            day_df = day_df.assign(**{mc: np.nan})
                    X = day_df[feature_cols].values
                    scores = np.mean(
                        [m.predict(X) for m in models], axis=0
                    )
                    for sym, score in zip(day_df["symbol"].values, scores):
                        window_predictions.append({
                            "date": d, "symbol": sym,
                            "predicted_score": float(score),
                        })
                    task.progress = (
                        window_base
                        + window_pct * (0.6 + 0.4 * (day_idx + 1) / n_window_dates)
                    )

                all_predictions.extend(window_predictions)

                # Compute per-window IC
                window_ic = None
                if window_predictions:
                    from scipy.stats import spearmanr

                    win_pred_df = pd.DataFrame(window_predictions)
                    close_sorted = close_df.sort_values(
                        ["symbol", "date"]
                    ).copy()
                    close_sorted["fwd_ret"] = close_sorted.groupby(
                        "symbol"
                    )["close"].transform(
                        lambda x: x.shift(-forward_days) / x - 1
                    )
                    win_merged = win_pred_df.merge(
                        close_sorted[["symbol", "date", "fwd_ret"]],
                        on=["symbol", "date"], how="left",
                    ).dropna(subset=["fwd_ret", "predicted_score"])

                    if len(win_merged) >= 20:
                        win_ics = []
                        for _, grp in win_merged.groupby("date"):
                            if len(grp) >= 10:
                                c, _ = spearmanr(
                                    grp["predicted_score"], grp["fwd_ret"]
                                )
                                if not np.isnan(c):
                                    win_ics.append(c)
                        if win_ics:
                            window_ic = round(float(np.mean(win_ics)), 6)

                # Direction model training for this window (non-fatal, optional)
                window_dir_auc = None
                window_dir_brier = None
                try:
                    from app.services.direction_service import _train_direction_model

                    pool = get_db_pool()
                    dir_result = await _train_direction_model(
                        market, retrain_cutoff, forward_days, pool, config,
                        persist=False,
                    )
                    if dir_result:
                        window_dir_auc = dir_result.get("auc")
                        window_dir_brier = dir_result.get("brier_score")
                except ImportError:
                    pass  # direction_service not available
                except Exception as e:
                    logger.debug(
                        "Rolling BT direction training failed for window %d: %s",
                        retrain_num, e,
                    )

                per_retrain_metrics.append({
                    "retrain_date": str(retrain_cutoff),
                    "train_ic": round(train_ic, 6) if train_ic else None,
                    "window_ic": window_ic,
                    "n_dates": len(window_dates),
                    "n_predictions": len(window_predictions),
                    "direction_auc": (
                        round(window_dir_auc, 6) if window_dir_auc else None
                    ),
                    "direction_brier": (
                        round(window_dir_brier, 6) if window_dir_brier else None
                    ),
                    "status": "completed",
                })

                # Release model references
                del models, train_result

                logger.info(
                    "Rolling BT %s: retrain %d/%d done, window_ic=%s",
                    task.task_id, retrain_num, total_retrains, window_ic,
                )

            # Step 7: Aggregate
            if not all_predictions:
                raise RuntimeError("All retrain windows failed")

            predictions_df = pd.DataFrame(all_predictions)
            task.progress = 92.0
            task.status_detail = "Aggregating metrics"

            # Step 8: Compute aggregate validation metrics
            from app.services.ml_backtest_service import backtest_service

            val_metrics = backtest_service._compute_validation_metrics(
                predictions_df, close_df, forward_days, val_trading_dates,
            )

            # Step 9: Compute cumulative returns
            cumulative_returns = backtest_service._compute_cumulative_returns(
                predictions_df, close_df, forward_days,
            )

            # Aggregate direction model metrics across retrain windows
            dir_aucs = [
                m["direction_auc"] for m in per_retrain_metrics
                if m.get("direction_auc") is not None
            ]
            dir_briers = [
                m["direction_brier"] for m in per_retrain_metrics
                if m.get("direction_brier") is not None
            ]

            # Build result dict for agent consumption
            result = _sanitize_for_json({
                "backtest_type": "rolling",
                "retrain_interval": retrain_interval,
                "retrain_count": total_retrains,
                "n_validation_dates": len(val_trading_dates),
                "val_ic": val_metrics.get("val_ic", 0.0),
                "val_icir": val_metrics.get("val_icir", 0.0),
                "val_spread": val_metrics.get("val_spread", 0.0),
                "val_direction_accuracy": val_metrics.get(
                    "val_direction_accuracy", 0.0
                ),
                "val_hit_rate": val_metrics.get("val_hit_rate", 0.0),
                "val_max_drawdown": val_metrics.get("val_max_drawdown"),
                "avg_direction_auc": (
                    round(float(np.mean(dir_aucs)), 6) if dir_aucs else None
                ),
                "avg_direction_brier": (
                    round(float(np.mean(dir_briers)), 6) if dir_briers else None
                ),
                "per_retrain_metrics": per_retrain_metrics,
                "cumulative_returns": cumulative_returns,
                "ic_curve": val_metrics.get("ic_curve", []),
            })

            task.result = result
            task.status = "completed"
            task.progress = 100.0
            task.completed_at = datetime.now()

            duration = time.monotonic() - t0
            logger.info(
                "ML tools rolling backtest %s completed: val_IC=%.4f, "
                "%d retrains, duration=%.0fs",
                task.task_id,
                result.get("val_ic", 0),
                total_retrains,
                duration,
            )

        except Exception as e:
            logger.error(
                "ML tools rolling backtest %s failed: %s",
                task.task_id, e, exc_info=True,
            )
            task.status = "failed"
            task.error = str(e)[:2000]
            task.completed_at = datetime.now()

        finally:
            self._training_markets.discard(market)
            try:
                loop = asyncio.get_running_loop()
                loop.call_later(
                    _MODEL_CACHE_TTL,
                    self._cleanup_task_models,
                    task.task_id,
                )
            except RuntimeError as exc:
                logger.debug("Could not schedule cleanup: %s", exc)

    # --- 6. Deploy ---

    async def deploy(
        self,
        market: str,
        backtest_id: str,
        effective_config: dict[str, Any],
        iteration: int,
        val_ic: float,
        train_ic: float | None = None,
        train_icir: float | None = None,
    ) -> dict[str, str]:
        """Persist a validated backtest result to the ml_backtests table.

        Writes the result as a completed backtest record that can be
        queried via the standard backtest list/detail endpoints.
        """
        pool = get_db_pool()

        bt_id = uuid.UUID(backtest_id) if backtest_id else uuid.uuid4()

        # Extract from effective_config or use defaults
        fwd_days = effective_config.get("forward_days", 5)
        val_days = effective_config.get("validation_days", 60)

        results_json = {
            "deployed_by": "ml_agent",
            "iteration": iteration,
            "val_ic": val_ic,
            "train_ic": train_ic,
            "train_icir": train_icir,
        }

        try:
            async with pool.acquire(timeout=10) as conn:
                await conn.execute(
                    _SQL_UPSERT_BACKTEST,
                    bt_id,                                           # $1 id
                    market,                                          # $2 market
                    date.today(),                                    # $3 cutoff_date
                    val_days,                                        # $4 validation_days
                    fwd_days,                                        # $5 forward_days
                    json.dumps(effective_config),                    # $6 config_override
                    json.dumps(effective_config),                    # $7 effective_config
                    "completed",                                     # $8 status
                    json.dumps(results_json, default=_json_default), # $9 results
                    val_ic,                                          # $10 val_ic
                    train_ic,                                        # $11 train_ic
                    train_icir,                                      # $12 train_icir
                    f"ml-agent-{uuid.uuid4().hex[:8]}",             # $13 agent_run_id
                    iteration,                                       # $14 agent_iteration
                )
        except Exception as e:
            logger.error("Failed to deploy backtest result: %s", e)
            raise RuntimeError(f"Deploy failed: {e}")

        logger.info(
            "ML tools deploy: backtest_id=%s, market=%s, val_ic=%.4f",
            bt_id,
            market,
            val_ic,
        )

        return {"status": "ok"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _json_default(obj: Any) -> Any:
    """JSON serializer fallback for numpy and date types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively convert numpy types to native Python for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(item) for item in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    if isinstance(obj, np.ndarray):
        return _sanitize_for_json(obj.tolist())
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


# Module singleton
ml_tools_service = MLToolsService()
