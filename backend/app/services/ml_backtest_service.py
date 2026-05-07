"""ML Backtest Service -- historical cutoff backtesting with LLM agent loop.

Orchestrates:
1. Universe resolution (reuses production logic)
2. One-time feature matrix build for the full date range
3. Training via prediction_service.train_for_backtest()
4. Multi-day inference on validation window
5. Validation metric computation (IC curve, quintile returns, direction accuracy)
6. Optional LLM agent loop (Profiler -> Strategist -> Train -> Evaluate -> iterate)
7. Result persistence to ml_backtests table

All progress is tracked in BacktestTask for frontend observability.

AlphaForge adaptation:
- Uses get_db_pool() (asyncpg pool) instead of settings_cache.pool.
- All ML logic is identical to WebStock's data-processor.
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

from app.config import get_settings
from app.core.database import get_db_pool
from app.services.market_config import MarketConfig, apply_override, get_market_config
from app.services.prediction_service import prediction_service

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BacktestTask -- structured progress tracking
# ---------------------------------------------------------------------------


@dataclass
class BacktestTask:
    """In-memory backtest task with structured observability."""

    task_id: str
    backtest_id: str
    market: str
    status: str = "pending"  # pending/running/completed/failed
    progress: float = 0.0
    message: str = ""
    current_phase: str = ""  # profiling/training/inference/evaluating/storing
    current_iteration: int = 0
    max_iterations: int = 1
    iterations: list[dict[str, Any]] = field(default_factory=list)
    current_retrain: int = 0
    total_retrains: int = 0
    error: Optional[str] = None
    _asyncio_task: Optional[asyncio.Task] = field(
        default=None, repr=False, compare=False
    )
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

    def to_dict(self) -> dict[str, Any]:
        elapsed = (
            (datetime.now() - self.created_at).total_seconds()
            if self.created_at
            else 0
        )
        return {
            "task_id": self.task_id,
            "backtest_id": self.backtest_id,
            "market": self.market,
            "status": self.status,
            "progress": round(self.progress, 1),
            "message": self.message,
            "current_phase": self.current_phase,
            "current_iteration": self.current_iteration,
            "max_iterations": self.max_iterations,
            "iterations": self.iterations,
            "current_retrain": self.current_retrain or None,
            "total_retrains": self.total_retrains or None,
            "elapsed_seconds": round(elapsed, 1),
            "created_at": (
                self.created_at.isoformat() if self.created_at else None
            ),
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------

_SQL_INSERT_BACKTEST = """
INSERT INTO ml_backtests (
    id, market, cutoff_date, validation_days, forward_days,
    config_override, effective_config, status, results
) VALUES ($1::uuid, $2, $3, $4, $5, $6::jsonb, $7::jsonb, $8, '{}'::jsonb)
"""

_SQL_UPDATE_BACKTEST_COMPLETED = """
UPDATE ml_backtests SET
    status = 'completed',
    train_ic = $2, train_icir = $3, train_ndcg = $4,
    fold_ics = $5::jsonb,
    ensemble_size = $6, feature_count = $7, symbol_count = $8,
    val_ic = $9, val_icir = $10,
    val_direction_accuracy = $11, val_spread = $12,
    val_q1_return = $13, val_q5_return = $14,
    val_hit_rate = $15, val_max_drawdown = $16,
    results = $17::jsonb,
    effective_config = $18::jsonb,
    duration_seconds = $19,
    agent_run_id = $20, agent_iteration = $21,
    completed_at = NOW()
WHERE id = $1::uuid
"""

_SQL_UPDATE_BACKTEST_FAILED = """
UPDATE ml_backtests SET status = 'failed', error = $2, duration_seconds = $3,
    completed_at = NOW()
WHERE id = $1::uuid
"""

_SQL_GET_BACKTEST = """
SELECT * FROM ml_backtests WHERE id = $1::uuid
"""

_SQL_LIST_BACKTESTS = """
SELECT id, market, cutoff_date, validation_days, forward_days, status,
    train_ic, train_icir, val_ic, val_icir, val_direction_accuracy, val_spread,
    agent_iteration, duration_seconds, created_at, completed_at
FROM ml_backtests
WHERE ($1::text IS NULL OR market = $1)
ORDER BY created_at DESC
LIMIT $2
"""

_SQL_DELETE_BACKTEST = """
DELETE FROM ml_backtests WHERE id = $1::uuid RETURNING id
"""


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class MLBacktestService:
    """Historical cutoff backtesting orchestrator."""

    def __init__(self) -> None:
        self._tasks: dict[str, BacktestTask] = {}
        self._running_markets: set[str] = set()

    # --- Task management ---

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        task = self._tasks.get(task_id)
        return task.to_dict() if task else None

    # --- API methods ---

    async def start_backtest(
        self,
        market: str,
        cutoff_date: date,
        validation_days: int = 60,
        forward_days: int = 5,
        config_override: dict[str, Any] | None = None,
        use_llm_agents: bool = False,
        max_iterations: int = 3,
        backtest_type: str = "static",
        retrain_interval: int = 5,
    ) -> tuple[str, str]:
        """Start a backtest task. Returns (task_id, backtest_id)."""
        if market in self._running_markets:
            raise RuntimeError(f"Backtest already running for market {market}")

        task_id = f"bt-{uuid.uuid4().hex[:12]}"
        backtest_id = str(uuid.uuid4())
        max_iter = max_iterations if use_llm_agents else 1

        task = BacktestTask(
            task_id=task_id,
            backtest_id=backtest_id,
            market=market,
            max_iterations=max_iter,
        )
        self._tasks[task_id] = task
        self._running_markets.add(market)

        # Build effective config, storing backtest_type metadata
        eff_config = dataclasses.asdict(get_market_config(market))
        if backtest_type == "rolling":
            eff_config["backtest_type"] = "rolling"
            eff_config["retrain_interval"] = retrain_interval

        # Insert initial DB record
        try:
            pool = get_db_pool()
            async with pool.acquire(timeout=10) as conn:
                await conn.execute(
                    _SQL_INSERT_BACKTEST,
                    uuid.UUID(backtest_id),
                    market,
                    cutoff_date,
                    validation_days,
                    forward_days,
                    json.dumps(config_override) if config_override else None,
                    json.dumps(eff_config),
                    "pending",
                )
        except Exception as e:
            logger.error("Failed to insert backtest record: %s", e)
            self._running_markets.discard(market)
            del self._tasks[task_id]
            raise

        # Route by backtest type
        if backtest_type == "rolling":
            coro = self._run_rolling_backtest(
                task,
                market,
                cutoff_date,
                validation_days,
                forward_days,
                config_override,
                retrain_interval,
            )
        else:
            coro = self._run_backtest(
                task,
                market,
                cutoff_date,
                validation_days,
                forward_days,
                config_override,
                use_llm_agents,
                max_iter,
            )

        async_task = asyncio.create_task(coro)
        task._asyncio_task = async_task
        return task_id, backtest_id

    async def list_backtests(
        self, market: str | None = None, limit: int = 50
    ) -> list[dict]:
        try:
            pool = get_db_pool()
        except RuntimeError:
            return []
        try:
            async with pool.acquire(timeout=10) as conn:
                rows = await conn.fetch(_SQL_LIST_BACKTESTS, market, limit)
            return [dict(r) for r in rows]
        except Exception as e:
            logger.error("Failed to list backtests: %s", e)
            return []

    async def get_backtest(self, backtest_id: str) -> dict | None:
        try:
            pool = get_db_pool()
        except RuntimeError:
            return None
        try:
            async with pool.acquire(timeout=10) as conn:
                row = await conn.fetchrow(
                    _SQL_GET_BACKTEST, uuid.UUID(backtest_id)
                )
            return dict(row) if row else None
        except Exception as e:
            logger.error("Failed to get backtest %s: %s", backtest_id, e)
            return None

    async def delete_backtest(self, backtest_id: str) -> bool:
        try:
            pool = get_db_pool()
        except RuntimeError:
            return False
        try:
            async with pool.acquire(timeout=10) as conn:
                row = await conn.fetchrow(
                    _SQL_DELETE_BACKTEST, uuid.UUID(backtest_id)
                )
            return row is not None
        except Exception as e:
            logger.error("Failed to delete backtest %s: %s", backtest_id, e)
            return False

    # --- Core backtest logic ---

    async def _run_rolling_backtest(
        self,
        task: BacktestTask,
        market: str,
        cutoff_date: date,
        validation_days: int,
        forward_days: int,
        config_override: dict[str, Any] | None,
        retrain_interval: int,
    ) -> None:
        """Rolling retrain backtest: retrain every N trading days.

        Simulates production behavior where models are retrained periodically.
        Builds features and prices once, then iterates through retrain windows.
        """
        t0 = time.monotonic()
        try:
            task.status = "running"
            task.message = "Starting rolling backtest"
            await self._update_db_status(task.backtest_id, "running")

            base_config = apply_override(market, config_override)

            # Step 1: Resolve symbols
            task.current_phase = "resolving"
            task.progress = 1.0
            task.message = "Resolving universe symbols"
            symbols = await self._resolve_symbols(market)
            logger.info(
                "Rolling backtest %s: resolved %d symbols for %s",
                task.task_id, len(symbols), market,
            )

            # Step 2: Build full feature matrix (one-time)
            task.current_phase = "building_features"
            task.progress = 3.0
            task.message = "Building feature matrix (one-time)"

            train_start = cutoff_date - timedelta(days=730)
            val_end = cutoff_date + timedelta(days=int(validation_days * 1.5))

            from app.services.feature_service import feature_service

            full_feature_df = await feature_service.build_feature_matrix(
                market=market,
                symbols=symbols,
                start_date=train_start.isoformat(),
                end_date=val_end.isoformat(),
                config_override=base_config,
            )
            if full_feature_df.empty:
                raise RuntimeError("Feature matrix is empty")

            full_feature_df["date"] = pd.to_datetime(full_feature_df["date"])
            logger.info(
                "Rolling backtest %s: feature matrix %d rows x %d cols",
                task.task_id, len(full_feature_df),
                len(full_feature_df.columns) - 2,
            )

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
                window = val_trading_dates[i : i + retrain_interval]
                retrain_windows.append(window)

            total_retrains = len(retrain_windows)
            task.total_retrains = total_retrains
            logger.info(
                "Rolling backtest %s: %d validation dates -> "
                "%d retrain windows (interval=%d)",
                task.task_id, len(val_trading_dates),
                total_retrains, retrain_interval,
            )

            # Step 6: Rolling retrain loop
            all_predictions: list[dict[str, Any]] = []
            per_retrain_metrics: list[dict[str, Any]] = []

            for win_idx, window_dates in enumerate(retrain_windows):
                retrain_num = win_idx + 1
                task.current_retrain = retrain_num
                task.progress = 5.0 + 90.0 * win_idx / total_retrains

                # Retrain cutoff = day before first date in this window
                retrain_cutoff_ts = window_dates[0] - pd.Timedelta(days=1)
                retrain_cutoff = retrain_cutoff_ts.date()

                # 6a: Training phase
                task.current_phase = "training"
                task.message = (
                    f"Retrain {retrain_num}/{total_retrains}: "
                    f"Training (cutoff {retrain_cutoff})"
                )

                train_slice = full_feature_df[
                    full_feature_df["date"] <= retrain_cutoff_ts
                ].copy()

                if len(train_slice) < 100:
                    logger.warning(
                        "Retrain %d: only %d training rows, skipping",
                        retrain_num, len(train_slice),
                    )
                    per_retrain_metrics.append({
                        "retrain_date": str(retrain_cutoff),
                        "train_ic": None,
                        "window_ic": None,
                        "n_dates": len(window_dates),
                        "status": "skipped",
                    })
                    continue

                try:
                    train_result = await prediction_service.train_for_backtest(
                        market=market,
                        symbols=symbols,
                        forward_days=forward_days,
                        cutoff_date=retrain_cutoff,
                        config=base_config,
                        feature_df=train_slice,
                    )
                except Exception as e:
                    logger.error(
                        "Retrain %d training failed: %s", retrain_num, e,
                    )
                    per_retrain_metrics.append({
                        "retrain_date": str(retrain_cutoff),
                        "train_ic": None,
                        "window_ic": None,
                        "n_dates": len(window_dates),
                        "status": "failed",
                        "error": str(e)[:200],
                    })
                    continue

                models = train_result["models"]
                feature_cols = train_result["feature_cols"]
                train_ic = train_result.get("ic")

                # 6b: Inference on window dates
                task.current_phase = "inference"
                task.message = (
                    f"Retrain {retrain_num}/{total_retrains}: "
                    f"Inference on {len(window_dates)} dates"
                )

                window_predictions: list[dict[str, Any]] = []
                for d in window_dates:
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
                    for sym, score in zip(
                        day_df["symbol"].values, scores
                    ):
                        window_predictions.append({
                            "date": d,
                            "symbol": sym,
                            "predicted_score": float(score),
                        })

                all_predictions.extend(window_predictions)

                # 6c: Compute per-window IC
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
                        on=["symbol", "date"],
                        how="left",
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

                per_retrain_metrics.append({
                    "retrain_date": str(retrain_cutoff),
                    "train_ic": round(train_ic, 6) if train_ic else None,
                    "window_ic": window_ic,
                    "n_dates": len(window_dates),
                    "n_predictions": len(window_predictions),
                    "status": "completed",
                })

                # 6d: Release model references
                del models, train_result

                logger.info(
                    "Rolling backtest %s: retrain %d/%d done, "
                    "window_ic=%s, %d predictions",
                    task.task_id, retrain_num, total_retrains,
                    window_ic, len(window_predictions),
                )

            # Step 7: Aggregate all predictions
            if not all_predictions:
                raise RuntimeError(
                    "All retrain windows failed -- no predictions"
                )

            predictions_df = pd.DataFrame(all_predictions)

            # Step 8: Compute aggregate validation metrics
            task.current_phase = "computing_metrics"
            task.progress = 95.0
            task.message = "Computing aggregate metrics"

            val_metrics = self._compute_validation_metrics(
                predictions_df, close_df, forward_days, val_trading_dates,
            )

            # Step 9: Compute cumulative returns for Q5/Q1/Spread
            cumulative_returns = self._compute_cumulative_returns(
                predictions_df, close_df, forward_days,
            )

            # Build rolling-specific results
            rolling_results = {
                "backtest_type": "rolling",
                "retrain_interval": retrain_interval,
                "retrain_count": total_retrains,
                "per_retrain_metrics": per_retrain_metrics,
                "cumulative_returns": cumulative_returns,
            }
            # Merge with standard validation metrics
            combined_result = {**val_metrics, **rolling_results}

            # Step 10: Save
            task.current_phase = "storing"
            task.progress = 98.0
            task.message = "Saving results"

            duration = time.monotonic() - t0
            await self._save_result(
                task, combined_result, base_config,
                duration, None, 0,
            )

            val_ic = val_metrics.get("val_ic", 0)
            quality_tag = "PASS" if val_ic and val_ic > 0 else "FAIL"

            task.status = "completed"
            task.progress = 100.0
            task.message = (
                f"Rolling backtest [{quality_tag}] -- "
                f"val_IC={val_ic:.4f}, "
                f"{total_retrains} retrains"
            )
            task.completed_at = datetime.now()

            logger.info(
                "Rolling backtest %s completed: val_IC=%.4f, "
                "%d retrains, duration=%.0fs",
                task.task_id, val_ic or 0,
                total_retrains, duration,
            )

        except Exception as e:
            logger.error(
                "Rolling backtest %s failed: %s",
                task.task_id, e, exc_info=True,
            )
            task.status = "failed"
            task.error = str(e)
            task.completed_at = datetime.now()
            duration = time.monotonic() - t0
            await self._mark_failed(task.backtest_id, str(e), duration)
        finally:
            self._running_markets.discard(market)

    async def _run_backtest(
        self,
        task: BacktestTask,
        market: str,
        cutoff_date: date,
        validation_days: int,
        forward_days: int,
        config_override: dict[str, Any] | None,
        use_llm_agents: bool,
        max_iterations: int,
    ) -> None:
        """Main backtest coroutine with full observability."""
        t0 = time.monotonic()
        try:
            task.status = "running"
            task.message = "Starting backtest"
            await self._update_db_status(task.backtest_id, "running")

            # Step 1: Resolve symbols
            task.current_phase = "resolving"
            task.progress = self._calc_progress(0, max_iterations, 0.0)
            task.message = "Resolving universe symbols"
            symbols = await self._resolve_symbols(market)
            logger.info(
                "Backtest %s: resolved %d symbols for %s",
                task.task_id,
                len(symbols),
                market,
            )

            # Step 2: Build complete feature matrix (one-time for full date range)
            task.current_phase = "building_features"
            task.progress = self._calc_progress(0, max_iterations, 5.0)
            task.message = "Building feature matrix (one-time)"

            # Date range: training lookback + validation window
            train_start = cutoff_date - timedelta(days=730)
            # Buffer for non-trading days
            val_end = cutoff_date + timedelta(days=int(validation_days * 1.5))

            base_config = apply_override(market, config_override)

            from app.services.feature_service import feature_service

            full_feature_df = await feature_service.build_feature_matrix(
                market=market,
                symbols=symbols,
                start_date=train_start.isoformat(),
                end_date=val_end.isoformat(),
                config_override=base_config,
            )

            if full_feature_df.empty:
                raise RuntimeError("Feature matrix is empty")

            full_feature_df["date"] = pd.to_datetime(full_feature_df["date"])
            logger.info(
                "Backtest %s: feature matrix %d rows x %d cols",
                task.task_id,
                len(full_feature_df),
                len(full_feature_df.columns) - 2,
            )

            # Fetch close prices for the full range (training + validation)
            close_df = await prediction_service._fetch_close_prices(
                market, symbols, train_start.isoformat(), val_end.isoformat()
            )
            if close_df.empty:
                raise RuntimeError("Close price data is empty")
            close_df["date"] = pd.to_datetime(close_df["date"])

            # Determine validation trading dates
            cutoff_ts = pd.Timestamp(cutoff_date)
            all_dates = sorted(full_feature_df["date"].unique())
            val_trading_dates = [d for d in all_dates if d > cutoff_ts][
                :validation_days
            ]

            if len(val_trading_dates) < 5:
                raise RuntimeError(
                    f"Only {len(val_trading_dates)} validation dates after cutoff "
                    f"{cutoff_date} (need at least 5)"
                )

            logger.info(
                "Backtest %s: %d validation trading dates (%s to %s)",
                task.task_id,
                len(val_trading_dates),
                val_trading_dates[0].strftime("%Y-%m-%d"),
                val_trading_dates[-1].strftime("%Y-%m-%d"),
            )

            # Training feature matrix (up to cutoff)
            train_feature_df = full_feature_df[
                full_feature_df["date"] <= cutoff_ts
            ].copy()

            # --- LLM Agent Loop (or single iteration) ---
            best_result: dict[str, Any] | None = None
            best_val_ic: float = -999.0
            best_config: MarketConfig = base_config
            best_iteration: int = 0
            agent_run_id = (
                f"agent-{uuid.uuid4().hex[:8]}" if use_llm_agents else None
            )

            profile = None  # Reused across iterations

            for iteration in range(max_iterations):
                task.current_iteration = iteration + 1
                iter_t0 = time.monotonic()
                iter_started_at = datetime.now()
                iter_phases: dict[str, Any] = {}

                # Phase: Profiling (only first iteration or if LLM agents enabled)
                if use_llm_agents:
                    task.current_phase = "profiling"
                    task.progress = self._calc_progress(
                        iteration, max_iterations, 2.0
                    )
                    task.message = (
                        f"Iteration {iteration + 1}/{max_iterations}: "
                        f"Profiling data"
                    )

                    if profile is None:
                        phase_t0 = time.monotonic()
                        try:
                            from app.services.ml_agents.profiler import profiler

                            profile = await profiler.analyze(
                                train_feature_df,
                                market,
                                symbols,
                            )
                            iter_phases["profiler"] = {
                                "status": "completed",
                                "duration_ms": round(
                                    (time.monotonic() - phase_t0) * 1000
                                ),
                                "summary": (
                                    f"{market.upper()} {len(symbols)} stocks, "
                                    f"regime: {profile.regime_analysis[:50]}"
                                ),
                            }
                        except Exception as e:
                            logger.warning("Profiler failed: %s", e)
                            iter_phases["profiler"] = {
                                "status": "failed",
                                "error": str(e),
                            }

                # Phase: Strategist
                current_config = base_config
                if use_llm_agents:
                    task.current_phase = "strategist"
                    task.progress = self._calc_progress(
                        iteration, max_iterations, 5.0
                    )
                    task.message = (
                        f"Iteration {iteration + 1}/{max_iterations}: "
                        f"Generating config"
                    )

                    phase_t0 = time.monotonic()
                    try:
                        from app.services.ml_agents.strategist import strategist

                        # On retry iterations, pass evaluator feedback
                        prev_eval = None
                        if iteration > 0 and task.iterations:
                            last_iter = task.iterations[-1]
                            eval_phase = last_iter.get("phases", {}).get(
                                "evaluator", {}
                            )
                            if eval_phase.get("decision") in (
                                "retry",
                                "reject",
                            ):
                                prev_eval = {
                                    "decision": eval_phase.get("decision"),
                                    "reasoning": eval_phase.get(
                                        "reasoning", ""
                                    ),
                                    "suggested_adjustments": eval_phase.get(
                                        "suggested_adjustments", {}
                                    ),
                                    "val_ic": eval_phase.get("val_ic"),
                                    "val_spread": eval_phase.get("val_spread"),
                                }

                        tc = await strategist.generate(
                            profile,
                            market,
                            previous_evaluation=prev_eval,
                        )
                        current_config = tc.to_market_config()

                        config_changes = []
                        base_dict = dataclasses.asdict(base_config)
                        new_dict = dataclasses.asdict(current_config)
                        for k, v in new_dict.items():
                            if base_dict.get(k) != v:
                                config_changes.append(
                                    f"{k}: {base_dict.get(k)}->{v}"
                                )

                        iter_phases["strategist"] = {
                            "status": "completed",
                            "duration_ms": round(
                                (time.monotonic() - phase_t0) * 1000
                            ),
                            "config_changes": config_changes,
                            "reasoning": tc.reasoning[:300],
                        }
                    except Exception as e:
                        logger.warning(
                            "Strategist failed, using base config: %s", e
                        )
                        iter_phases["strategist"] = {
                            "status": "failed",
                            "error": str(e),
                        }

                # Phase: Training
                task.current_phase = "training"
                task.progress = self._calc_progress(
                    iteration, max_iterations, 15.0
                )
                task.message = (
                    f"Iteration {iteration + 1}/{max_iterations}: "
                    f"Training model"
                )

                phase_t0 = time.monotonic()
                try:
                    logger.info(
                        "Iteration %d training with config: "
                        "lgb_overrides=%s, boost=%d, early_stop=%d",
                        iteration + 1,
                        current_config.lgb_overrides,
                        current_config.num_boost_round,
                        current_config.early_stopping_rounds,
                    )
                    train_result = await prediction_service.train_for_backtest(
                        market=market,
                        symbols=symbols,
                        forward_days=forward_days,
                        cutoff_date=cutoff_date,
                        config=current_config,
                        feature_df=train_feature_df,
                    )
                    iter_phases["training"] = {
                        "status": "completed",
                        "duration_seconds": round(
                            time.monotonic() - phase_t0, 1
                        ),
                        "fold_ics": [
                            round(ic, 4) for ic in train_result["fold_ics"]
                        ],
                        "mean_ic": round(train_result["ic"], 4),
                        "feature_count": train_result["feature_count"],
                    }
                except Exception as e:
                    logger.error(
                        "Training failed in iteration %d: %s",
                        iteration + 1,
                        e,
                    )
                    iter_phases["training"] = {
                        "status": "failed",
                        "error": str(e),
                    }
                    # Record failed iteration and continue
                    task.iterations.append(
                        {
                            "iteration": iteration + 1,
                            "started_at": iter_started_at.isoformat(),
                            "completed_at": datetime.now().isoformat(),
                            "duration_seconds": round(
                                time.monotonic() - iter_t0, 1
                            ),
                            "phases": iter_phases,
                        }
                    )
                    continue

                # Phase: Inference on validation dates
                task.current_phase = "inference"
                task.progress = self._calc_progress(
                    iteration, max_iterations, 60.0
                )
                task.message = (
                    f"Iteration {iteration + 1}/{max_iterations}: "
                    f"Running inference on {len(val_trading_dates)} dates"
                )

                phase_t0 = time.monotonic()
                models = train_result["models"]
                feature_cols = train_result["feature_cols"]

                predictions_rows: list[dict[str, Any]] = []
                for d in val_trading_dates:
                    day_df = full_feature_df[full_feature_df["date"] == d]
                    if day_df.empty:
                        continue
                    # Align columns -- add any missing feature columns as NaN
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
                    for sym, score in zip(
                        day_df["symbol"].values, scores
                    ):
                        predictions_rows.append(
                            {
                                "date": d,
                                "symbol": sym,
                                "predicted_score": float(score),
                            }
                        )

                predictions_df = pd.DataFrame(predictions_rows)
                iter_phases["inference"] = {
                    "status": "completed",
                    "duration_seconds": round(time.monotonic() - phase_t0, 1),
                    "prediction_dates": len(val_trading_dates),
                    "total_predictions": len(predictions_rows),
                    "avg_symbols_per_date": round(
                        len(predictions_rows)
                        / max(len(val_trading_dates), 1)
                    ),
                }

                # Phase: Compute validation metrics
                task.current_phase = "computing_metrics"
                task.progress = self._calc_progress(
                    iteration, max_iterations, 80.0
                )
                task.message = (
                    f"Iteration {iteration + 1}/{max_iterations}: "
                    f"Computing validation metrics"
                )

                val_metrics = self._compute_validation_metrics(
                    predictions_df,
                    close_df,
                    forward_days,
                    val_trading_dates,
                )

                # Phase: Evaluator (LLM)
                evaluator_result = None
                if use_llm_agents:
                    task.current_phase = "evaluating"
                    task.progress = self._calc_progress(
                        iteration, max_iterations, 90.0
                    )
                    task.message = (
                        f"Iteration {iteration + 1}/{max_iterations}: "
                        f"Evaluating results"
                    )

                    phase_t0 = time.monotonic()
                    try:
                        from app.services.ml_agents.evaluator import evaluator

                        # Build feature importance top-20
                        fi = train_result.get("feature_importance", {})
                        fi_top20 = [
                            {"feature": k, "gain": round(v, 4)}
                            for k, v in list(fi.items())[:20]
                        ]

                        evaluator_result = await evaluator.evaluate(
                            market=market,
                            training_results={
                                "ic": train_result["ic"],
                                "icir": train_result["icir"],
                                "ndcg_at_10": train_result.get("ndcg"),
                                "fold_ics": train_result["fold_ics"],
                                "best_iters": train_result["best_iters"],
                                "feature_importance_top20": fi_top20,
                                "val_ic": val_metrics.get("val_ic"),
                                "val_icir": val_metrics.get("val_icir"),
                                "val_direction_accuracy": val_metrics.get(
                                    "val_direction_accuracy"
                                ),
                                "val_spread": val_metrics.get("val_spread"),
                            },
                            training_config=dataclasses.asdict(current_config),
                            data_profile={
                                "regime_analysis": (
                                    profile.regime_analysis if profile else ""
                                ),
                                "data_quality_warnings": (
                                    profile.data_quality_warnings
                                    if profile
                                    else []
                                ),
                                "universe_size": len(symbols),
                            },
                            quality_thresholds={
                                "min_ic": current_config.min_ic_threshold,
                                "min_icir": current_config.min_icir_threshold,
                            },
                            is_retry=iteration > 0,
                        )

                        iter_phases["evaluator"] = {
                            "status": "completed",
                            "duration_ms": round(
                                (time.monotonic() - phase_t0) * 1000
                            ),
                            "decision": evaluator_result.decision,
                            "reasoning": evaluator_result.reasoning[:300],
                            "suggested_adjustments": (
                                evaluator_result.suggested_adjustments
                            ),
                            "confidence": evaluator_result.confidence,
                            "val_ic": val_metrics.get("val_ic"),
                            "val_spread": val_metrics.get("val_spread"),
                        }
                    except Exception as e:
                        logger.warning("Evaluator failed: %s", e)
                        iter_phases["evaluator"] = {
                            "status": "failed",
                            "error": str(e),
                        }

                # Record iteration
                task.iterations.append(
                    {
                        "iteration": iteration + 1,
                        "started_at": iter_started_at.isoformat(),
                        "completed_at": datetime.now().isoformat(),
                        "duration_seconds": round(
                            time.monotonic() - iter_t0, 1
                        ),
                        "phases": iter_phases,
                    }
                )

                # Track best result
                current_val_ic = val_metrics.get("val_ic", -999.0)
                if (
                    current_val_ic is not None
                    and current_val_ic > best_val_ic
                ):
                    best_val_ic = current_val_ic
                    best_result = {**train_result, **val_metrics}
                    best_config = current_config
                    best_iteration = iteration + 1

                # Check evaluator decision
                if evaluator_result:
                    if evaluator_result.decision == "deploy":
                        logger.info(
                            "Evaluator says deploy at iteration %d",
                            iteration + 1,
                        )
                        break
                    elif evaluator_result.decision == "reject":
                        if iteration < max_iterations - 1:
                            # Non-final iteration: treat reject as retry --
                            # the whole point of the loop is to iterate.
                            # Strategist reads feedback from
                            # task.iterations[-1].phases.evaluator
                            logger.info(
                                "Evaluator says reject at iteration %d, "
                                "but %d iterations remain -- will retry",
                                iteration + 1,
                                max_iterations - iteration - 1,
                            )
                        else:
                            logger.info(
                                "Evaluator says reject at final iteration %d",
                                iteration + 1,
                            )
                            break
                    # else "retry" -- continue loop; strategist reads
                    # feedback from task.iterations[-1].phases.evaluator
                elif not use_llm_agents:
                    # No evaluator (non-LLM mode) -- single iteration
                    break
                else:
                    # Evaluator failed (exception) -- continue loop if
                    # iterations remain, don't let a transient LLM error
                    # terminate the optimization loop
                    if iteration < max_iterations - 1:
                        logger.warning(
                            "Evaluator failed at iteration %d, "
                            "continuing to next iteration",
                            iteration + 1,
                        )
                    else:
                        logger.warning(
                            "Evaluator failed at final iteration %d",
                            iteration + 1,
                        )

            # Store best result
            if best_result is None:
                raise RuntimeError(
                    "All iterations failed -- no valid result"
                )

            task.current_phase = "storing"
            task.progress = 95.0
            task.message = "Saving results"

            duration = time.monotonic() - t0
            await self._save_result(
                task,
                best_result,
                best_config,
                duration,
                agent_run_id,
                best_iteration,
            )

            # Determine quality verdict based on market thresholds
            quality_passed = best_val_ic > 0
            quality_tag = "PASS" if quality_passed else "FAIL"

            task.status = "completed"
            task.progress = 100.0
            task.message = (
                f"Completed [{quality_tag}] -- "
                f"val_IC={best_val_ic:.4f} "
                f"(best iteration {best_iteration})"
            )
            task.completed_at = datetime.now()

            logger.info(
                "Backtest %s completed: val_IC=%.4f, iterations=%d, "
                "duration=%.0fs",
                task.task_id,
                best_val_ic,
                len(task.iterations),
                duration,
            )

        except Exception as e:
            logger.error(
                "Backtest %s failed: %s", task.task_id, e, exc_info=True
            )
            task.status = "failed"
            task.error = str(e)
            task.completed_at = datetime.now()
            duration = time.monotonic() - t0
            await self._mark_failed(task.backtest_id, str(e), duration)
        finally:
            self._running_markets.discard(market)

    # --- Helper methods ---

    @staticmethod
    def _calc_progress(
        iteration: int, max_iterations: int, phase_pct: float
    ) -> float:
        """Calculate overall progress percentage."""
        per_iter = 100.0 / max_iterations
        return min(
            99.9, iteration * per_iter + phase_pct * per_iter / 100.0
        )

    async def _resolve_symbols(self, market: str) -> list[str]:
        """Resolve universe symbols (reuse production logic).

        Delegates to prediction_service._resolve_symbols which handles
        the full priority chain: explicit symbols -> index constituents
        -> full-market fallback via StockPulse.
        """
        symbols = await prediction_service._resolve_symbols(market)
        if not symbols:
            raise RuntimeError(
                f"No symbols resolved for market={market}. "
                f"Configure a prediction universe or ensure "
                f"StockPulse can return symbols."
            )
        return symbols

    def _compute_validation_metrics(
        self,
        predictions_df: pd.DataFrame,
        close_df: pd.DataFrame,
        forward_days: int,
        val_dates: list,
    ) -> dict[str, Any]:
        """Compute validation metrics from predictions vs actual returns."""
        if predictions_df.empty:
            return {"val_ic": None, "val_icir": None}

        # Compute actual forward returns
        close_sorted = close_df.sort_values(["symbol", "date"]).copy()
        close_sorted["actual_fwd_return"] = close_sorted.groupby("symbol")[
            "close"
        ].transform(lambda x: x.shift(-forward_days) / x - 1)

        # Merge predictions with actuals
        merged = predictions_df.merge(
            close_sorted[["symbol", "date", "actual_fwd_return"]],
            on=["symbol", "date"],
            how="left",
        )
        merged = merged.dropna(
            subset=["actual_fwd_return", "predicted_score"]
        )

        if len(merged) < 50:
            logger.warning(
                "Too few prediction-actual pairs: %d", len(merged)
            )
            return {"val_ic": None, "val_icir": None}

        # Per-date IC (Spearman rank correlation)
        from scipy.stats import spearmanr

        ic_per_date: list[dict[str, Any]] = []
        for d, group in merged.groupby("date"):
            if len(group) < 10:
                continue
            corr, _ = spearmanr(
                group["predicted_score"], group["actual_fwd_return"]
            )
            if not np.isnan(corr):
                ic_per_date.append({"date": str(d), "ic": round(corr, 6)})

        if not ic_per_date:
            return {"val_ic": None, "val_icir": None}

        ic_series = [x["ic"] for x in ic_per_date]
        val_ic = float(np.mean(ic_series))
        val_icir = (
            float(np.mean(ic_series) / np.std(ic_series))
            if np.std(ic_series) > 0
            else 0.0
        )

        # Quintile analysis
        merged["quintile"] = merged.groupby("date")[
            "predicted_score"
        ].transform(
            lambda x: pd.qcut(
                x.rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]
            )
        )

        q_returns = merged.groupby("quintile")["actual_fwd_return"].mean()
        q1_return = float(q_returns.get(1, 0))  # worst predicted
        q5_return = float(q_returns.get(5, 0))  # best predicted
        val_spread = q5_return - q1_return

        # Direction accuracy: predicted top quintile actually positive?
        top_q = merged[merged["quintile"] == 5]
        direction_acc = (
            float((top_q["actual_fwd_return"] > 0).mean())
            if len(top_q) > 0
            else None
        )

        # Hit rate: predicted top quintile beats median?
        median_return = merged.groupby("date")[
            "actual_fwd_return"
        ].transform("median")
        merged["beats_median"] = merged["actual_fwd_return"] > median_return
        top_q_beats = merged[merged["quintile"] == 5]["beats_median"]
        hit_rate = (
            float(top_q_beats.mean()) if len(top_q_beats) > 0 else None
        )

        # Max drawdown of top quintile cumulative returns
        max_dd = None
        if len(top_q) > 0:
            daily_rets = (
                top_q.groupby("date")["actual_fwd_return"]
                .mean()
                .sort_index()
            )
            if len(daily_rets) > 1:
                cum = (1 + daily_rets).cumprod()
                peak = cum.expanding().max()
                drawdown = (cum - peak) / peak
                max_dd = float(drawdown.min())

        return {
            "val_ic": round(val_ic, 6),
            "val_icir": round(val_icir, 4),
            "val_direction_accuracy": (
                round(direction_acc, 4) if direction_acc is not None else None
            ),
            "val_spread": round(val_spread, 6),
            "val_q1_return": round(q1_return, 6),
            "val_q5_return": round(q5_return, 6),
            "val_hit_rate": (
                round(hit_rate, 4) if hit_rate is not None else None
            ),
            "val_max_drawdown": (
                round(max_dd, 4) if max_dd is not None else None
            ),
            "ic_curve": ic_per_date,
            "quintile_returns": {
                int(k): round(float(v), 6) for k, v in q_returns.items()
            },
        }

    @staticmethod
    def _compute_cumulative_returns(
        predictions_df: pd.DataFrame,
        close_df: pd.DataFrame,
        forward_days: int,
    ) -> dict[str, list[dict[str, Any]]]:
        """Compute daily cumulative returns for Q5, Q1, and spread."""
        close_sorted = close_df.sort_values(["symbol", "date"]).copy()
        close_sorted["fwd_ret"] = close_sorted.groupby("symbol")[
            "close"
        ].transform(lambda x: x.shift(-forward_days) / x - 1)

        merged = predictions_df.merge(
            close_sorted[["symbol", "date", "fwd_ret"]],
            on=["symbol", "date"],
            how="left",
        ).dropna(subset=["fwd_ret", "predicted_score"])

        if len(merged) < 50:
            return {"q5": [], "q1": [], "spread": []}

        # Assign quintiles per date
        merged["quintile"] = merged.groupby("date")[
            "predicted_score"
        ].transform(
            lambda x: pd.qcut(
                x.rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]
            )
        )

        # Daily mean return per quintile
        q5_daily = (
            merged[merged["quintile"] == 5]
            .groupby("date")["fwd_ret"]
            .mean()
            .sort_index()
        )
        q1_daily = (
            merged[merged["quintile"] == 1]
            .groupby("date")["fwd_ret"]
            .mean()
            .sort_index()
        )

        # Align dates
        common_dates = q5_daily.index.intersection(q1_daily.index)
        q5_daily = q5_daily.loc[common_dates]
        q1_daily = q1_daily.loc[common_dates]
        spread_daily = q5_daily - q1_daily

        # Cumulative returns
        q5_cum = (1 + q5_daily).cumprod() - 1
        q1_cum = (1 + q1_daily).cumprod() - 1
        spread_cum = (1 + spread_daily).cumprod() - 1

        def _series_to_list(s: pd.Series) -> list[dict[str, Any]]:
            return [
                {"date": str(d)[:10], "cumret": round(float(v), 6)}
                for d, v in s.items()
            ]

        return {
            "q5": _series_to_list(q5_cum),
            "q1": _series_to_list(q1_cum),
            "spread": _series_to_list(spread_cum),
        }

    async def _save_result(
        self,
        task: BacktestTask,
        result: dict[str, Any],
        config: MarketConfig,
        duration: float,
        agent_run_id: str | None,
        agent_iteration: int,
    ) -> None:
        """Persist backtest results to DB."""
        try:
            pool = get_db_pool()
        except RuntimeError:
            logger.warning("No DB pool, cannot save backtest result")
            return

        # Build results JSON (exclude models -- not serializable)
        results_json = {
            k: v
            for k, v in result.items()
            if k not in ("models", "feature_cols") and v is not None
        }

        try:
            async with pool.acquire(timeout=10) as conn:
                await conn.execute(
                    _SQL_UPDATE_BACKTEST_COMPLETED,
                    uuid.UUID(task.backtest_id),
                    result.get("ic"),
                    result.get("icir"),
                    result.get("ndcg"),
                    json.dumps(result.get("fold_ics", [])),
                    result.get("ensemble_size"),
                    result.get("feature_count"),
                    result.get("symbol_count"),
                    result.get("val_ic"),
                    result.get("val_icir"),
                    result.get("val_direction_accuracy"),
                    result.get("val_spread"),
                    result.get("val_q1_return"),
                    result.get("val_q5_return"),
                    result.get("val_hit_rate"),
                    result.get("val_max_drawdown"),
                    json.dumps(results_json, default=str),
                    json.dumps(dataclasses.asdict(config)),
                    duration,
                    agent_run_id,
                    agent_iteration,
                )
        except Exception as e:
            logger.error("Failed to save backtest result: %s", e)

    async def _update_db_status(
        self, backtest_id: str, status: str
    ) -> None:
        try:
            pool = get_db_pool()
        except RuntimeError:
            return
        try:
            async with pool.acquire(timeout=10) as conn:
                await conn.execute(
                    "UPDATE ml_backtests SET status = $2 WHERE id = $1::uuid",
                    uuid.UUID(backtest_id),
                    status,
                )
        except Exception as e:
            logger.error("Failed to update backtest status to %s: %s", status, e)

    async def _mark_failed(
        self, backtest_id: str, error: str, duration: float
    ) -> None:
        try:
            pool = get_db_pool()
        except RuntimeError:
            return
        try:
            async with pool.acquire(timeout=10) as conn:
                await conn.execute(
                    _SQL_UPDATE_BACKTEST_FAILED,
                    uuid.UUID(backtest_id),
                    error[:2000],
                    duration,
                )
        except Exception as e:
            logger.error("Failed to mark backtest as failed: %s", e)


# Module singleton
backtest_service = MLBacktestService()
