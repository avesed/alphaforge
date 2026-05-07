"""LightGBM prediction service -- training + inference engine.

Training pipeline:
1. QlibContext.ensure_init(market)
2. Resolve universe symbols via StockPulse API
3. feature_service.build_feature_matrix() -> ~87 features x N stocks x T days
4. Label engineering: forward N-day return -> percentile score (continuous target)
5. Purged time-series split (train/val) with gap = forward_days
6. LightGBM training: objective='lambdarank', early_stopping_rounds=50
7. joblib.dump() model -> /app/data/predictions/{market}/{YYYYMMDD}/model.pkl
8. Evaluate IC/ICIR/NDCG -> write to prediction_models via StockPulse API

Inference pipeline:
1. joblib.load() latest model
2. feature_service.build_feature_matrix() for latest date
3. Predict -> score -> cross-sectional rank -> direction
4. Write to stock_predictions via StockPulse API + Redis cache (24h)

AlphaForge adaptation:
- Prediction/model read/write operations go through local PredictionStore.
- Market data (prices, fundamentals, sectors) still via StockPulseAsyncClient.
- Model files remain on local disk (joblib.dump/load).
"""

import asyncio
import hashlib
import json
import logging
import math
import os
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import joblib
import lightgbm as lgb
import msgpack
import numpy as np
import pandas as pd
import redis.asyncio as aioredis

from app.config import get_settings
from app.services.feature_service import (
    ALPHA158_FEATURES,
    FUNDAMENTAL_FEATURES,
    SENTIMENT_FEATURES,
    feature_service,
)
from app.services.market_config import MarketConfig, get_market_config
from app.services.prediction_store import prediction_store
from app.services.stockpulse_client import get_stockpulse_async_client

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PREDICTION_CACHE_TTL = 86400
_MAX_CONCURRENT_PREDICTIONS = 1
_TRAIN_LOOKBACK_DAYS = 730
_MIN_TRAIN_DATES = 60
_MIN_SYMBOLS_PER_DATE = 25

DIRECTION_UP_THRESHOLD = 0.70
DIRECTION_DOWN_THRESHOLD = 0.30

_ENSEMBLE_SEEDS: list[int] = [42, 137, 271, 419, 503, 631, 769, 887, 953, 1031]

_BASE_LGB_PARAMS: dict[str, Any] = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [5, 10, 20],
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbose": -1,
}


def _get_lgb_params(market: str, cfg: MarketConfig | None = None) -> dict[str, Any]:
    params = dict(_BASE_LGB_PARAMS)
    resolved = cfg or get_market_config(market)
    params.update(resolved.lgb_overrides)
    return params


def _get_boost_round(market: str, cfg: MarketConfig | None = None) -> int:
    return (cfg or get_market_config(market)).num_boost_round


def _get_early_stopping(market: str, cfg: MarketConfig | None = None) -> int:
    return (cfg or get_market_config(market)).early_stopping_rounds


def _get_prediction_horizons() -> list[int]:
    raw = get_settings().PREDICTION_HORIZONS
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _numpy_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if math.isnan(v) or math.isinf(v) else v
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _safe_round(val: float, decimals: int) -> float | None:
    if math.isnan(val) or math.isinf(val):
        return None
    return round(val, decimals)


def _rank_auc(probs: np.ndarray, labels: np.ndarray) -> float | None:
    from scipy.stats import rankdata
    pos = np.where(labels == 1)[0]
    neg = np.where(labels == 0)[0]
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = rankdata(probs)
    auc = (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


_redis_client: Optional[aioredis.Redis] = None
_redis_lock = asyncio.Lock()


async def _get_redis_client() -> aioredis.Redis:
    global _redis_client
    if _redis_client is None:
        async with _redis_lock:
            if _redis_client is None:
                settings = get_settings()
                _redis_client = aioredis.from_url(
                    settings.REDIS_URL, decode_responses=False
                )
    return _redis_client


def _prediction_cache_key(market: str) -> str:
    return f"pred:latest:{market}"


# ---------------------------------------------------------------------------
# Task dataclass
# ---------------------------------------------------------------------------


@dataclass
class PredictionTask:
    task_id: str
    market: str
    status: str = "pending"
    progress: float = 0.0
    message: str = ""
    results: Optional[dict] = None
    error: Optional[str] = None
    _asyncio_task: Optional[asyncio.Task] = field(
        default=None, repr=False, compare=False
    )
    _psi_data: Optional[dict] = field(
        default=None, repr=False, compare=False
    )
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        d = {
            "task_id": self.task_id,
            "market": self.market,
            "status": self.status,
            "progress": self.progress,
            "message": self.message,
            "results": self.results,
            "error": self.error,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
        }
        return json.loads(json.dumps(d, default=_numpy_default))


# ---------------------------------------------------------------------------
# PredictionService
# ---------------------------------------------------------------------------


class PredictionService:
    """LightGBM prediction service -- training and inference engine.

    All data access goes through StockPulseAsyncClient.
    Model files are stored on local disk.
    """

    _TASK_MAX_AGE_SECONDS = 3600

    def __init__(self) -> None:
        self._tasks: dict[str, PredictionTask] = {}
        self._lock = asyncio.Lock()

    def _cleanup_old_tasks(self) -> None:
        now = datetime.now()
        to_delete = [
            tid
            for tid, t in self._tasks.items()
            if t.status in ("completed", "failed")
            and t.completed_at is not None
            and (now - t.completed_at).total_seconds() > self._TASK_MAX_AGE_SECONDS
        ]
        for tid in to_delete:
            self._tasks.pop(tid, None)
        if to_delete:
            logger.debug("Cleaned up %d old prediction tasks", len(to_delete))

    # ------------------------------------------------------------------
    # Public API: task management
    # ------------------------------------------------------------------

    async def run_prediction(
        self,
        market: str,
        force_retrain: bool = False,
        forward_days: int = 5,
    ) -> str:
        async with self._lock:
            self._cleanup_old_tasks()
            running = sum(
                1
                for t in self._tasks.values()
                if t.status in ("pending", "training", "predicting")
            )
            if running >= _MAX_CONCURRENT_PREDICTIONS:
                raise RuntimeError(
                    f"Maximum concurrent predictions ({_MAX_CONCURRENT_PREDICTIONS}) "
                    f"reached. Wait for existing task to complete."
                )
            task_id = uuid.uuid4().hex[:16]
            task = PredictionTask(task_id=task_id, market=market)
            self._tasks[task_id] = task

        logger.info(
            "Prediction task created: task_id=%s, market=%s, "
            "force_retrain=%s, forward_days=%d",
            task_id, market, force_retrain, forward_days,
        )

        coro = self._run_prediction_async(task, market, forward_days, force_retrain)
        bg_task = asyncio.create_task(coro, name=f"predict-{task_id}")
        task._asyncio_task = bg_task
        return task_id

    async def run_multi_horizon(
        self,
        market: str,
        force_retrain: bool = False,
    ) -> str:
        horizons = _get_prediction_horizons()
        async with self._lock:
            self._cleanup_old_tasks()
            running = sum(
                1
                for t in self._tasks.values()
                if t.status in ("pending", "training", "predicting")
            )
            if running >= _MAX_CONCURRENT_PREDICTIONS:
                raise RuntimeError(
                    f"Maximum concurrent predictions ({_MAX_CONCURRENT_PREDICTIONS}) "
                    f"reached. Wait for existing task to complete."
                )
            task_id = uuid.uuid4().hex[:16]
            task = PredictionTask(task_id=task_id, market=market)
            self._tasks[task_id] = task

        logger.info(
            "Multi-horizon prediction task created: task_id=%s, market=%s, "
            "horizons=%s, force_retrain=%s",
            task_id, market, horizons, force_retrain,
        )

        coro = self._run_multi_horizon_async(task, market, horizons, force_retrain)
        bg_task = asyncio.create_task(coro, name=f"predict-multi-{task_id}")
        task._asyncio_task = bg_task
        return task_id

    async def _run_multi_horizon_async(
        self,
        task: PredictionTask,
        market: str,
        horizons: list[int],
        force_retrain: bool,
    ) -> None:
        n_horizons = len(horizons)
        prediction_date = date.today()
        try:
            for i, h in enumerate(horizons):
                pct_base = (i / n_horizons) * 90
                task.message = f"Horizon {h}d ({i + 1}/{n_horizons})"
                task.progress = pct_base
                logger.info(
                    "Multi-horizon: starting horizon %dd (%d/%d) for %s",
                    h, i + 1, n_horizons, market,
                )
                await self._run_prediction_async(
                    task, market, h, force_retrain,
                    _progress_base=pct_base,
                    _progress_range=90.0 / n_horizons,
                    _skip_completion=True,
                )

            # Direction model (non-fatal)
            task.message = "Training direction model"
            task.progress = 95.0
            dir_result = await self._run_direction_step(
                task, market, force_retrain=force_retrain,
                prediction_date=prediction_date,
            )

            task.status = "completed"
            task.progress = 100.0
            task.completed_at = datetime.now()
            task.message = f"Completed: {n_horizons} horizons"
            task.results = task.results or {}
            task.results["horizons"] = horizons
            if dir_result:
                task.results["direction_model"] = dir_result

        except Exception as e:
            logger.error(
                "Multi-horizon prediction failed for %s: %s",
                market, e, exc_info=True,
            )
            task.status = "failed"
            task.error = str(e)
            task.completed_at = datetime.now()

    def get_task(self, task_id: str) -> Optional[dict]:
        task = self._tasks.get(task_id)
        if task is None:
            return None
        return task.to_dict()

    def list_tasks(self) -> list[dict]:
        tasks = sorted(
            self._tasks.values(),
            key=lambda t: t.created_at,
            reverse=True,
        )
        return [t.to_dict() for t in tasks]

    async def cancel_task(self, task_id: str) -> bool:
        task = self._tasks.get(task_id)
        if task is None:
            return False
        if task.status not in ("pending", "training", "predicting"):
            return False
        task.status = "failed"
        task.error = "Cancelled by user"
        task.completed_at = datetime.now()
        if task._asyncio_task is not None and not task._asyncio_task.done():
            task._asyncio_task.cancel()
        logger.info("Prediction task cancelled: task_id=%s", task_id)
        return True

    def delete_task(self, task_id: str) -> bool:
        task = self._tasks.get(task_id)
        if task is None:
            return False
        if task.status in ("pending", "training", "predicting"):
            return False
        del self._tasks[task_id]
        logger.info("Prediction task deleted: task_id=%s", task_id)
        return True

    # ------------------------------------------------------------------
    # Direction model integration
    # ------------------------------------------------------------------

    async def _run_direction_step(
        self,
        task: "PredictionTask",
        market: str,
        force_retrain: bool = False,
        prediction_date: Optional[date] = None,
    ) -> Optional[dict]:
        try:
            from app.services.direction_service import train_and_predict_direction

            prev_msg = task.message
            task.message = "Training direction model"

            result = await train_and_predict_direction(
                market, force_retrain=force_retrain,
                prediction_date=prediction_date,
            )

            if result:
                n_updated = result.get("predictions_updated", 0)
                logger.info(
                    "Direction model for %s: updated=%d, quality_passed=%s",
                    market, n_updated, result.get("quality_passed"),
                )
                if n_updated > 0:
                    await self._refresh_prediction_cache(market)
                return {
                    "predictions_updated": n_updated,
                    "quality_passed": result.get("quality_passed"),
                    "auc": result.get("auc"),
                    "brier_score": result.get("brier_score"),
                }

            task.message = prev_msg
            return None

        except Exception as e:
            logger.error(
                "Direction model failed for %s (non-fatal): %s",
                market, e, exc_info=True,
            )
            return None

    # ------------------------------------------------------------------
    # Public API: predictions query (via StockPulse)
    # ------------------------------------------------------------------

    async def get_latest_predictions(
        self,
        market: str,
        top_n: int = 50,
        symbol: Optional[str] = None,
        forward_days: Optional[int] = None,
    ) -> list[dict]:
        # 1. Try Redis cache (only when no filters applied)
        if symbol is None and forward_days is None:
            cached = await self._read_prediction_cache(market)
            if cached is not None:
                logger.debug("Prediction cache hit: market=%s", market)
                return cached[:top_n]

        # 2. Query via local PredictionStore
        try:
            return await prediction_store.get_latest_predictions(
                market, top_n, symbol=symbol, forward_days=forward_days,
            )
        except Exception as e:
            logger.error("Failed to query latest predictions: %s", e)
            return []

    async def get_models(self, market: Optional[str] = None) -> list[dict]:
        try:
            return await prediction_store.get_models(market)
        except Exception as e:
            logger.error("Failed to query prediction models: %s", e)
            return []

    async def get_prediction_history(
        self, market: str, days: int = 30
    ) -> list[dict]:
        try:
            return await prediction_store.get_prediction_history(market, days)
        except Exception as e:
            logger.error("Failed to query prediction history: %s", e)
            return []

    async def backfill_returns(self) -> dict:
        """Backfill actual returns for past predictions via local PredictionStore."""
        try:
            # Get candidates from all markets
            all_updates: list[dict] = []
            for mkt in ("us", "hk", "cn"):
                candidates = await prediction_store.get_predictions_for_backfill(mkt)
                if not candidates:
                    continue

                # Group by (symbol, market) for batched price lookups
                symbol_groups: dict[tuple[str, str], list[dict]] = {}
                for row in candidates:
                    key = (row["symbol"], row["market"])
                    if key not in symbol_groups:
                        symbol_groups[key] = []
                    symbol_groups[key].append(row)

                for (symbol, market), predictions in symbol_groups.items():
                    try:
                        actual_returns = await self._compute_actual_returns(
                            symbol, predictions, market=market
                        )
                        for pred in predictions:
                            ret = actual_returns.get(pred["id"])
                            if ret is not None:
                                all_updates.append({
                                    "market": pred["market"],
                                    "symbol": pred["symbol"],
                                    "prediction_date": pred["prediction_date"],
                                    "forward_days": pred.get("forward_days", 5),
                                    "actual_return": float(ret),
                                })
                    except Exception as e:
                        logger.warning(
                            "Backfill price fetch failed for %s (%s): %s",
                            symbol, market, e,
                        )

            if all_updates:
                updated = await prediction_store.backfill_returns(all_updates)
                logger.info("Backfill complete: %d updates sent", len(all_updates))
                return {"updated": updated, "failed": 0, "total": len(all_updates)}
            else:
                logger.info("No predictions need backfilling")
                return {"updated": 0, "failed": 0, "total": 0}

        except Exception as e:
            logger.error("Backfill failed: %s", e)
            return {"error": str(e)}

    async def check_retrain_needed(self, market: str) -> bool:
        try:
            r = await _get_redis_client()
            key = f"prediction:retrain_needed:{market}"
            val = await r.get(key)
            if val:
                await r.delete(key)
                return True
        except Exception:
            pass
        return False

    async def _flag_retrain_needed(self, market: str) -> None:
        try:
            r = await _get_redis_client()
            key = f"prediction:retrain_needed:{market}"
            await r.set(key, b"1", ex=86400 * 2)
            logger.info("Flagged %s for forced retrain", market)
        except Exception as e:
            logger.debug("Failed to set retrain flag for %s: %s", market, e)

    async def check_direction_retrain_needed(self, market: str) -> bool:
        try:
            r = await _get_redis_client()
            key = f"prediction:direction_retrain:{market}"
            val = await r.get(key)
            if val:
                await r.delete(key)
                return True
        except Exception:
            pass
        return False

    async def _flag_direction_retrain(self, market: str) -> None:
        try:
            r = await _get_redis_client()
            key = f"prediction:direction_retrain:{market}"
            await r.set(key, b"1", ex=86400 * 2)
            logger.info("Flagged %s direction model for forced retrain", market)
        except Exception as e:
            logger.debug("Failed to set direction retrain flag for %s: %s", market, e)

    def shutdown(self) -> None:
        for task in self._tasks.values():
            if task._asyncio_task and not task._asyncio_task.done():
                task._asyncio_task.cancel()
                logger.info(
                    "Cancelled prediction task on shutdown: %s", task.task_id
                )

    # ------------------------------------------------------------------
    # Core async pipeline
    # ------------------------------------------------------------------

    async def _run_prediction_async(
        self,
        task: PredictionTask,
        market: str,
        forward_days: int,
        force_retrain: bool,
        _progress_base: float = 0.0,
        _progress_range: float = 100.0,
        _skip_completion: bool = False,
    ) -> None:
        def _p(pct: float) -> float:
            return _progress_base + (pct / 100.0) * _progress_range

        try:
            # Step 1: Resolve universe symbols
            task.status = "training"
            task.progress = _p(5)
            task.message = f"[{forward_days}d] Resolving universe symbols"

            symbols = await self._resolve_symbols(market)
            if not symbols:
                raise RuntimeError(
                    f"No symbols resolved for market={market}."
                )

            logger.info("Resolved %d symbols for market=%s", len(symbols), market)
            task.message = f"[{forward_days}d] Resolved {len(symbols)} symbols"
            task.progress = _p(10)

            # Step 1.5: Data freshness check
            is_fresh, freshness_msg = self._check_data_freshness(market)
            if not is_fresh:
                logger.warning("Skipping prediction for %s: %s", market, freshness_msg)
                if _skip_completion:
                    task.message = f"[{forward_days}d] Skipped: {freshness_msg}"
                    return
                task.status = "completed"
                task.progress = 100.0
                task.completed_at = datetime.now()
                task.message = f"Skipped: {freshness_msg}"
                task.results = {"market": market, "skipped": True, "reason": freshness_msg}
                return

            # Step 2: Check if retraining is needed
            today = date.today()
            model_id: Optional[Any] = None
            model_path: Optional[str] = None
            trained_this_run = False

            if not force_retrain:
                existing = self._check_existing_model_on_disk(market, today, forward_days)
                if existing is not None:
                    model_path = existing
                    logger.info("Existing model found for %s/%s", market, today.isoformat())
                    task.message = f"[{forward_days}d] Using existing model"
                    task.progress = _p(70)

            # Step 3: Train if needed
            quality_passed = True
            if model_path is None:
                model_id, model_path, quality_passed = await self._train_model(
                    task, market, symbols, forward_days, today
                )
                trained_this_run = True

                if not quality_passed:
                    # Fall back to latest model on disk
                    fallback = self._find_latest_model_on_disk(market, forward_days)
                    if fallback:
                        logger.warning(
                            "Quality gate failed -- falling back to previous model: %s",
                            fallback,
                        )
                        model_path = fallback

            # Step 4: Inference
            task.status = "predicting"
            task.progress = _p(75)
            task.message = f"[{forward_days}d] Running inference"

            prediction_count = await self._run_inference(
                task, market, symbols, model_id, model_path, forward_days, today
            )

            if _skip_completion:
                task.progress = _p(100)
                task.message = f"[{forward_days}d] Done: {prediction_count} predictions"
                return

            task.status = "completed"
            task.progress = 100.0
            task.completed_at = datetime.now()
            task.message = (
                f"Completed: {prediction_count} predictions"
                + (" (model retrained)" if trained_this_run else "")
            )
            task.results = {
                "market": market,
                "prediction_count": prediction_count,
                "prediction_date": today.isoformat(),
                "forward_days": forward_days,
                "symbol_count": len(symbols),
                "retrained": trained_this_run,
            }
            if task._psi_data is not None:
                task.results["feature_psi"] = task._psi_data

            # Direction model (non-fatal)
            dir_result = await self._run_direction_step(
                task, market, force_retrain=trained_this_run,
                prediction_date=today,
            )
            if dir_result:
                task.results["direction_model"] = dir_result

            logger.info(
                "Prediction pipeline completed: task_id=%s, market=%s, predictions=%d",
                task.task_id, market, prediction_count,
            )

        except asyncio.CancelledError:
            if _skip_completion:
                raise
            task.status = "failed"
            task.error = "Task was cancelled"
            task.completed_at = datetime.now()
        except Exception as e:
            if _skip_completion:
                raise
            task.status = "failed"
            task.error = str(e)
            task.completed_at = datetime.now()
            logger.error(
                "Prediction task failed: task_id=%s, error=%s",
                task.task_id, e, exc_info=True,
            )

    # ------------------------------------------------------------------
    # Step 1: Symbol resolution (via StockPulse API)
    # ------------------------------------------------------------------

    async def _resolve_symbols(self, market: str) -> list[str]:
        """Resolve the prediction universe for a market.

        Fetches all symbols from StockPulse, then filters to top N by
        average daily trading volume (from Qlib local data) to select
        liquid, well-covered stocks with good data quality.
        """
        settings = get_settings()
        max_size = settings.PREDICTION_UNIVERSE_SIZE

        try:
            client = await get_stockpulse_async_client()
            all_symbols = await client.get_universe_symbols(market)
            if not all_symbols:
                logger.error("Symbol fetch failed for market=%s", market)
                return []
            logger.info(
                "Fetched %d total symbols for market=%s", len(all_symbols), market,
            )
        except Exception as e:
            logger.error("Symbol fetch failed for market=%s: %s", market, e)
            return []

        if len(all_symbols) <= max_size:
            return all_symbols

        # Filter to top N by average volume via Qlib local data
        try:
            vol_map = await asyncio.get_running_loop().run_in_executor(
                None, self._get_avg_volumes, market, all_symbols, settings,
            )
            if vol_map:
                sorted_syms = sorted(vol_map, key=vol_map.get, reverse=True)
                symbols = sorted_syms[:max_size]
                logger.info(
                    "Filtered universe to top %d by avg_volume for %s "
                    "(min_vol=%.0f, max_vol=%.0f)",
                    len(symbols), market,
                    vol_map.get(symbols[-1], 0),
                    vol_map.get(symbols[0], 0),
                )
                return symbols
        except Exception as e:
            logger.warning(
                "Volume filtering failed for %s: %s. "
                "Falling back to market_cap.", market, e,
            )

        # Fallback: market cap via StockPulse
        try:
            client = await get_stockpulse_async_client()
            cap_map = await client.get_market_caps(market, all_symbols[:3000])
            if cap_map:
                sorted_syms = sorted(cap_map, key=cap_map.get, reverse=True)
                symbols = sorted_syms[:max_size]
                logger.info(
                    "Filtered universe to top %d by market_cap for %s (fallback)",
                    len(symbols), market,
                )
                return symbols
        except Exception as e:
            logger.warning("Market cap fallback also failed: %s", e)

        logger.warning("No filtering available, using first %d symbols", max_size)
        return all_symbols[:max_size]

    @staticmethod
    def _get_avg_volumes(
        market: str, symbols: list[str], settings,
    ) -> dict[str, float]:
        """Compute 60-day average volume from Qlib local data (synchronous)."""
        from app.context import QlibContext
        from app.utils.symbol_mapping import normalize_symbol_for_qlib, qlib_to_stockpulse

        QlibContext.ensure_init(market, settings.QLIB_DATA_DIR)
        from qlib.data import D
        from datetime import date, timedelta

        end = date.today().isoformat()
        start = (date.today() - timedelta(days=90)).isoformat()

        qlib_syms = [normalize_symbol_for_qlib(s, market) for s in symbols]
        q_to_ws = dict(zip(qlib_syms, symbols))

        try:
            df = D.features(qlib_syms, ["$volume"], start_time=start, end_time=end)
            if df.empty:
                return {}
            avg_vol = df.groupby(level=0).mean()
            result = {}
            for q_sym in avg_vol.index:
                ws_sym = q_to_ws.get(q_sym, q_sym)
                vol = avg_vol.loc[q_sym, "$volume"]
                if vol > 0:
                    result[ws_sym] = float(vol)
            return result
        except Exception as e:
            logger.warning("Qlib volume query failed: %s", e)
            return {}

    # ------------------------------------------------------------------
    # Step 2: Check existing model (disk only)
    # ------------------------------------------------------------------

    def _check_existing_model_on_disk(
        self, market: str, model_date: date, forward_days: int,
    ) -> Optional[str]:
        """Check if a model already exists on disk for today."""
        settings = get_settings()
        date_str = model_date.strftime("%Y%m%d")
        model_path = os.path.join(
            settings.PREDICTION_DATA_DIR, market, date_str, "model.pkl"
        )
        if os.path.exists(model_path):
            return model_path
        return None

    def _find_latest_model_on_disk(
        self, market: str, forward_days: int,
    ) -> Optional[str]:
        """Find the latest model file on disk for a market."""
        settings = get_settings()
        market_dir = os.path.join(settings.PREDICTION_DATA_DIR, market)
        if not os.path.isdir(market_dir):
            return None

        try:
            date_dirs = sorted(os.listdir(market_dir), reverse=True)
        except OSError:
            return None

        for dirname in date_dirs:
            model_path = os.path.join(market_dir, dirname, "model.pkl")
            if os.path.exists(model_path):
                return model_path
        return None

    # ------------------------------------------------------------------
    # Step 3: Training
    # ------------------------------------------------------------------

    async def _train_model(
        self,
        task: PredictionTask,
        market: str,
        symbols: list[str],
        forward_days: int,
        model_date: date,
    ) -> tuple[Any, str, bool]:
        task.message = "Building training feature matrix"
        task.progress = 15.0

        train_end_date = model_date - timedelta(days=forward_days)
        train_start_date = model_date - timedelta(days=_TRAIN_LOOKBACK_DAYS)

        train_start_str = train_start_date.isoformat()
        train_end_str = model_date.isoformat()

        feature_df = await feature_service.build_feature_matrix(
            market=market,
            symbols=symbols,
            start_date=train_start_str,
            end_date=train_end_str,
        )

        if feature_df.empty:
            raise RuntimeError(
                f"Feature matrix is empty for market={market}."
            )

        task.message = f"Feature matrix: {len(feature_df)} rows"
        task.progress = 30.0

        close_df = await self._fetch_close_prices(market, symbols, train_start_str, train_end_str)
        if close_df.empty:
            raise RuntimeError("Close price data is empty.")

        task.progress = 35.0

        feature_df["date"] = pd.to_datetime(feature_df["date"])
        close_df["date"] = pd.to_datetime(close_df["date"])

        df = feature_df.merge(
            close_df[["symbol", "date", "close"]],
            on=["symbol", "date"],
            how="left",
        )

        # Forward returns + winsorization
        df = df.sort_values(["symbol", "date"])
        df["forward_return"] = df.groupby("symbol")["close"].transform(
            lambda x: x.shift(-forward_days) / x - 1
        )
        df["forward_return"] = df.groupby("date")["forward_return"].transform(
            lambda x: x.clip(x.quantile(0.01), x.quantile(0.99))
        )

        # Get training config
        try:
            from app.services.ml_agents import get_training_config
            cfg = await get_training_config(market, feature_df, symbols)
            logger.info("Using ML agent-generated config for market=%s", market)
        except Exception as e:
            logger.warning("ML agent config failed, using defaults: %s", e)
            cfg = get_market_config(market)

        # Sector-neutral labels
        if cfg.use_sector_neutral_labels:
            try:
                client = await get_stockpulse_async_client()
                sector_map = await client.get_sector_map(market, symbols=symbols)
            except Exception:
                sector_map = {}
            if sector_map:
                df["_sector"] = df["symbol"].map(sector_map)
                sector_coverage = df["_sector"].notna().mean()
                if sector_coverage >= 0.3:
                    sector_mean = df.groupby(["date", "_sector"])["forward_return"].transform("mean")
                    has_sector = df["_sector"].notna()
                    df.loc[has_sector, "forward_return"] = (
                        df.loc[has_sector, "forward_return"] - sector_mean[has_sector]
                    )
                    logger.info("Applied sector-neutral excess returns")
                df = df.drop(columns=["_sector"])

        # Drop rows without forward return
        df = df.dropna(subset=["forward_return"])
        if len(df) < _MIN_TRAIN_DATES * _MIN_SYMBOLS_PER_DATE:
            raise RuntimeError(f"Insufficient labeled data: {len(df)} rows.")

        # Per-date percentile labels
        use_legacy = not cfg.use_balanced_quintiles

        def _label_fn(x: "pd.Series") -> "pd.Series":
            if len(x) < _MIN_SYMBOLS_PER_DATE:
                return pd.Series([2] * len(x), index=x.index)
            if use_legacy:
                return pd.qcut(x, q=5, labels=False, duplicates="drop")
            ranked = x.rank(method="first")
            return pd.qcut(ranked, q=5, labels=False).astype(float)

        df["label"] = df.groupby("date")["forward_return"].transform(_label_fn)
        df["label"] = df["label"].fillna(2).astype(float)

        task.message = "Splitting train/validation sets"
        task.progress = 40.0

        unique_dates = sorted(df["date"].unique())
        sort_cols = ["symbol", "date"] if cfg.use_temporal_sort else ["date", "symbol"]

        meta_cols = {"symbol", "date", "close", "forward_return", "label"}
        feature_cols = [c for c in df.columns if c not in meta_cols]
        if not feature_cols:
            raise RuntimeError("No feature columns found")

        settings = get_settings()
        ensemble_size = settings.ENSEMBLE_SIZE
        n_folds = settings.WALKFORWARD_FOLDS

        splits = self._walk_forward_splits(
            unique_dates, n_folds=n_folds, forward_days=forward_days,
        )
        if not splits:
            raise RuntimeError("Could not generate walk-forward splits")

        fold_ics: list[float] = []
        fold_icirs: list[float] = []
        final_models: list[lgb.Booster] = []
        final_val_df: pd.DataFrame = pd.DataFrame()
        final_val_scores: np.ndarray = np.array([])
        final_train_dates: list = []
        final_val_dates: list = []
        final_train_df: pd.DataFrame = pd.DataFrame()

        for fold_idx, (tr_dates, va_dates) in enumerate(splits):
            is_final_fold = fold_idx == len(splits) - 1

            tr_mask = df["date"].isin(tr_dates)
            va_mask = df["date"].isin(va_dates)
            tr_df = df[tr_mask].copy().sort_values(sort_cols).reset_index(drop=True)
            va_df = df[va_mask].copy().sort_values(sort_cols).reset_index(drop=True)

            X_tr = tr_df[feature_cols].values
            y_tr = tr_df["label"].values
            X_va = va_df[feature_cols].values
            y_va = va_df["label"].values
            tr_group = tr_df.groupby("date", sort=True).size().values
            va_group = va_df.groupby("date", sort=True).size().values

            tr_set = lgb.Dataset(X_tr, label=y_tr, group=tr_group, feature_name=feature_cols)
            va_set = lgb.Dataset(X_va, label=y_va, group=va_group, feature_name=feature_cols, reference=tr_set)

            task.message = f"Walk-forward fold {fold_idx + 1}/{len(splits)}"

            models = await asyncio.to_thread(
                self._train_ensemble_sync, tr_set, va_set, market, ensemble_size, cfg,
            )

            va_scores_list = [m.predict(X_va) for m in models]
            va_scores = np.mean(va_scores_list, axis=0)
            va_actual = va_df["forward_return"].values

            _, fold_ic, fold_icir = self._compute_ic_metrics(va_df, va_scores, va_actual)
            fold_ics.append(fold_ic)
            fold_icirs.append(fold_icir)

            logger.info("  Fold %d IC=%.4f, ICIR=%.4f", fold_idx + 1, fold_ic, fold_icir)

            if is_final_fold:
                final_models = models
                final_val_df = va_df
                final_val_scores = va_scores
                final_train_dates = list(tr_dates)
                final_val_dates = list(va_dates)
                final_train_df = tr_df

        models = final_models
        val_df = final_val_df
        val_actual = val_df["forward_return"].values

        ic_mean = float(np.mean(fold_ics))
        icir = float(np.mean(fold_icirs)) if fold_icirs else 0.0

        task.message = "Ensemble trained, evaluating performance"
        task.progress = 55.0

        # Best NDCG
        ndcg_values = []
        for m in models:
            if m.best_score and "valid_0" in m.best_score:
                v = m.best_score["valid_0"]
                ndcg = v.get("ndcg@10", v.get("ndcg@5"))
                if ndcg is not None:
                    ndcg_values.append(ndcg)
        best_ndcg = float(np.mean(ndcg_values)) if ndcg_values else None

        # Quality gate
        min_ic = cfg.min_ic_threshold
        min_icir = cfg.min_icir_threshold
        quality_passed = (ic_mean > min_ic and icir > min_icir)

        if not quality_passed:
            logger.warning(
                "Model quality gate FAILED: mean_IC=%.4f (min=%.4f), mean_ICIR=%.4f (min=%.4f)",
                ic_mean, min_ic, icir, min_icir,
            )
        else:
            logger.info("Model quality gate passed: mean_IC=%.4f, mean_ICIR=%.4f", ic_mean, icir)

        # Feature importance
        feature_importance: dict[str, float] = {}
        try:
            importance_arrays = [m.feature_importance(importance_type='gain') for m in models]
            importance_values = np.mean(importance_arrays, axis=0)
            feature_importance = dict(
                sorted(
                    zip(feature_cols, (float(v) for v in importance_values)),
                    key=lambda x: x[1],
                    reverse=True,
                )
            )
        except Exception as e:
            logger.warning("Failed to extract feature importance: %s", e)

        # Save model to disk
        task.message = "Saving model"
        task.progress = 60.0
        model_path = self._save_model(models, market, model_date, feature_cols, feature_importance)

        # Save training distribution snapshot for PSI
        try:
            self._save_train_distribution(
                final_train_df, feature_cols, os.path.dirname(model_path),
            )
        except Exception as e:
            logger.warning("Failed to save training distribution: %s", e)

        # Record model metadata via StockPulse API
        task.message = "Recording model metadata"
        task.progress = 65.0

        has_fundamental = any(c in feature_cols for c in FUNDAMENTAL_FEATURES)
        has_sentiment = any(c in feature_cols for c in SENTIMENT_FEATURES)
        feature_sources = ["alpha158"]
        if has_fundamental:
            feature_sources.append("fundamental")
        if has_sentiment:
            feature_sources.append("sentiment")

        model_id = await self._record_model(
            market=market,
            model_date=model_date,
            train_start=pd.Timestamp(final_train_dates[0]).date(),
            train_end=pd.Timestamp(final_train_dates[-1]).date(),
            val_start=pd.Timestamp(final_val_dates[0]).date(),
            val_end=pd.Timestamp(final_val_dates[-1]).date(),
            forward_days=forward_days,
            feature_count=len(feature_cols),
            symbol_count=final_train_df["symbol"].nunique(),
            feature_sources=feature_sources,
            ic=ic_mean,
            icir=icir,
            ndcg=best_ndcg,
            model_path=model_path,
            feature_importance=feature_importance,
            quality_passed=quality_passed,
            extra_metadata={
                "ensemble_size": ensemble_size,
                "walkforward_folds": len(splits),
                "fold_ics": [round(ic, 6) for ic in fold_ics],
                "fold_icirs": [round(ir, 6) for ir in fold_icirs],
            },
        )

        task.progress = 70.0
        return model_id, model_path, quality_passed

    async def train_for_backtest(
        self,
        market: str,
        symbols: list[str],
        forward_days: int,
        cutoff_date: date,
        config: MarketConfig,
        feature_df: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Train models for backtest -- returns results in memory, no DB/disk writes."""
        settings = get_settings()

        if feature_df is None:
            train_start_date = cutoff_date - timedelta(days=_TRAIN_LOOKBACK_DAYS)
            feature_df = await feature_service.build_feature_matrix(
                market=market,
                symbols=symbols,
                start_date=train_start_date.isoformat(),
                end_date=cutoff_date.isoformat(),
                config_override=config,
            )

        if feature_df.empty:
            raise RuntimeError("Feature matrix is empty for backtest training")

        train_start_str = (cutoff_date - timedelta(days=_TRAIN_LOOKBACK_DAYS)).isoformat()
        close_df = await self._fetch_close_prices(
            market, symbols, train_start_str, cutoff_date.isoformat()
        )
        if close_df.empty:
            raise RuntimeError("Close price data is empty for backtest")

        feature_df["date"] = pd.to_datetime(feature_df["date"])
        close_df["date"] = pd.to_datetime(close_df["date"])
        df = feature_df.merge(
            close_df[["symbol", "date", "close"]],
            on=["symbol", "date"],
            how="left",
        )

        df = df.sort_values(["symbol", "date"])
        df["forward_return"] = df.groupby("symbol")["close"].transform(
            lambda x: x.shift(-forward_days) / x - 1
        )
        df["forward_return"] = df.groupby("date")["forward_return"].transform(
            lambda x: x.clip(x.quantile(0.01), x.quantile(0.99))
        )

        if config.use_sector_neutral_labels:
            try:
                client = await get_stockpulse_async_client()
                sector_map = await client.get_sector_map(market, symbols=symbols)
            except Exception:
                sector_map = {}
            if sector_map:
                df["_sector"] = df["symbol"].map(sector_map)
                sector_coverage = df["_sector"].notna().mean()
                if sector_coverage >= 0.3:
                    sector_mean = df.groupby(["date", "_sector"])["forward_return"].transform("mean")
                    has_sector = df["_sector"].notna()
                    df.loc[has_sector, "forward_return"] = (
                        df.loc[has_sector, "forward_return"] - sector_mean[has_sector]
                    )
                df = df.drop(columns=["_sector"])

        df = df.dropna(subset=["forward_return"])
        if len(df) < _MIN_TRAIN_DATES * _MIN_SYMBOLS_PER_DATE:
            raise RuntimeError(f"Insufficient labeled data for backtest: {len(df)} rows")

        use_legacy = not config.use_balanced_quintiles

        def _label_fn(x: "pd.Series") -> "pd.Series":
            if len(x) < _MIN_SYMBOLS_PER_DATE:
                return pd.Series([2] * len(x), index=x.index)
            if use_legacy:
                return pd.qcut(x, q=5, labels=False, duplicates="drop")
            ranked = x.rank(method="first")
            return pd.qcut(ranked, q=5, labels=False).astype(float)

        df["label"] = df.groupby("date")["forward_return"].transform(_label_fn)
        df["label"] = df["label"].fillna(2).astype(float)

        cutoff_ts = pd.Timestamp(cutoff_date)
        df = df[df["date"] <= cutoff_ts]

        unique_dates = sorted(df["date"].unique())
        sort_cols = ["symbol", "date"] if config.use_temporal_sort else ["date", "symbol"]

        meta_cols = {"symbol", "date", "close", "forward_return", "label"}
        feature_cols = [c for c in df.columns if c not in meta_cols]
        if not feature_cols:
            raise RuntimeError("No feature columns found for backtest")

        ensemble_size = settings.ENSEMBLE_SIZE
        n_folds = settings.WALKFORWARD_FOLDS

        splits = self._walk_forward_splits(
            unique_dates, n_folds=n_folds, forward_days=forward_days,
        )
        if not splits:
            raise RuntimeError("Could not generate walk-forward splits for backtest")

        fold_ics: list[float] = []
        fold_icirs: list[float] = []
        final_models: list[lgb.Booster] = []

        for fold_idx, (tr_dates, va_dates) in enumerate(splits):
            is_final_fold = fold_idx == len(splits) - 1

            tr_mask = df["date"].isin(tr_dates)
            va_mask = df["date"].isin(va_dates)
            tr_df = df[tr_mask].copy().sort_values(sort_cols).reset_index(drop=True)
            va_df = df[va_mask].copy().sort_values(sort_cols).reset_index(drop=True)

            X_tr = tr_df[feature_cols].values
            y_tr = tr_df["label"].values
            X_va = va_df[feature_cols].values
            y_va = va_df["label"].values
            tr_group = tr_df.groupby("date", sort=True).size().values
            va_group = va_df.groupby("date", sort=True).size().values

            tr_set = lgb.Dataset(X_tr, label=y_tr, group=tr_group, feature_name=feature_cols)
            va_set = lgb.Dataset(X_va, label=y_va, group=va_group, feature_name=feature_cols, reference=tr_set)

            models = await asyncio.to_thread(
                self._train_ensemble_sync, tr_set, va_set, market, ensemble_size, config,
            )

            va_scores = np.mean([m.predict(X_va) for m in models], axis=0)
            _, fold_ic, fold_icir = self._compute_ic_metrics(
                va_df, va_scores, va_df["forward_return"].values,
            )
            fold_ics.append(fold_ic)
            fold_icirs.append(fold_icir)

            if is_final_fold:
                final_models = models

        ic_mean = float(np.mean(fold_ics))
        icir = float(np.mean(fold_icirs)) if fold_icirs else 0.0

        best_iters = [
            m.best_iteration if m.best_iteration >= 0 else config.num_boost_round
            for m in final_models
        ]

        ndcg_values = []
        for m in final_models:
            if m.best_score and "valid_0" in m.best_score:
                v = m.best_score["valid_0"]
                ndcg = v.get("ndcg@10", v.get("ndcg@5"))
                if ndcg is not None:
                    ndcg_values.append(ndcg)
        best_ndcg = float(np.mean(ndcg_values)) if ndcg_values else None

        feature_importance: dict[str, float] = {}
        try:
            importance_arrays = [m.feature_importance(importance_type='gain') for m in final_models]
            importance_values = np.mean(importance_arrays, axis=0)
            feature_importance = dict(
                sorted(
                    zip(feature_cols, (float(v) for v in importance_values)),
                    key=lambda x: x[1],
                    reverse=True,
                )
            )
        except Exception:
            pass

        quality_passed = (ic_mean > config.min_ic_threshold and icir > config.min_icir_threshold)

        return {
            "models": final_models,
            "feature_cols": feature_cols,
            "ic": ic_mean,
            "icir": icir,
            "ndcg": best_ndcg,
            "fold_ics": fold_ics,
            "fold_icirs": fold_icirs,
            "best_iters": best_iters,
            "feature_importance": feature_importance,
            "quality_passed": quality_passed,
            "ensemble_size": ensemble_size,
            "symbol_count": df["symbol"].nunique(),
            "feature_count": len(feature_cols),
        }

    @staticmethod
    def _walk_forward_splits(
        unique_dates: list, n_folds: int, forward_days: int,
    ) -> list[tuple[list, list]]:
        total = len(unique_dates)
        if n_folds <= 1:
            split_idx = int(total * 0.8)
            if split_idx < _MIN_TRAIN_DATES:
                return []
            val_start = min(split_idx + forward_days, total - 1)
            train_dates = unique_dates[:split_idx]
            val_dates = unique_dates[val_start:]
            if len(val_dates) < 5:
                return []
            return [(train_dates, val_dates)]

        val_size = max(total // (n_folds + 2), 10)
        splits: list[tuple[list, list]] = []

        for i in range(n_folds):
            val_end_idx = total - (n_folds - 1 - i) * val_size
            val_start_idx = val_end_idx - val_size
            train_end_idx = val_start_idx - forward_days

            if train_end_idx < _MIN_TRAIN_DATES:
                continue
            if val_start_idx < 0 or val_end_idx > total:
                continue

            train_dates = unique_dates[:train_end_idx]
            val_dates = unique_dates[val_start_idx:val_end_idx]

            if len(val_dates) < 5:
                continue
            splits.append((train_dates, val_dates))

        return splits

    @staticmethod
    def _train_ensemble_sync(
        train_set: lgb.Dataset,
        val_set: lgb.Dataset,
        market: str = "us",
        ensemble_size: int = 5,
        cfg: MarketConfig | None = None,
    ) -> list[lgb.Booster]:
        seeds = _ENSEMBLE_SEEDS[:ensemble_size]
        models: list[lgb.Booster] = []

        for i, seed in enumerate(seeds):
            logger.info("Training ensemble member %d/%d (seed=%d) for %s", i + 1, ensemble_size, seed, market)
            params = _get_lgb_params(market, cfg)
            params["seed"] = seed
            params["feature_fraction_seed"] = seed
            params["bagging_seed"] = seed

            num_boost_round = _get_boost_round(market, cfg)
            early_stopping = _get_early_stopping(market, cfg)
            callbacks = [lgb.early_stopping(early_stopping), lgb.log_evaluation(period=50)]

            model = lgb.train(
                params, train_set,
                valid_sets=[val_set], valid_names=["valid_0"],
                num_boost_round=num_boost_round, callbacks=callbacks,
            )
            models.append(model)

        return models

    @staticmethod
    def _compute_ic_metrics(
        val_df: pd.DataFrame,
        predicted_scores: np.ndarray,
        actual_returns: np.ndarray,
    ) -> tuple[pd.Series, float, float]:
        temp = val_df[["date"]].copy()
        temp["pred"] = predicted_scores
        temp["actual"] = actual_returns

        ic_per_date = temp.groupby("date").apply(
            lambda g: g["pred"].corr(g["actual"], method="spearman")
            if len(g) >= 5
            else np.nan,
            include_groups=False,
        )
        ic_per_date = ic_per_date.dropna()

        if len(ic_per_date) == 0:
            return pd.Series(dtype=float), 0.0, 0.0

        ic_mean = float(ic_per_date.mean())
        ic_std = float(ic_per_date.std())
        icir = ic_mean / ic_std if ic_std > 1e-10 else 0.0

        return ic_per_date, ic_mean, icir

    @staticmethod
    def _save_model(
        models: list[lgb.Booster] | lgb.Booster,
        market: str,
        model_date: date,
        feature_cols: list[str],
        feature_importance: dict[str, float] | None = None,
    ) -> str:
        if isinstance(models, lgb.Booster):
            models = [models]

        settings = get_settings()
        date_str = model_date.strftime("%Y%m%d")
        model_dir = Path(settings.PREDICTION_DATA_DIR) / market / date_str
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = str(model_dir / "model.pkl")
        joblib.dump(models, model_path)

        features_meta: dict[str, Any] = {
            "features": feature_cols,
            "count": len(feature_cols),
            "ensemble_size": len(models),
        }
        if feature_importance is not None:
            features_meta["feature_importance"] = feature_importance

        features_path = str(model_dir / "features.json")
        with open(features_path, "w") as f:
            json.dump(features_meta, f, default=_numpy_default)

        return model_path

    @staticmethod
    def _save_train_distribution(
        train_df: pd.DataFrame, feature_cols: list[str], model_dir: str,
    ) -> None:
        dist: dict[str, list[float]] = {}
        percentiles = np.linspace(0, 100, 11).tolist()
        for col in feature_cols:
            if col not in train_df.columns:
                continue
            vals = train_df[col].dropna().values
            if len(vals) < 20:
                continue
            dist[col] = [float(v) for v in np.percentile(vals, percentiles)]

        path = os.path.join(model_dir, "train_distribution.json")
        with open(path, "w") as f:
            json.dump(dist, f, default=_numpy_default)
        logger.info("Saved training distribution snapshot: %d features", len(dist))

    @staticmethod
    def _compute_inference_psi(
        inference_df: pd.DataFrame, feature_cols: list[str], model_dir: str,
    ) -> dict[str, float] | None:
        dist_path = os.path.join(model_dir, "train_distribution.json")
        if not os.path.exists(dist_path):
            return None
        with open(dist_path) as f:
            train_dist = json.load(f)

        _EPS = 1e-6
        psi_scores: dict[str, float] = {}
        for col in feature_cols:
            if col not in train_dist or col not in inference_df.columns:
                continue
            bin_edges = train_dist[col]
            if len(bin_edges) < 3:
                continue
            infer_vals = inference_df[col].dropna().values
            if len(infer_vals) < 10:
                continue
            edges = np.array(bin_edges)
            edges[0] = -np.inf
            edges[-1] = np.inf
            n_bins = len(edges) - 1
            expected = np.full(n_bins, 1.0 / n_bins) + _EPS
            counts = np.histogram(infer_vals, bins=edges)[0]
            actual = counts / len(infer_vals) + _EPS
            psi = float(np.sum((actual - expected) * np.log(actual / expected)))
            psi_scores[col] = round(psi, 6)

        return psi_scores if psi_scores else None

    async def _record_model(
        self,
        market: str,
        model_date: date,
        train_start: date,
        train_end: date,
        val_start: date,
        val_end: date,
        forward_days: int,
        feature_count: int,
        symbol_count: int,
        feature_sources: list[str],
        ic: float,
        icir: float,
        ndcg: Optional[float],
        model_path: str,
        feature_importance: dict[str, float] | None = None,
        quality_passed: bool = True,
        extra_metadata: dict[str, Any] | None = None,
    ) -> Any:
        """Write model metadata via local PredictionStore."""
        metadata: dict[str, Any] = {
            "lgb_params": _get_lgb_params(market),
            "num_boost_round": _get_boost_round(market),
            "early_stopping": _get_early_stopping(market),
        }
        if extra_metadata:
            metadata.update(extra_metadata)
        if feature_importance:
            top_items = list(feature_importance.items())[:30]
            metadata["feature_importance_top30"] = dict(top_items)

        # Generate a deterministic model ID from key attributes
        id_seed = f"{market}:{model_date.isoformat()}:{forward_days}:ranking"
        model_hash = hashlib.sha256(id_seed.encode()).hexdigest()[:32]

        model_data = {
            "id": model_hash,
            "market": market,
            "model_date": model_date.isoformat(),
            "forward_days": forward_days,
            "feature_count": feature_count,
            "symbol_count": symbol_count,
            "ic": float(ic),
            "icir": float(icir),
            "ndcg": float(ndcg) if ndcg is not None else None,
            "fold_ics": extra_metadata.get("fold_ics") if extra_metadata else None,
            "best_iterations": extra_metadata.get("best_iterations") if extra_metadata else None,
            "ensemble_size": extra_metadata.get("ensemble_size", 1) if extra_metadata else 1,
            "feature_importance": feature_importance,
            "file_path": model_path,
            "training_config": json.dumps(metadata, default=_numpy_default),
            "quality": "approved" if quality_passed else "rejected",
            "model_type": "ranking",
        }

        try:
            model_id = await prediction_store.write_model(model_data)
            return model_id
        except Exception as e:
            logger.error("Failed to record model: %s", e)
            raise RuntimeError(f"Model recording failed: {e}") from e

    # ------------------------------------------------------------------
    # Step 4: Inference
    # ------------------------------------------------------------------

    async def _run_inference(
        self,
        task: PredictionTask,
        market: str,
        symbols: list[str],
        model_id: Any,
        model_path: Optional[str],
        forward_days: int,
        prediction_date: date,
    ) -> int:
        if model_path is None or not os.path.exists(model_path):
            model_path = self._find_latest_model_on_disk(market, forward_days)

        if model_path is None or not os.path.exists(model_path):
            raise RuntimeError(f"No model file found for market={market}")

        task.message = "Loading model"
        task.progress = 78.0

        loaded = await asyncio.to_thread(joblib.load, model_path)
        models = loaded if isinstance(loaded, list) else [loaded]

        # Load feature names
        features_path = os.path.join(os.path.dirname(model_path), "features.json")

        def _load_feature_meta() -> Optional[dict]:
            if not os.path.exists(features_path):
                return None
            with open(features_path) as f:
                return json.load(f)

        feature_meta = await asyncio.to_thread(_load_feature_meta)
        feature_cols = feature_meta["features"] if feature_meta else feature_service.get_feature_names()

        # Build inference features
        task.message = "Building inference features"
        task.progress = 82.0

        inference_end = prediction_date.isoformat()
        inference_start = (prediction_date - timedelta(days=90)).isoformat()

        inference_df = await feature_service.build_feature_matrix(
            market=market, symbols=symbols,
            start_date=inference_start, end_date=inference_end,
        )

        if inference_df.empty:
            raise RuntimeError("Inference feature matrix is empty.")

        inference_df["date"] = pd.to_datetime(inference_df["date"])

        # Pick latest date with adequate coverage
        date_symbol_counts = inference_df.groupby("date")["symbol"].nunique().sort_index()
        max_date = date_symbol_counts.index.max()
        max_date_count = date_symbol_counts.loc[max_date]
        total_symbols = inference_df["symbol"].nunique()

        settings = get_settings()
        min_coverage = settings.INFERENCE_MIN_COVERAGE

        if max_date_count >= total_symbols * min_coverage:
            latest_date = max_date
        else:
            threshold = total_symbols * min_coverage
            candidates = date_symbol_counts[date_symbol_counts >= threshold]
            if candidates.empty:
                raise RuntimeError(
                    f"Insufficient symbol coverage for inference: "
                    f"best date has {max_date_count}/{total_symbols} symbols."
                )
            latest_date = candidates.index.max()

        latest_df = inference_df[inference_df["date"] == latest_date].copy()

        # Align feature columns
        missing_features = [c for c in feature_cols if c not in latest_df.columns]
        if missing_features:
            missing_pct = len(missing_features) / len(feature_cols)
            if missing_pct > 0.25:
                logger.error("Inference: %.0f%% features missing", missing_pct * 100)
                await self._flag_retrain_needed(market)
            for col in missing_features:
                latest_df[col] = np.nan

        X_inference = latest_df[feature_cols].values

        # Feature drift detection
        model_dir = os.path.dirname(model_path)
        psi_scores = self._compute_inference_psi(latest_df, feature_cols, model_dir)
        if psi_scores:
            high_drift = {k: v for k, v in psi_scores.items() if v > 0.2}
            moderate_drift = {k: v for k, v in psi_scores.items() if 0.1 < v <= 0.2}
            if high_drift:
                logger.warning("HIGH feature drift (PSI>0.2) in %d features", len(high_drift))
            task._psi_data = {
                "high_drift_count": len(high_drift),
                "moderate_drift_count": len(moderate_drift),
                "top_drifted": dict(sorted(psi_scores.items(), key=lambda x: -x[1])[:10]),
            }

        # Predict (ensemble average)
        task.message = "Generating predictions"
        task.progress = 88.0

        def _ensemble_predict() -> np.ndarray:
            return np.mean([m.predict(X_inference) for m in models], axis=0)

        scores = await asyncio.to_thread(_ensemble_predict)
        ranks = pd.Series(scores).rank(pct=True).values

        directions = []
        for rank_val in ranks:
            if rank_val >= DIRECTION_UP_THRESHOLD:
                directions.append("up")
            elif rank_val <= DIRECTION_DOWN_THRESHOLD:
                directions.append("down")
            else:
                directions.append("sideways")

        results_df = pd.DataFrame({
            "symbol": latest_df["symbol"].values,
            "predicted_score": scores,
            "percentile_rank": ranks,
            "predicted_direction": directions,
        })
        results_df = results_df.sort_values("predicted_score", ascending=False)

        # Write predictions via StockPulse API
        task.message = "Writing predictions"
        task.progress = 92.0
        await self._write_predictions(market, prediction_date, model_id, results_df, forward_days)

        # Cache in Redis
        task.message = "Caching predictions"
        task.progress = 96.0
        await self._write_prediction_cache(market, results_df, prediction_date)

        return len(results_df)

    async def _write_predictions(
        self,
        market: str,
        prediction_date: date,
        model_id: Any,
        results_df: pd.DataFrame,
        forward_days: int,
    ) -> None:
        """Write prediction results via local PredictionStore."""
        predictions = [
            {
                "market": market,
                "prediction_date": prediction_date.isoformat(),
                "model_id": str(model_id) if model_id else None,
                "symbol": row["symbol"],
                "rank_score": float(row["predicted_score"]),
                "percentile_rank": float(row["percentile_rank"]),
                "forward_days": forward_days,
            }
            for _, row in results_df.iterrows()
        ]

        try:
            written = await prediction_store.write_predictions(predictions)
            logger.info(
                "Wrote %d predictions: market=%s, date=%s",
                written, market, prediction_date.isoformat(),
            )
        except Exception as e:
            logger.error("Failed to write predictions: %s", e)
            raise RuntimeError(f"Prediction write failed: {e}") from e

    # ------------------------------------------------------------------
    # Close price fetch (via StockPulse)
    # ------------------------------------------------------------------

    async def _fetch_close_prices(
        self,
        market: str,
        symbols: list[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Fetch close prices via StockPulse async client."""
        try:
            client = await get_stockpulse_async_client()
            batch_size = 30
            all_data: dict = {}
            for i in range(0, len(symbols), batch_size):
                batch = symbols[i:i + batch_size]
                try:
                    batch_data = await client.get_bars_batch_async(
                        batch, market, start_date, end_date,
                    )
                    all_data.update(batch_data)
                except Exception as e:
                    logger.warning("Close price batch %d-%d failed: %s", i, i + len(batch), e)
        except Exception as e:
            logger.error("Close price fetch failed: %s", e)
            return pd.DataFrame()

        if not all_data:
            return pd.DataFrame()

        rows = []
        for symbol, bars in all_data.items():
            dates = bars.get("dates", [])
            closes = bars.get("close", [])
            for d, c in zip(dates, closes):
                if c is not None:
                    rows.append({"symbol": symbol, "date": d, "close": float(c)})

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        return df

    # ------------------------------------------------------------------
    # Actual return computation (for backfill)
    # ------------------------------------------------------------------

    async def _compute_actual_returns(
        self,
        symbol: str,
        predictions: list[dict],
        market: str = "us",
    ) -> dict:
        min_date = min(p["prediction_date"] for p in predictions)
        max_date = max(p["prediction_date"] for p in predictions)
        max_forward = max(p["forward_days"] for p in predictions)

        if isinstance(min_date, str):
            min_date = date.fromisoformat(min_date)
        if isinstance(max_date, str):
            max_date = date.fromisoformat(max_date)

        start_str = min_date.isoformat()
        calendar_buffer = int(max_forward * 7 / 5) + 15
        end_str = (max_date + timedelta(days=calendar_buffer)).isoformat()

        try:
            client = await get_stockpulse_async_client()
            data = await client.get_bars_batch_async(
                [symbol], market, start_str, end_str,
            )
        except Exception as e:
            logger.warning("Price fetch failed for %s: %s", symbol, e)
            return {}

        if symbol not in data:
            return {}

        dates = data[symbol].get("dates", [])
        closes = data[symbol].get("close", [])

        if not dates or not closes:
            return {}

        price_map: dict[date, float] = {}
        for d_str, c in zip(dates, closes):
            if c is not None:
                d = pd.Timestamp(d_str).date()
                price_map[d] = float(c)
        trading_dates = sorted(price_map.keys())
        date_to_idx: dict[date, int] = {d: i for i, d in enumerate(trading_dates)}

        results: dict = {}
        for pred in predictions:
            pred_date = pred["prediction_date"]
            if isinstance(pred_date, str):
                pred_date = date.fromisoformat(pred_date)
            fwd = pred["forward_days"]

            base_price = price_map.get(pred_date)
            if base_price is None:
                continue
            base_idx = date_to_idx.get(pred_date)
            if base_idx is None:
                continue
            target_idx = base_idx + fwd
            if target_idx >= len(trading_dates):
                continue
            future_price = price_map[trading_dates[target_idx]]
            if future_price is not None and base_price > 0:
                actual_return = (future_price / base_price) - 1.0
                results[pred["id"]] = actual_return

        return results

    # ------------------------------------------------------------------
    # Redis prediction cache
    # ------------------------------------------------------------------

    async def _read_prediction_cache(self, market: str) -> Optional[list[dict]]:
        key = _prediction_cache_key(market)
        try:
            r = await _get_redis_client()
            data = await r.get(key)
            if data is None:
                return None
            return msgpack.unpackb(data, raw=False)
        except Exception as e:
            logger.warning("Prediction cache read failed: %s", e)
            return None

    async def _write_prediction_cache(
        self, market: str, results_df: pd.DataFrame, prediction_date: date,
    ) -> None:
        if results_df.empty:
            return
        key = _prediction_cache_key(market)
        try:
            records = []
            for _, row in results_df.iterrows():
                records.append({
                    "symbol": row["symbol"],
                    "predicted_score": float(row["predicted_score"]),
                    "percentile_rank": float(row["percentile_rank"]),
                    "predicted_direction": row["predicted_direction"],
                    "prediction_date": prediction_date.isoformat(),
                })

            packed = msgpack.packb(records, use_bin_type=True)
            r = await _get_redis_client()
            await r.setex(key, _PREDICTION_CACHE_TTL, packed)
            logger.info("Cached %d predictions: key=%s", len(records), key)
        except Exception as e:
            logger.warning("Prediction cache write failed: %s", e)

    async def _refresh_prediction_cache(self, market: str) -> None:
        try:
            settings = get_settings()
            default_horizon = int(settings.PREDICTION_HORIZONS.split(",")[0].strip())
            predictions = await prediction_store.get_latest_predictions(
                market, 500, forward_days=default_horizon,
            )
            if not predictions:
                return
            key = _prediction_cache_key(market)
            packed = msgpack.packb(predictions, use_bin_type=True)
            r = await _get_redis_client()
            await r.setex(key, _PREDICTION_CACHE_TTL, packed)
            logger.info("Refreshed prediction cache: market=%s, %d records", market, len(predictions))
        except Exception as e:
            logger.warning("Prediction cache refresh failed: %s", e)

    # ------------------------------------------------------------------
    # Data freshness check
    # ------------------------------------------------------------------

    def _check_data_freshness(self, market: str) -> tuple[bool, str]:
        settings = get_settings()
        max_stale = settings.PREDICTION_MAX_STALE_DAYS
        if max_stale <= 0:
            return True, "Freshness check disabled"

        market_map = {"cn": "cn_data", "us": "us_data", "hk": "hk_data"}
        qlib_market = market_map.get(market)
        if not qlib_market:
            return True, f"Unknown market {market}, skipping freshness check"

        calendar_path = os.path.join(
            settings.QLIB_DATA_DIR, qlib_market, "calendars", "day.txt"
        )

        if not os.path.exists(calendar_path):
            return False, f"Calendar file not found: {calendar_path}"

        try:
            with open(calendar_path) as f:
                lines = f.read().strip().splitlines()
            if not lines:
                return False, "Calendar file is empty"

            last_date_str = lines[-1].strip()
            last_date = date.fromisoformat(last_date_str)
        except Exception as e:
            return False, f"Failed to parse calendar: {e}"

        today = date.today()
        gap_days = (today - last_date).days
        approx_trading_gap = int(gap_days * 5 / 7)

        if approx_trading_gap > max_stale:
            return False, (
                f"Qlib data stale: last_date={last_date_str}, "
                f"gap={gap_days}d (~{approx_trading_gap} trading days)"
            )

        return True, f"Data fresh: last_date={last_date_str}, gap={gap_days}d"

    # ------------------------------------------------------------------
    # Model file cleanup
    # ------------------------------------------------------------------

    async def cleanup_old_models(self) -> dict:
        settings = get_settings()
        retention_days = settings.MODEL_RETENTION_DAYS
        base_dir = settings.PREDICTION_DATA_DIR
        cutoff = date.today() - timedelta(days=retention_days)

        deleted = 0
        kept = 0
        errors = 0

        for market_dir in ("cn", "us", "hk"):
            market_path = os.path.join(base_dir, market_dir)
            if not os.path.isdir(market_path):
                continue

            try:
                date_dirs = sorted(os.listdir(market_path))
            except OSError:
                continue

            for dirname in date_dirs:
                dirpath = os.path.join(market_path, dirname)
                if not os.path.isdir(dirpath):
                    continue
                try:
                    dir_date = date(int(dirname[:4]), int(dirname[4:6]), int(dirname[6:8]))
                except (ValueError, IndexError):
                    continue

                if dir_date < cutoff:
                    try:
                        import shutil
                        shutil.rmtree(dirpath)
                        deleted += 1
                    except OSError as e:
                        logger.warning("Failed to delete %s: %s", dirpath, e)
                        errors += 1
                else:
                    kept += 1

        logger.info(
            "Model cleanup: deleted=%d, kept=%d, errors=%d",
            deleted, kept, errors,
        )
        return {"deleted": deleted, "kept": kept, "errors": errors}


# ---------------------------------------------------------------------------
# Module singleton + shutdown
# ---------------------------------------------------------------------------

prediction_service = PredictionService()


def shutdown_prediction_service() -> None:
    prediction_service.shutdown()
