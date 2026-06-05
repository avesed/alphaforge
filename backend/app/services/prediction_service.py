"""LightGBM prediction service -- training + inference engine.

Training pipeline:
1. QlibContext.ensure_init(market)
2. Resolve universe symbols via StockPulse API
3. feature_service.build_feature_matrix() -> ~87 features x N stocks x T days
4. Label engineering: forward N-day return -> percentile score (continuous target)
5. Purged time-series split (train/val) with gap = forward_days
6. LightGBM training: objective='lambdarank', early_stopping_rounds=50
7. joblib.dump() model -> /app/data/predictions/{market}/{YYYYMMDD}/model.pkl
   (5d/legacy) or model.{forward_days}d.pkl for other horizons
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
import bisect
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
from app.services.prediction_store import deterministic_model_id, prediction_store
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
# Quality markers (on-disk hot-path cache for model quality; DB authoritative)
# ---------------------------------------------------------------------------


def _ranking_model_filename(forward_days: int) -> str:
    """Per-horizon ranking model artifact filename within a date directory.

    Multiple horizons (e.g. 5d + 20d) share the same ``{market}/{date}/``
    directory, so the ranking artifact must be keyed on ``forward_days`` to
    avoid one horizon's ``_save_model`` overwriting another's model.

    The 5d horizon keeps the legacy ``model.pkl`` name so existing
    single-horizon directories on disk (all written before multi-horizon was
    enabled) and the DB ``file_path`` rows pointing at them remain valid.
    Other horizons use ``model.{forward_days}d.pkl``.
    """
    return "model.pkl" if forward_days == 5 else f"model.{forward_days}d.pkl"


def _ranking_features_filename(forward_days: int) -> str:
    """Per-horizon ranking feature-list filename within a date directory.

    Mirrors ``_ranking_model_filename`` / ``_direction_features_filename``:
    horizons share the ``{market}/{date}/`` dir, so the feature list must be
    keyed on ``forward_days`` -- otherwise the last horizon trained overwrites
    ``features.json`` for the others, and with feature selection ON each horizon
    selects a DIFFERENT subset, causing a silent positional mismatch at serving.

    The 5d horizon keeps the legacy ``features.json`` name so already-deployed
    5d models keep working with zero migration; other horizons use
    ``features.{forward_days}d.json``.
    """
    return "features.json" if forward_days == 5 else f"features.{forward_days}d.json"


def _ranking_train_dist_filename(forward_days: int) -> str:
    """Per-horizon PSI training-distribution filename within a date directory.

    Same horizon-keying rationale as ``_ranking_features_filename``; 5d keeps the
    legacy ``train_distribution.json`` name for zero-migration backward compat.
    """
    return (
        "train_distribution.json" if forward_days == 5
        else f"train_distribution.{forward_days}d.json"
    )


def _quality_marker_name(model_type: str, forward_days: int) -> str:
    """Marker filename, e.g. ``quality.ranking.5.json``.

    The forward_days segment future-proofs multiple horizons sharing a date
    directory; ranking and direction markers are independent and never
    overwrite each other.
    """
    return f"quality.{model_type}.{forward_days}.json"


def _write_quality_marker(
    model_dir: str, model_type: str, forward_days: int, quality: str,
) -> None:
    """Best-effort write of the on-disk quality marker.

    DB is authoritative (decision 3); the marker is a hot-path cache, so a
    write failure is logged and swallowed -- reads fall back to the DB.
    """
    path = os.path.join(model_dir, _quality_marker_name(model_type, forward_days))
    try:
        with open(path, "w") as f:
            json.dump({"quality": quality}, f)
    except OSError as e:
        logger.warning("Failed to write quality marker %s: %s", path, e)


def _read_quality_marker(
    model_dir: str, model_type: str, forward_days: int,
) -> Optional[str]:
    """Read the on-disk quality marker; returns None if absent/unreadable."""
    path = os.path.join(model_dir, _quality_marker_name(model_type, forward_days))
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        quality = data.get("quality")
        return quality if isinstance(quality, str) else None
    except (OSError, ValueError):
        return None


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
        # Trading-day guard (Batch D, decision 10): cheap weekend early-return
        # so the daily scheduler never produces weekend orphan rows across any
        # horizon or the direction model. force_retrain bypasses.
        cal_today = date.today()
        if cal_today.weekday() >= 5 and not force_retrain:
            skip_msg = (
                f"Non-trading day (weekend {cal_today.isoformat()}), "
                f"skipping to avoid orphan rows"
            )
            logger.info(
                "Multi-horizon: skipping prediction for %s: %s", market, skip_msg,
            )
            task.status = "completed"
            task.progress = 100.0
            task.completed_at = datetime.now()
            task.message = f"Skipped: {skip_msg}"
            task.results = {"market": market, "skipped": True, "reason": skip_msg}
            return
        # Align the direction-step prediction_date to the latest trading day
        # (the per-horizon ranking runs align independently inside
        # _run_prediction_async). Fail-open to today if calendar unavailable.
        prediction_date = self._resolve_prediction_date(market, cal_today)
        try:
            dir_results: dict[int, dict] = {}
            for i, h in enumerate(horizons):
                # Reserve the last 10% of the bar for the direction step so each
                # horizon trains BOTH the ranking model and its own direction
                # model (keyed on forward_days, so artifacts never collide).
                slice_size = 90.0 / n_horizons
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
                    _progress_range=slice_size * 0.85,
                    _skip_completion=True,
                )

                # Per-horizon direction model (non-fatal). Each horizon gets its
                # own up_probability on its own (market, date, forward_days)
                # rows -- _update_up_probabilities filters by forward_days.
                task.message = f"Direction model [{h}d] ({i + 1}/{n_horizons})"
                task.progress = pct_base + slice_size * 0.9
                dir_result = await self._run_direction_step(
                    task, market, force_retrain=force_retrain,
                    prediction_date=prediction_date, forward_days=h,
                )
                if dir_result:
                    dir_results[h] = dir_result

            task.status = "completed"
            task.progress = 100.0
            task.completed_at = datetime.now()
            task.message = f"Completed: {n_horizons} horizons"
            task.results = task.results or {}
            task.results["horizons"] = horizons
            if dir_results:
                # Keep the default-horizon (5d) result under the legacy key for
                # backward compatibility; expose all horizons under a map.
                if 5 in dir_results:
                    task.results["direction_model"] = dir_results[5]
                task.results["direction_models"] = dir_results

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
        forward_days: int = 5,
    ) -> Optional[dict]:
        try:
            from app.services.direction_service import train_and_predict_direction

            prev_msg = task.message
            task.message = f"Training direction model [{forward_days}d]"

            result = await train_and_predict_direction(
                market, force_retrain=force_retrain,
                prediction_date=prediction_date,
                forward_days=forward_days,
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

    @staticmethod
    def _resolve_default_horizon() -> int:
        """The horizon served when a caller does not specify forward_days.

        DEFAULT_TRADE_HORIZON (30d) if it is among the trained horizons, else the
        largest trained horizon, else the configured value. 30d is preferred over
        5d: highest decile spread and ~6x lower turnover (5d is net-negative after
        cost). Used so the cache-miss / store-fallback path serves a single
        deterministic horizon instead of a 5d/20d/30d-mixed ranking.
        """
        s = get_settings()
        trained = _get_prediction_horizons()
        if s.DEFAULT_TRADE_HORIZON in trained:
            return s.DEFAULT_TRADE_HORIZON
        return max(trained) if trained else s.DEFAULT_TRADE_HORIZON

    async def get_latest_predictions(
        self,
        market: str,
        top_n: int = 50,
        symbol: Optional[str] = None,
        forward_days: Optional[int] = None,
    ) -> list[dict]:
        # Resolve the default served horizon up front so the cache-miss / store
        # path serves a single horizon, not a forward_days=None mixed ranking.
        resolved_fd = (
            forward_days if forward_days is not None else self._resolve_default_horizon()
        )

        # 1. Try Redis cache (only for the default, unfiltered request -- the
        # per-market cache is refreshed to the default horizon by
        # _refresh_prediction_cache, so a hit already holds the default).
        if symbol is None and forward_days is None:
            cached = await self._read_prediction_cache(market)
            if cached is not None:
                logger.debug("Prediction cache hit: market=%s", market)
                return cached[:top_n]

        # 2. Query via local PredictionStore with the RESOLVED horizon (never
        # None -> never a 5d/20d/30d-mixed result on cache miss).
        try:
            return await prediction_store.get_latest_predictions(
                market, top_n, symbol=symbol, forward_days=resolved_fd,
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

    async def backfill_returns(
        self, markets: Optional[tuple[str, ...]] = None,
    ) -> dict:
        """Backfill actual returns for past predictions via local PredictionStore.

        Parameters
        ----------
        markets:
            Optional tuple of market codes to restrict the backfill to (e.g.
            ``("cn",)`` for the per-market chained backfill that runs right
            after a market's qlib sync). ``None`` backfills all markets (the
            daily fallback job). Unknown codes are ignored; empty/invalid
            input falls back to all markets for safety.
        """
        try:
            default_markets = ("us", "hk", "cn")
            if markets:
                selected = tuple(
                    m.lower() for m in markets if m and m.lower() in default_markets
                )
                target_markets = selected or default_markets
            else:
                target_markets = default_markets
            # Get candidates from the selected markets
            all_updates: list[dict] = []
            for mkt in target_markets:
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

    async def compute_sla_status(
        self, markets: tuple[str, ...] = ("us", "hk", "cn"),
    ) -> dict:
        """Compute per-market freshness SLA status (Batch D / decision 9).

        For each market compares:
          * freshest ``prediction_models.model_date`` age (calendar days)
            against ``MarketConfig.sla_max_model_age_days``;
          * freshest ``stock_predictions.prediction_date`` lag against
            ``MarketConfig.sla_max_prediction_lag_days``.

        Returns a structured dict suitable for persisting to Redis and for the
        admin scheduler surface. ``breach`` is True when ANY market exceeds
        either threshold. Per-market errors are isolated (one bad market does
        not abort the whole check). Each ``get_models`` / ``get_latest_*`` call
        opens its OWN session, so no concurrent-query-on-one-connection issue.
        """
        today = date.today()
        per_market: list[dict] = []
        any_breach = False

        for market in markets:
            entry: dict[str, Any] = {
                "market": market,
                "model_age_days": None,
                "max_model_age_days": None,
                "model_age_breach": False,
                "prediction_lag_days": None,
                "max_prediction_lag_days": None,
                "prediction_lag_breach": False,
                "latest_model_date": None,
                "latest_prediction_date": None,
                "error": None,
            }
            try:
                cfg = get_market_config(market)
                entry["max_model_age_days"] = cfg.sla_max_model_age_days
                entry["max_prediction_lag_days"] = cfg.sla_max_prediction_lag_days

                # --- freshest model date (sequential; own session) ---
                try:
                    models = await prediction_store.get_models(market)
                except Exception as exc:
                    models = []
                    logger.warning(
                        "SLA: failed to fetch models for %s: %s", market, exc,
                    )
                model_dates: list[date] = []
                for m in models or []:
                    mds = m.get("model_date")
                    if not mds:
                        continue
                    try:
                        model_dates.append(date.fromisoformat(mds))
                    except (ValueError, TypeError):
                        continue
                if model_dates:
                    latest_model = max(model_dates)
                    age = (today - latest_model).days
                    entry["latest_model_date"] = latest_model.isoformat()
                    entry["model_age_days"] = age
                    entry["model_age_breach"] = age > cfg.sla_max_model_age_days
                else:
                    # No model at all is itself an SLA breach.
                    entry["model_age_breach"] = True

                # --- freshest prediction date (sequential; own session) ---
                try:
                    preds = await prediction_store.get_latest_predictions(
                        market, top_n=1,
                    )
                except Exception as exc:
                    preds = []
                    logger.warning(
                        "SLA: failed to fetch predictions for %s: %s",
                        market, exc,
                    )
                pred_date: Optional[date] = None
                if preds:
                    pds = preds[0].get("prediction_date")
                    if pds:
                        try:
                            pred_date = date.fromisoformat(pds)
                        except (ValueError, TypeError):
                            pred_date = None
                if pred_date is not None:
                    lag = (today - pred_date).days
                    entry["latest_prediction_date"] = pred_date.isoformat()
                    entry["prediction_lag_days"] = lag
                    entry["prediction_lag_breach"] = (
                        lag > cfg.sla_max_prediction_lag_days
                    )
                else:
                    entry["prediction_lag_breach"] = True

            except Exception as exc:
                entry["error"] = str(exc)
                logger.warning("SLA: check failed for %s: %s", market, exc)

            if entry["model_age_breach"] or entry["prediction_lag_breach"]:
                any_breach = True
            per_market.append(entry)

        return {
            "checked_at": datetime.now().isoformat(),
            "breach": any_breach,
            "markets": per_market,
        }

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

            # Step 1.6: Trading-day guard (Batch D, decision 10).
            # The daily scheduler fires every calendar day, so without a guard
            # weekend/holiday runs write non-trading-day prediction_date rows
            # that can NEVER be backfilled (no close price for that date).
            #
            #  (a) Cheap weekend early-return: Sat/Sun is never a trading day in
            #      any of our markets -- skip without touching the calendar or
            #      doing any expensive work. force_retrain bypasses (an admin
            #      explicitly asked to retrain).
            #  (b) prediction_date alignment: for weekdays (incl. holidays) we
            #      align the written date to the latest trading day on-or-before
            #      today so every row is backfillable. Holidays therefore reuse
            #      the prior trading day's date (idempotent upsert).
            cal_today = date.today()
            if cal_today.weekday() >= 5 and not force_retrain:
                skip_msg = (
                    f"Non-trading day (weekend {cal_today.isoformat()}), "
                    f"skipping to avoid orphan rows"
                )
                logger.info("Skipping prediction for %s: %s", market, skip_msg)
                if _skip_completion:
                    task.message = f"[{forward_days}d] Skipped: {skip_msg}"
                    return
                task.status = "completed"
                task.progress = 100.0
                task.completed_at = datetime.now()
                task.message = f"Skipped: {skip_msg}"
                task.results = {"market": market, "skipped": True, "reason": skip_msg}
                return

            # Step 2: Check if retraining is needed
            settings = get_settings()
            binding = settings.QUALITY_GATE_BINDING
            # Align the prediction date to the actual data day (latest trading
            # day on-or-before today). Fail-open to today if the calendar is
            # unavailable. ``today`` below is therefore the aligned trading day.
            today = self._resolve_prediction_date(market, cal_today)
            model_id: Optional[Any] = None
            model_path: Optional[str] = None
            trained_this_run = False
            # serving_quality drives the low_confidence tag on written rows +
            # cache; default "approved" (existing-model reuse / passed gate).
            serving_quality: str = "approved"

            if not force_retrain:
                existing = self._check_existing_model_on_disk(market, today, forward_days)
                if existing is not None:
                    model_path = existing
                    logger.info("Existing model found for %s/%s", market, today.isoformat())
                    task.message = f"[{forward_days}d] Using existing model"
                    task.progress = _p(70)
                    # Resolve the reused model's id + quality so a rejected
                    # model already on disk is also subject to the serving
                    # policy below (binding) and the correct model_id is
                    # written.
                    if binding:
                        existing_quality = await prediction_store.get_model_quality(
                            market, today, forward_days, "ranking",
                        )
                        model_id = deterministic_model_id(
                            market, today, forward_days, "ranking",
                        ) if existing_quality is not None else None
                        if existing_quality == "rejected":
                            # Treat reuse of a rejected model exactly like a
                            # fresh rejected gate result.
                            serving_quality, model_id, model_path = (
                                await self._resolve_serving_after_reject(
                                    market, forward_days, today,
                                    rejected_model_id=model_id,
                                    rejected_model_path=model_path,
                                )
                            )

            # Step 3: Train if needed
            quality_passed = True
            if model_path is None:
                model_id, model_path, quality_passed = await self._train_model(
                    task, market, symbols, forward_days, today
                )
                trained_this_run = True

                if not quality_passed:
                    if binding:
                        # Decision 1: prefer the latest prior approved model;
                        # otherwise serve today's rejected model tagged
                        # low_confidence (never silently suppress).
                        serving_quality, model_id, model_path = (
                            await self._resolve_serving_after_reject(
                                market, forward_days, today,
                                rejected_model_id=model_id,
                                rejected_model_path=model_path,
                            )
                        )
                    else:
                        # Legacy behavior (binding disabled): fall back to the
                        # most recent model on disk regardless of quality.
                        fallback = self._find_latest_model_on_disk_legacy(
                            market, forward_days,
                        )
                        if fallback:
                            logger.warning(
                                "Quality gate failed -- legacy fallback to: %s",
                                fallback,
                            )
                            model_path = fallback
                        serving_quality = "rejected"

            # Step 4: Inference
            task.status = "predicting"
            task.progress = _p(75)
            task.message = f"[{forward_days}d] Running inference"

            prediction_count = await self._run_inference(
                task, market, symbols, model_id, model_path, forward_days, today,
                serving_quality=serving_quality,
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
                "serving_quality": serving_quality,
                "low_confidence": serving_quality == "rejected",
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

        Priority:
        1. Stored universe from system_settings (stable, admin-configured)
        2. Compute top-N by dollar volume from Qlib, auto-save to settings
        3. Market cap fallback via StockPulse

        After resolution, ``MarketConfig.excluded_symbols`` (e.g. the US ETF
        blocklist) is applied uniformly to EVERY return path -- including the
        stored-universe early return, which may have been auto-saved before
        the blocklist existed and is therefore ETF-polluted. When the blocklist
        is empty (CN/HK) this is a no-op.
        """
        cfg = get_market_config(market)
        excluded = cfg.excluded_symbols

        def _filter(syms: list[str]) -> list[str]:
            """Drop excluded tickers (case-insensitive). No-op if empty.

            US tickers are plain (no suffix); CN/HK universe symbols may carry
            suffixes (e.g. ``600000.SS``), but those markets have an empty
            blocklist so the comparison never accidentally strips them.
            """
            if not excluded:
                return syms
            filtered = [s for s in syms if s.upper() not in excluded]
            removed = len(syms) - len(filtered)
            if removed:
                logger.info(
                    "Excluded %d blocklisted symbol(s) from %s universe "
                    "(%d -> %d)",
                    removed, market, len(syms), len(filtered),
                )
            return filtered

        resolved = await self._resolve_symbols_raw(market, excluded=excluded)
        return _filter(resolved)

    async def _resolve_symbols_raw(
        self, market: str, excluded: frozenset[str] = frozenset(),
    ) -> list[str]:
        """Resolve the raw universe (pre-blocklist) for ``_resolve_symbols``.

        ``excluded`` is used here ONLY to prune the candidate pool BEFORE the
        top-N-by-dollar-volume / market-cap selection, so blocklisted ETFs do
        not consume universe slots. The caller still applies the final
        blocklist filter to every return path (stored universe included).
        """
        settings = get_settings()
        max_size = settings.PREDICTION_UNIVERSE_SIZE

        # 1. Check stored universe in system_settings
        stored = await self._get_stored_universe(market)
        if stored:
            logger.info(
                "Using stored universe for %s: %d symbols", market, len(stored),
            )
            return stored

        # 2. Compute from Qlib dollar volume and auto-save
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

        # Prune blocklisted ETFs from the candidate pool BEFORE top-N selection
        # so the top-N (and the auto-saved stored universe) is computed on
        # stocks only. No-op when the blocklist is empty (CN/HK).
        if excluded:
            all_symbols = [s for s in all_symbols if s.upper() not in excluded]

        if len(all_symbols) <= max_size:
            return all_symbols

        try:
            vol_map = await asyncio.get_running_loop().run_in_executor(
                None, self._get_dollar_volumes, market, all_symbols, settings,
            )
            if vol_map:
                sorted_syms = sorted(vol_map, key=vol_map.get, reverse=True)
                symbols = sorted_syms[:max_size]
                logger.info(
                    "Computed universe: top %d by dollar_volume for %s "
                    "(min=%.0f, max=%.0f). Saving to system_settings.",
                    len(symbols), market,
                    vol_map.get(symbols[-1], 0),
                    vol_map.get(symbols[0], 0),
                )
                await self._save_stored_universe(market, symbols)
                return symbols
        except Exception as e:
            logger.warning("Volume computation failed for %s: %s", market, e)

        # 3. Fallback: market cap
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
    async def _get_stored_universe(market: str) -> list[str] | None:
        """Read stored universe from system_settings."""
        from app.core.orm import get_session_factory
        from app.models.system_setting import SystemSetting
        from sqlalchemy import select
        import json as _json

        key = f"universe:{market}"
        factory = get_session_factory()
        async with factory() as session:
            result = await session.execute(
                select(SystemSetting.value).where(SystemSetting.key == key)
            )
            raw = result.scalar()

        if not raw:
            return None
        try:
            symbols = _json.loads(raw)
            if isinstance(symbols, list) and len(symbols) >= 10:
                return symbols
        except Exception:
            pass
        return None

    @staticmethod
    async def _save_stored_universe(market: str, symbols: list[str]) -> None:
        """Save universe to system_settings for future stability."""
        from app.core.orm import get_session_factory
        from app.models.system_setting import SystemSetting
        from sqlalchemy.dialects.postgresql import insert as pg_insert
        import json as _json

        key = f"universe:{market}"
        factory = get_session_factory()
        async with factory() as session:
            stmt = pg_insert(SystemSetting).values(
                key=key, value=_json.dumps(symbols),
            ).on_conflict_do_update(
                index_elements=["key"],
                set_={"value": _json.dumps(symbols)},
            )
            await session.execute(stmt)
            await session.commit()

    @staticmethod
    def _get_dollar_volumes(
        market: str, symbols: list[str], settings,
    ) -> dict[str, float]:
        """Compute 60-day average dollar volume from Qlib (synchronous).

        Dollar volume (price × volume) is a better liquidity proxy than
        raw share volume, since it normalizes across different price levels.
        """
        from app.context import QlibContext
        from app.utils.symbol_mapping import normalize_symbol_for_qlib

        QlibContext.ensure_init(market, settings.QLIB_DATA_DIR)
        from qlib.data import D
        from datetime import date, timedelta

        end = date.today().isoformat()
        start = (date.today() - timedelta(days=90)).isoformat()

        qlib_syms = [normalize_symbol_for_qlib(s, market) for s in symbols]
        q_to_ws = dict(zip(qlib_syms, symbols))

        result: dict[str, float] = {}
        chunk_size = 500
        for i in range(0, len(qlib_syms), chunk_size):
            chunk = qlib_syms[i : i + chunk_size]
            try:
                df = D.features(chunk, ["$volume", "$close"], start_time=start, end_time=end)
                if df.empty:
                    continue
                avg = df.groupby(level=0).mean()
                for q_sym in avg.index:
                    ws_sym = q_to_ws.get(q_sym, q_sym)
                    vol = avg.loc[q_sym, "$volume"]
                    close = avg.loc[q_sym, "$close"]
                    if vol > 0 and close > 0:
                        result[ws_sym] = float(vol * close)
            except Exception as e:
                logger.warning("Qlib volume chunk %d failed: %s", i, e)

        return result

    # ------------------------------------------------------------------
    # Step 2: Check existing model (disk only)
    # ------------------------------------------------------------------

    def _check_existing_model_on_disk(
        self, market: str, model_date: date, forward_days: int,
    ) -> Optional[str]:
        """Check if a model already exists on disk for today + this horizon.

        Keyed on ``forward_days`` (via ``_ranking_model_filename``) so a
        multi-horizon run does not reuse a different horizon's artifact.
        """
        settings = get_settings()
        date_str = model_date.strftime("%Y%m%d")
        model_path = os.path.join(
            settings.PREDICTION_DATA_DIR, market, date_str,
            _ranking_model_filename(forward_days),
        )
        if os.path.exists(model_path):
            return model_path
        return None

    async def _find_latest_model_on_disk(
        self,
        market: str,
        forward_days: int,
        exclude_date: Optional[date] = None,
        model_type: str = "ranking",
    ) -> Optional[str]:
        """Find the latest *approved* model file on disk for a market.

        Quality-aware fallback resolver (Batch A). Walks date directories
        newest-first and returns the first one that:
          1. is not ``exclude_date`` (e.g. today's just-rejected model);
          2. has the expected model file (ranking=model.pkl,
             direction=direction_model.pkl);
          3. is ``approved`` -- checked via the on-disk marker first, then
             falling back to the DB (decision 3) for the ~38 legacy
             marker-less directories.

        Returns the model file path, or None when no approved model exists.

        NOTE: when QUALITY_GATE_BINDING is disabled, callers should use the
        legacy resolver semantics; this method always enforces approved-only.
        """
        settings = get_settings()
        market_dir = os.path.join(settings.PREDICTION_DATA_DIR, market)
        if not os.path.isdir(market_dir):
            return None

        try:
            date_dirs = sorted(os.listdir(market_dir), reverse=True)
        except OSError:
            return None

        if model_type == "direction":
            from app.services.direction_service import _direction_model_filename
            model_filename = _direction_model_filename(forward_days)
        else:
            model_filename = _ranking_model_filename(forward_days)
        exclude_str = exclude_date.strftime("%Y%m%d") if exclude_date is not None else None

        for dirname in date_dirs:
            if exclude_str is not None and dirname == exclude_str:
                continue

            dir_path = os.path.join(market_dir, dirname)
            model_path = os.path.join(dir_path, model_filename)
            if not os.path.exists(model_path):
                continue

            # Resolve the model date from the directory name.
            try:
                dir_date = date(int(dirname[:4]), int(dirname[4:6]), int(dirname[6:8]))
            except (ValueError, IndexError):
                continue

            # 1) marker (hot path)
            quality = _read_quality_marker(dir_path, model_type, forward_days)
            # 2) DB fallback when marker absent or non-terminal
            if quality not in ("approved", "rejected"):
                try:
                    quality = await prediction_store.get_model_quality(
                        market, dir_date, forward_days, model_type,
                    )
                except Exception as e:
                    logger.warning(
                        "DB quality lookup failed for %s/%s: %s",
                        market, dirname, e,
                    )
                    quality = None

            if quality == "approved":
                return model_path

        return None

    def _find_latest_model_on_disk_legacy(
        self, market: str, forward_days: int,
    ) -> Optional[str]:
        """Legacy (quality-blind) disk resolver -- newest ranking model.

        Used only when QUALITY_GATE_BINDING is disabled, to preserve the
        original behavior as a one-flag revert path. Keyed on ``forward_days``
        so multi-horizon runs do not cross-serve a different horizon's model.
        """
        settings = get_settings()
        market_dir = os.path.join(settings.PREDICTION_DATA_DIR, market)
        if not os.path.isdir(market_dir):
            return None
        try:
            date_dirs = sorted(os.listdir(market_dir), reverse=True)
        except OSError:
            return None
        model_filename = _ranking_model_filename(forward_days)
        for dirname in date_dirs:
            model_path = os.path.join(market_dir, dirname, model_filename)
            if os.path.exists(model_path):
                return model_path
        return None

    async def _resolve_serving_after_reject(
        self,
        market: str,
        forward_days: int,
        model_date: date,
        rejected_model_id: Optional[Any],
        rejected_model_path: Optional[str],
    ) -> tuple[str, Optional[Any], Optional[str]]:
        """Resolve the serving model when today's candidate is rejected.

        Implements decision 1 (serve-but-tag-low-confidence):
          (2a) Prefer the latest prior *approved* model (DB authoritative for
               id + file_path; disk for the actual artifact). The serving rows
               are written with the approved model's id, so the read-path JOIN
               derives ``quality='approved'``.
          (2b) When no approved model has ever existed for this market, serve
               today's rejected model, returning ``serving_quality='rejected'``
               so written rows + cache are tagged ``low_confidence``.

        Returns ``(serving_quality, model_id, model_path)``.
        """
        approved = await prediction_store.get_latest_approved_model(
            market, model_type="ranking", forward_days=forward_days,
            before_date=model_date,
        )
        if approved is not None:
            approved_path = approved.get("file_path")
            approved_id = approved.get("id")
            # Tolerate a missing/cleaned artifact: fall back to a disk scan for
            # any prior approved model before giving up to the rejected one.
            if not approved_path or not os.path.exists(approved_path):
                disk_path = await self._find_latest_model_on_disk(
                    market, forward_days, exclude_date=model_date,
                    model_type="ranking",
                )
                if disk_path is not None:
                    logger.warning(
                        "Approved model artifact missing (%s); using disk "
                        "approved model: %s",
                        approved_path, disk_path,
                    )
                    return "approved", approved_id, disk_path
            else:
                logger.warning(
                    "Quality gate failed for %s/%s -- serving prior approved "
                    "model (id=%s, date=%s)",
                    market, model_date.isoformat(),
                    approved_id, approved.get("model_date"),
                )
                return "approved", approved_id, approved_path

        # (2b) No approved model anywhere -> serve today's rejected, tagged.
        logger.warning(
            "Quality gate failed for %s/%s and no prior approved model "
            "exists -- serving today's rejected model tagged low_confidence",
            market, model_date.isoformat(),
        )
        return "rejected", rejected_model_id, rejected_model_path

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

        # Preserve the RAW (winsorized, NOT sector-neutral) forward return for
        # the net-cost gate's economic spread. forward_return is overwritten
        # in-place by the sector-neutral demean below (US) -- a training LABEL
        # transform, not tradeable P&L. The gate nets a RAW bps cost against the
        # spread, so the spread must be in raw return units too (same basis).
        df["raw_forward_return"] = df["forward_return"]

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

        meta_cols = {"symbol", "date", "close", "forward_return", "raw_forward_return", "label"}
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

        # Leakage-safe feature selection (flag-gated; default OFF). Computed
        # ONLY on the first fold's training dates -- a strict prefix that ends
        # forward_days before the earliest validation date, so it never sees any
        # walk-forward validation window. Reducing feature_cols here (before the
        # fold loop) makes every downstream consumer (datasets, importance, PSI,
        # _save_model) follow automatically.
        feature_cols, selection_diag = self._maybe_select_features(
            df, feature_cols, splits, market, cfg,
        )

        fold_ics: list[float] = []
        fold_icirs: list[float] = []
        # Per-fold gross quintile spread + rebalance turnover for the net-of-cost
        # quality gate (computed from each fold's own validation rows + realized
        # forward_return -- leakage-safe).
        fold_gross_spreads: list[float] = []
        fold_turnovers: list[float] = []
        # Per-date IC series for each fold. Walk-forward validation windows are
        # non-overlapping (see _walk_forward_splits / _evaluate_quality_gate
        # docstring), so these can be pooled into one distribution for the
        # significance gate. Stop discarding the series (was "_" previously).
        all_daily_ics: list[pd.Series] = []
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

            fold_ic_series, fold_ic, fold_icir = self._compute_ic_metrics(va_df, va_scores, va_actual)
            fold_ics.append(fold_ic)
            fold_icirs.append(fold_icir)
            all_daily_ics.append(fold_ic_series)

            # Gate spread uses RAW returns (va_actual is sector-neutral for US,
            # a label transform -- not tradeable P&L; cost is in raw bps).
            fold_gross, fold_turnover, _ = self._compute_fold_spread_turnover(
                va_df, va_scores, va_df["raw_forward_return"].values, forward_days,
            )
            fold_gross_spreads.append(fold_gross)
            fold_turnovers.append(fold_turnover)

            logger.info(
                "  Fold %d IC=%.4f, ICIR=%.4f, gross_spread=%.4f, turnover=%.3f",
                fold_idx + 1, fold_ic, fold_icir, fold_gross, fold_turnover,
            )

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

        # Quality gate (shared helper -- same logic as train_for_backtest).
        # In shadow mode (QUALITY_GATE_ENFORCE_SIGNIFICANCE=false, default) the
        # binding approved/rejected decision is the LEGACY gate; the new
        # significance gate is computed and logged for calibration. When the
        # flag is true the significance gate becomes binding.
        quality_passed, gate_metrics = self._evaluate_quality_gate(
            fold_ics, fold_icirs, all_daily_ics, cfg,
            fold_gross_spreads=fold_gross_spreads,
            fold_turnovers=fold_turnovers,
            forward_days=forward_days,
        )

        if not quality_passed:
            logger.warning(
                "Model quality gate FAILED (binding=%s): mean_IC=%.4f (min=%.4f), "
                "mean_ICIR=%.4f (min=%.4f)",
                "significance" if gate_metrics["enforce_significance"] else "legacy",
                ic_mean, cfg.min_ic_threshold, icir, cfg.min_icir_threshold,
            )
        else:
            logger.info(
                "Model quality gate passed (binding=%s): mean_IC=%.4f, mean_ICIR=%.4f",
                "significance" if gate_metrics["enforce_significance"] else "legacy",
                ic_mean, icir,
            )

        # Always emit the significance-gate shadow verdict for calibration.
        self._log_significance_shadow(market, forward_days, fold_ics, gate_metrics, cfg)
        # Net-of-cost gate one-line summary (shadow or enforced).
        self._log_net_cost_gate(market, forward_days, gate_metrics)

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

        # Save model to disk (marker starts as "pending")
        task.message = "Saving model"
        task.progress = 60.0
        model_path = self._save_model(
            models, market, model_date, feature_cols, feature_importance,
            model_type="ranking", forward_days=forward_days,
            selection_diag=selection_diag,
        )

        # Rewrite the on-disk quality marker now that the gate has decided.
        # DB remains authoritative (decision 3); this is a hot-path cache so
        # the rewrite is best-effort (handled inside _write_quality_marker).
        _write_quality_marker(
            os.path.dirname(model_path), "ranking", forward_days,
            "approved" if quality_passed else "rejected",
        )

        # Save training distribution snapshot for PSI
        try:
            self._save_train_distribution(
                final_train_df, feature_cols, os.path.dirname(model_path),
                forward_days=forward_days,
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
                # Significance-gate diagnostics (Batch C, decision 7/8). DB `ic`
                # column stays = mean(fold_ics); these are stored in
                # training_config for calibration and historical recompute.
                "pooled_icir": round(gate_metrics["pooled_icir"], 6),
                "t_stat": round(gate_metrics["t_stat"], 6),
                "n_validation_dates": gate_metrics["n_validation_dates"],
                "min_fold_ic": round(gate_metrics["min_fold_ic"], 6),
                "pooled_ic_mean": round(gate_metrics["pooled_ic_mean"], 6),
                "daily_ics": gate_metrics["daily_ics"],
                "significance_gate_passed": gate_metrics["significance_passed"],
                "legacy_gate_passed": gate_metrics["legacy_passed"],
                "significance_gate_enforced": gate_metrics["enforce_significance"],
                # Net-of-cost / turnover gate diagnostics (shadow or enforced).
                "mean_gross_spread": round(gate_metrics["mean_gross_spread"], 6),
                "mean_turnover": round(gate_metrics["mean_turnover"], 6),
                "mean_net_spread": round(gate_metrics["mean_net_spread"], 6),
                "net_cost_gate_passed": gate_metrics["net_cost_passed"],
                "net_cost_gate_enabled": gate_metrics["net_cost_gate_enabled"],
                "fold_gross_spreads": [round(s, 6) for s in fold_gross_spreads],
                "fold_turnovers": [round(t, 6) for t in fold_turnovers],
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

        # Preserve RAW (winsorized, non-sector-neutral) return for the net-cost
        # gate spread -- same basis as the raw bps cost (see _train_model).
        df["raw_forward_return"] = df["forward_return"]

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

        meta_cols = {"symbol", "date", "close", "forward_return", "raw_forward_return", "label"}
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

        # Leakage-safe feature selection (flag-gated; default OFF). Same first-
        # fold-training-window restriction as the production path -- see
        # _maybe_select_features. config carries the per-market thresholds.
        feature_cols, selection_diag = self._maybe_select_features(
            df, feature_cols, splits, market, config,
        )

        fold_ics: list[float] = []
        fold_icirs: list[float] = []
        # Per-fold gross spread + rebalance turnover (net-cost gate, same as the
        # production path -- shared helper so the two cannot drift).
        fold_gross_spreads: list[float] = []
        fold_turnovers: list[float] = []
        # Per-date IC series per fold (non-overlapping windows -> poolable).
        all_daily_ics: list[pd.Series] = []
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
            va_actual = va_df["forward_return"].values
            fold_ic_series, fold_ic, fold_icir = self._compute_ic_metrics(
                va_df, va_scores, va_actual,
            )
            fold_ics.append(fold_ic)
            fold_icirs.append(fold_icir)
            all_daily_ics.append(fold_ic_series)

            # Gate spread uses RAW returns (va_actual is sector-neutral for US,
            # a label transform -- not tradeable P&L; cost is in raw bps).
            fold_gross, fold_turnover, _ = self._compute_fold_spread_turnover(
                va_df, va_scores, va_df["raw_forward_return"].values, forward_days,
            )
            fold_gross_spreads.append(fold_gross)
            fold_turnovers.append(fold_turnover)

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

        # Shared quality-gate helper (same logic as production _train_model)
        # to prevent the two paths drifting. config is the per-market
        # MarketConfig (may carry backtest overrides). Thread gross spread +
        # turnover so the net-cost gate matches the production path.
        quality_passed, gate_metrics = self._evaluate_quality_gate(
            fold_ics, fold_icirs, all_daily_ics, config,
            fold_gross_spreads=fold_gross_spreads,
            fold_turnovers=fold_turnovers,
            forward_days=forward_days,
        )
        self._log_net_cost_gate(market, forward_days, gate_metrics)

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
            "pooled_icir": gate_metrics["pooled_icir"],
            "t_stat": gate_metrics["t_stat"],
            "n_validation_dates": gate_metrics["n_validation_dates"],
            "min_fold_ic": gate_metrics["min_fold_ic"],
            "mean_gross_spread": gate_metrics["mean_gross_spread"],
            "mean_turnover": gate_metrics["mean_turnover"],
            "mean_net_spread": gate_metrics["mean_net_spread"],
            "net_cost_gate_passed": gate_metrics["net_cost_passed"],
            "net_cost_gate_enabled": gate_metrics["net_cost_gate_enabled"],
            "ensemble_size": ensemble_size,
            "symbol_count": df["symbol"].nunique(),
            "feature_count": len(feature_cols),
            "selection_diag": selection_diag,
        }

    @staticmethod
    def _maybe_select_features(
        df: pd.DataFrame,
        feature_cols: list[str],
        splits: list[tuple[list, list]],
        market: str,
        cfg: MarketConfig,
    ) -> tuple[list[str], dict | None]:
        """Run leakage-safe feature selection when both flags are enabled.

        Feature selection is applied to the RANKING model ONLY by design; the
        direction classifier keeps its full feature set and persists its own
        per-horizon ``direction_features.{fd}d.json`` independently.

        Selection is computed ONLY on ``splits[0][0]`` -- the first walk-forward
        fold's TRAINING dates. With expanding-window splits this is a strict
        prefix that ends ``forward_days`` before the earliest validation date,
        so it never overlaps ANY validation fold. Computing IC/correlation on
        the full df would leak validation folds into selection and falsely
        inflate walk-forward IC -- the exact failure mode we must avoid.

        Returns ``(feature_cols, diagnostics)``. ``diagnostics`` is the
        selection record (kept/input/dropped_*/corr_threshold/min_abs_ic) when
        selection ran, else ``None``. No-op (returns the input unchanged, diag
        None) when the flag is off, splits are empty, or the selection window
        has too few unique dates.
        """
        settings = get_settings()
        if not (settings.FEATURE_SELECTION_ENABLED and cfg.use_feature_selection):
            return feature_cols, None

        if not splits:
            logger.warning("Feature selection skipped: no walk-forward splits")
            return feature_cols, None

        selection_dates = set(splits[0][0])
        sel_df = df[df["date"].isin(selection_dates)]
        n_dates = sel_df["date"].nunique()
        if n_dates < 20:
            logger.warning(
                "Feature selection skipped: only %d selection dates (<20)",
                n_dates,
            )
            return feature_cols, None

        selected, diag = feature_service.select_features(
            sel_df,
            feature_cols,
            label_col="forward_return",
            corr_threshold=cfg.feature_selection_corr_threshold,
            min_abs_ic=cfg.feature_selection_min_abs_ic,
        )

        dropped = diag.get("dropped_redundant", []) + diag.get("dropped_low_ic", [])
        logger.info(
            "Feature selection [%s]: kept %d/%d (dropped %d redundant, %d low-IC) "
            "on %d selection dates; dropped=%s",
            market,
            len(selected),
            len(feature_cols),
            len(diag.get("dropped_redundant", [])),
            len(diag.get("dropped_low_ic", [])),
            n_dates,
            dropped[:15] if len(dropped) <= 15 else f"{dropped[:15]} (+{len(dropped) - 15} more)",
        )
        return selected, diag

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
    def _compute_fold_spread_turnover(
        val_df: pd.DataFrame,
        predicted_scores: np.ndarray,
        actual_returns: np.ndarray,
        forward_days: int,
    ) -> tuple[float, float, int]:
        """Gross quintile spread + rebalance turnover for ONE validation fold.

        Shared by ``_train_model`` and ``train_for_backtest`` so the net-cost
        gate cannot drift between the two paths. Uses ONLY the validation
        fold's own dates and its realized ``forward_return`` label (already the
        ``forward_days``-ahead return used for training), so it is leakage-safe.

        Quintile bucketing matches ``ml_backtest_service._compute_validation_metrics``
        (``pd.qcut(score.rank(method="first"), q=5, labels=[1..5])`` per date,
        Q5 = best predicted, Q1 = worst), but the spread is averaged PER-DATE (not
        pooled across all rows) and uses the RAW return the caller supplies as
        ``actual_returns`` -- the net-cost gate passes ``raw_forward_return``, NOT
        the sector-neutral training label, so spread units match the bps cost.

        Returns ``(gross_spread, turnover_per_rebalance, n_rebalances)``:
          * gross_spread -- mean over validation dates of
            (mean realized return of Q5) - (mean realized return of Q1).
          * turnover_per_rebalance -- average fraction of names REPLACED in the
            long (Q5) and short (Q1) baskets between consecutive REBALANCE dates,
            rebalancing every ``forward_days`` trading dates (NOT daily). For each
            consecutive rebalance pair: ``1 - |kept| / |basket|``, averaged over
            the long and short legs and over all rebalance transitions. 0.0 when
            there is fewer than one transition.
          * n_rebalances -- number of rebalance transitions used for turnover
            (0 when undefined).
        """
        if val_df.empty or len(predicted_scores) == 0:
            return 0.0, 0.0, 0

        tmp = val_df[["date", "symbol"]].copy()
        tmp["pred"] = predicted_scores
        tmp["actual"] = actual_returns
        tmp = tmp.dropna(subset=["actual", "pred"])
        if tmp.empty:
            return 0.0, 0.0, 0

        # Per-date quintiles (need >= 5 names/date to form 5 buckets).
        def _q(x: "pd.Series") -> "pd.Series":
            if len(x) < 5:
                return pd.Series([np.nan] * len(x), index=x.index)
            return pd.qcut(
                x.rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]
            )

        tmp["quintile"] = tmp.groupby("date")["pred"].transform(_q)
        tmp = tmp.dropna(subset=["quintile"])
        if tmp.empty:
            return 0.0, 0.0, 0
        tmp["quintile"] = tmp["quintile"].astype(int)

        # --- Gross spread: per-date (Q5 mean - Q1 mean), averaged over dates ---
        per_date = tmp.groupby(["date", "quintile"])["actual"].mean().unstack("quintile")
        if 5 in per_date.columns and 1 in per_date.columns:
            date_spread = (per_date[5] - per_date[1]).dropna()
            gross_spread = float(date_spread.mean()) if len(date_spread) > 0 else 0.0
        else:
            gross_spread = 0.0

        # --- Turnover at the rebalance cadence (every forward_days dates) ---
        ordered_dates = sorted(tmp["date"].unique())
        step = max(1, int(forward_days))
        rebalance_dates = ordered_dates[::step]

        # Basket membership (set of symbols) per rebalance date for each leg.
        def _leg_sets(q_val: int) -> list[set]:
            sets: list[set] = []
            for d in rebalance_dates:
                names = set(
                    tmp[(tmp["date"] == d) & (tmp["quintile"] == q_val)]["symbol"]
                )
                sets.append(names)
            return sets

        long_sets = _leg_sets(5)
        short_sets = _leg_sets(1)

        leg_turnovers: list[float] = []
        for sets in (long_sets, short_sets):
            for i in range(1, len(sets)):
                prev, cur = sets[i - 1], sets[i]
                if not cur or not prev:
                    continue
                kept = len(cur & prev)
                leg_turnovers.append(1.0 - kept / len(cur))

        n_rebalances = max(0, len(rebalance_dates) - 1)
        turnover = float(np.mean(leg_turnovers)) if leg_turnovers else 0.0
        return gross_spread, turnover, n_rebalances

    @staticmethod
    def _evaluate_quality_gate(
        fold_ics: list[float],
        fold_icirs: list[float],
        all_daily_ics: list["pd.Series"],
        cfg: MarketConfig,
        fold_gross_spreads: list[float] | None = None,
        fold_turnovers: list[float] | None = None,
        forward_days: int = 0,
    ) -> tuple[bool, dict[str, Any]]:
        """Evaluate both the legacy and the significance quality gates.

        Shared by the production trainer (``_train_model``) and the backtest
        trainer (``train_for_backtest``) so the two paths cannot drift.

        Returns ``(passed, metrics)`` where ``passed`` is the *binding*
        approved/rejected decision: it honours
        ``QUALITY_GATE_ENFORCE_SIGNIFICANCE`` (config). When that flag is false
        (default = SHADOW mode) the binding decision is the LEGACY gate
        (``ic_mean > min_ic AND icir > min_icir``); the significance gate is
        still computed and returned in ``metrics`` (and logged by the caller)
        for calibration. When the flag is true the binding decision is the
        SIGNIFICANCE gate.

        Pooling validity: walk-forward validation windows produced by
        ``_walk_forward_splits`` are NON-OVERLAPPING (each fold's val_dates is a
        disjoint ``unique_dates[val_start_idx:val_end_idx]`` slice with no shared
        dates), so concatenating their per-date IC series yields an unbiased
        pooled distribution with N = sum of per-fold validation-date counts.
        ``metrics`` keys:
            ic_mean, icir (legacy = mean(fold_ics)/mean(fold_icirs)),
            pooled_ic_mean, pooled_icir, t_stat, n_validation_dates,
            min_fold_ic, daily_ics (list[float], per-date IC pooled across
            folds -- persisted per decision 8), legacy_passed,
            significance_passed, enforce_significance,
            mean_gross_spread, mean_turnover, mean_net_spread,
            net_cost_gate_enabled, net_cost_passed.

        Net-cost gate (flag-gated by NET_COST_GATE_ENABLED, default OFF):
        ``mean_net_spread = mean_gross_spread - mean_turnover * 2 * cost`` where
        ``cost = TRADING_COST_BPS_ONEWAY / 10000`` (the 2x is the round trip:
        exit old + enter new). When enabled, ``mean_net_spread > 0`` is AND-ed
        onto the binding decision. When disabled the net metrics are computed,
        returned, and logged but do NOT affect ``passed`` (SHADOW), so default-OFF
        is byte-identical to the prior behavior.
        """
        settings = get_settings()
        enforce = settings.QUALITY_GATE_ENFORCE_SIGNIFICANCE

        # --- Legacy gate (mean of per-fold IC / per-fold ICIR) ---
        ic_mean = float(np.mean(fold_ics)) if fold_ics else 0.0
        icir = float(np.mean(fold_icirs)) if fold_icirs else 0.0
        min_ic = cfg.min_ic_threshold
        min_icir = cfg.min_icir_threshold
        legacy_passed = bool(ic_mean > min_ic and icir > min_icir)

        # --- Significance gate (pooled IC + per-fold lower bound + N + t) ---
        # Pool the per-date IC series across all folds. The windows are
        # non-overlapping (see docstring), so this is a clean concatenation.
        valid_series = [s for s in all_daily_ics if s is not None and len(s) > 0]
        if valid_series:
            pooled = pd.concat(valid_series)
            pooled = pooled.dropna()
        else:
            pooled = pd.Series(dtype=float)

        n_validation_dates = int(len(pooled))
        if n_validation_dates > 0:
            pooled_ic_mean = float(pooled.mean())
        else:
            pooled_ic_mean = 0.0
        # Population-corrected std (ddof=1) for the t-statistic.
        if n_validation_dates >= 2:
            pooled_std = float(pooled.std(ddof=1))
        else:
            pooled_std = 0.0
        if pooled_std > 1e-10:
            pooled_icir = pooled_ic_mean / pooled_std
            t_stat = pooled_icir * math.sqrt(n_validation_dates)
        else:
            pooled_icir = 0.0
            t_stat = 0.0

        min_fold_ic = float(min(fold_ics)) if fold_ics else 0.0
        folds_all_positive = (not cfg.require_all_folds_positive) or (min_fold_ic > 0)

        significance_passed = bool(
            pooled_ic_mean > min_ic
            and folds_all_positive
            and n_validation_dates >= cfg.min_validation_days
            and t_stat >= cfg.min_t_stat
        )

        base_passed = significance_passed if enforce else legacy_passed

        # --- Net-of-cost / turnover gate (flag-gated, default SHADOW) ---
        # Aggregate the per-fold gross spread + turnover into fold means, then
        # convert to a turnover-adjusted (net) spread on the same per-rebalance
        # basis. When the flag is OFF these are computed and returned but do NOT
        # alter the binding decision, so default-OFF is byte-identical.
        net_cost_enabled = bool(settings.NET_COST_GATE_ENABLED)
        cost_oneway = float(settings.TRADING_COST_BPS_ONEWAY) / 10000.0
        if fold_gross_spreads:
            mean_gross_spread = float(np.mean(fold_gross_spreads))
        else:
            mean_gross_spread = 0.0
        if fold_turnovers:
            mean_turnover = float(np.mean(fold_turnovers))
        else:
            mean_turnover = 0.0
        # Cost drag per rebalance = 4 x turnover x one-way cost. The 4 = two
        # legs (long Q5 + short Q1) x round-trip (exit replaced names + enter new
        # ones). mean_turnover is the AVERAGE one-way replacement fraction across
        # the two legs, so (f_long + f_short) = 2 x mean_turnover, and round-trip
        # doubles that again: cost x 2(round-trip) x 2(legs) x mean_turnover.
        mean_net_spread = mean_gross_spread - mean_turnover * 4.0 * cost_oneway

        # Annualize to a calendar-comparable basis so rebalance FREQUENCY is
        # penalized: a 5d model rebalances ~252/5 times/yr vs ~252/30 for 30d, so
        # the same per-rebalance turnover is far costlier annually. annual_net has
        # the SAME SIGN as the per-rebalance net (so it doesn't change the net>0
        # test); its value is the cross-horizon comparison + the turnover ceiling.
        ann_factor = (
            settings.TRADING_DAYS_PER_YEAR / max(1, int(forward_days))
            if forward_days else 0.0
        )
        annual_net_spread = mean_net_spread * ann_factor
        annual_turnover = mean_turnover * ann_factor
        # Turnover ceiling: reject churny models on an ANNUAL basis (the real 5d
        # penalty). Default MAX_ANNUAL_TURNOVER is high (effectively off) -- opt
        # in by lowering it after inspecting the shadow annual_turnover logs.
        turnover_ok = (ann_factor == 0.0) or (
            annual_turnover <= float(settings.MAX_ANNUAL_TURNOVER)
        )
        net_cost_passed = bool(mean_net_spread > 0.0 and turnover_ok)

        # Default-OFF must be a no-op: only AND the net requirement when enabled.
        passed = bool(base_passed and net_cost_passed) if net_cost_enabled else base_passed

        metrics: dict[str, Any] = {
            "ic_mean": ic_mean,
            "icir": icir,
            "pooled_ic_mean": pooled_ic_mean,
            "pooled_icir": pooled_icir,
            "t_stat": t_stat,
            "n_validation_dates": n_validation_dates,
            "min_fold_ic": min_fold_ic,
            # Per-date IC pooled across folds (decision 8: persist the series).
            "daily_ics": [round(float(v), 6) for v in pooled.tolist()],
            "legacy_passed": legacy_passed,
            "significance_passed": significance_passed,
            "enforce_significance": bool(enforce),
            # Net-of-cost / turnover gate diagnostics (persisted + shadow-logged).
            "mean_gross_spread": mean_gross_spread,
            "mean_turnover": mean_turnover,
            "mean_net_spread": mean_net_spread,
            "annual_net_spread": annual_net_spread,
            "annual_turnover": annual_turnover,
            "net_cost_gate_enabled": net_cost_enabled,
            "net_cost_passed": net_cost_passed,
        }
        return passed, metrics

    @staticmethod
    def _log_significance_shadow(
        market: str,
        forward_days: int,
        fold_ics: list[float],
        metrics: dict[str, Any],
        cfg: MarketConfig,
    ) -> None:
        """Emit a structured log of what the NEW significance gate would decide.

        Used in shadow mode (and harmless when enforcing). Reports the verdict,
        t-statistic, N, and -- when a fold is negative -- which fold(s) failed
        the per-fold-positive requirement, so the gate can be calibrated.
        """
        negative_folds = [i for i, v in enumerate(fold_ics) if v <= 0]
        logger.info(
            "[significance-gate %s] market=%s fwd=%d would_decide=%s "
            "t_stat=%.3f (min=%.2f) pooled_ic=%.4f (min=%.4f) pooled_icir=%.4f "
            "N=%d (min=%d) min_fold_ic=%.4f negative_folds=%s",
            "ENFORCED" if metrics.get("enforce_significance") else "SHADOW",
            market,
            forward_days,
            "PASS" if metrics.get("significance_passed") else "FAIL",
            metrics.get("t_stat", 0.0),
            cfg.min_t_stat,
            metrics.get("pooled_ic_mean", 0.0),
            cfg.min_ic_threshold,
            metrics.get("pooled_icir", 0.0),
            metrics.get("n_validation_dates", 0),
            cfg.min_validation_days,
            metrics.get("min_fold_ic", 0.0),
            negative_folds if negative_folds else "none",
        )

    @staticmethod
    def _log_net_cost_gate(
        market: str,
        forward_days: int,
        metrics: dict[str, Any],
    ) -> None:
        """One-line summary of the net-of-cost / turnover gate.

        Reports gross spread, rebalance turnover, the turnover-adjusted net
        spread, and whether the net-cost gate would (SHADOW) or did (ENFORCED)
        reject. Harmless when the gate is disabled -- it is purely informational
        in that case.
        """
        enabled = bool(metrics.get("net_cost_gate_enabled"))
        net_passed = bool(metrics.get("net_cost_passed"))
        logger.info(
            "[net-cost-gate %s] market=%s fwd=%d would_reject=%s "
            "gross_spread=%.4f turnover=%.3f net_spread=%.4f "
            "annual_net=%.4f annual_turnover=%.2f",
            "ENFORCED" if enabled else "SHADOW",
            market,
            forward_days,
            "no" if net_passed else "yes",
            metrics.get("mean_gross_spread", 0.0),
            metrics.get("mean_turnover", 0.0),
            metrics.get("mean_net_spread", 0.0),
            metrics.get("annual_net_spread", 0.0),
            metrics.get("annual_turnover", 0.0),
        )

    @staticmethod
    def _save_model(
        models: list[lgb.Booster] | lgb.Booster,
        market: str,
        model_date: date,
        feature_cols: list[str],
        feature_importance: dict[str, float] | None = None,
        model_type: str = "ranking",
        forward_days: int = 5,
        selection_diag: dict | None = None,
    ) -> str:
        if isinstance(models, lgb.Booster):
            models = [models]

        settings = get_settings()
        date_str = model_date.strftime("%Y%m%d")
        model_dir = Path(settings.PREDICTION_DATA_DIR) / market / date_str
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = str(model_dir / _ranking_model_filename(forward_days))
        joblib.dump(models, model_path)

        features_meta: dict[str, Any] = {
            "features": feature_cols,
            "count": len(feature_cols),
            "ensemble_size": len(models),
        }
        if feature_importance is not None:
            features_meta["feature_importance"] = feature_importance
        # Persist what feature selection removed (best-effort; absent when
        # selection didn't run, so default-OFF behavior is unchanged).
        if selection_diag is not None:
            features_meta["selection"] = selection_diag

        features_path = str(model_dir / _ranking_features_filename(forward_days))
        with open(features_path, "w") as f:
            json.dump(features_meta, f, default=_numpy_default)

        # Quality marker defaults to "pending"; the gate rewrites it to
        # approved/rejected after evaluation. Written here so the marker
        # exists even if the gate-write step is interrupted (DB fallback
        # covers absent markers regardless).
        _write_quality_marker(str(model_dir), model_type, forward_days, "pending")

        return model_path

    @staticmethod
    def _save_train_distribution(
        train_df: pd.DataFrame, feature_cols: list[str], model_dir: str,
        forward_days: int = 5,
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

        path = os.path.join(model_dir, _ranking_train_dist_filename(forward_days))
        with open(path, "w") as f:
            json.dump(dist, f, default=_numpy_default)
        logger.info("Saved training distribution snapshot: %d features", len(dist))

    @staticmethod
    def _compute_inference_psi(
        inference_df: pd.DataFrame, feature_cols: list[str], model_dir: str,
        forward_days: int = 5,
    ) -> dict[str, float] | None:
        dist_path = os.path.join(
            model_dir, _ranking_train_dist_filename(forward_days),
        )
        # Backward-compat: fall back to the unkeyed legacy filename for old
        # models and the selection-OFF case (all horizons shared the full set).
        if not os.path.exists(dist_path):
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
        serving_quality: Optional[str] = None,
    ) -> int:
        if model_path is None or not os.path.exists(model_path):
            # Generic fallback path (e.g. existing-model-on-disk reuse): pick
            # the latest approved model. The serving policy in
            # _run_prediction_async resolves fallbacks before calling this.
            model_path = await self._find_latest_model_on_disk(
                market, forward_days, model_type="ranking",
            )

        if model_path is None or not os.path.exists(model_path):
            raise RuntimeError(f"No model file found for market={market}")

        task.message = "Loading model"
        task.progress = 78.0

        loaded = await asyncio.to_thread(joblib.load, model_path)
        models = loaded if isinstance(loaded, list) else [loaded]

        # Load feature names. Read the horizon-keyed file first; fall back to
        # the unkeyed legacy filename for old models and the selection-OFF case
        # (where every horizon shared the identical full feature set).
        model_basedir = os.path.dirname(model_path)
        features_path = os.path.join(
            model_basedir, _ranking_features_filename(forward_days),
        )
        if not os.path.exists(features_path):
            features_path = os.path.join(model_basedir, "features.json")

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
        psi_scores = self._compute_inference_psi(
            latest_df, feature_cols, model_dir, forward_days,
        )
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

        # Cache in Redis. The cache carries the serving model's quality so a
        # low-confidence (rejected-model) run never masquerades as a
        # high-confidence cache hit and vice versa (decision 4/1#5).
        task.message = "Caching predictions"
        task.progress = 96.0
        await self._write_prediction_cache(
            market, results_df, prediction_date, serving_quality=serving_quality,
        )

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
        self,
        market: str,
        results_df: pd.DataFrame,
        prediction_date: date,
        serving_quality: Optional[str] = None,
    ) -> None:
        if results_df.empty:
            return
        key = _prediction_cache_key(market)
        low_confidence = serving_quality == "rejected"
        try:
            records = []
            for _, row in results_df.iterrows():
                records.append({
                    "symbol": row["symbol"],
                    "predicted_score": float(row["predicted_score"]),
                    "percentile_rank": float(row["percentile_rank"]),
                    "predicted_direction": row["predicted_direction"],
                    "prediction_date": prediction_date.isoformat(),
                    "quality": serving_quality,
                    "low_confidence": low_confidence,
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
            # Serve the configured default trade horizon (30d by default --
            # strongest spread / lowest turnover). Fall back to the largest
            # trained horizon when DEFAULT_TRADE_HORIZON is not in the trained
            # set, so this never crashes on a non-standard PREDICTION_HORIZONS.
            trained_horizons = _get_prediction_horizons()
            if settings.DEFAULT_TRADE_HORIZON in trained_horizons:
                default_horizon = settings.DEFAULT_TRADE_HORIZON
            elif trained_horizons:
                default_horizon = max(trained_horizons)
            else:
                default_horizon = settings.DEFAULT_TRADE_HORIZON
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

    def _calendar_path(self, market: str) -> Optional[str]:
        """Return the qlib day.txt calendar path for a market (or None).

        Shared by the freshness check and the trading-day guard so the two
        agree on the same calendar source.
        """
        settings = get_settings()
        market_map = {"cn": "cn_data", "us": "us_data", "hk": "hk_data"}
        qlib_market = market_map.get(market.lower())
        if not qlib_market:
            return None
        return os.path.join(
            settings.QLIB_DATA_DIR, qlib_market, "calendars", "day.txt"
        )

    def _load_trading_days(self, market: str) -> Optional[list[date]]:
        """Load the sorted list of trading days from the qlib calendar.

        Returns None when the calendar is missing / empty / unparseable so
        callers can fail-OPEN. Parse errors on individual lines are skipped
        rather than aborting the whole list (robustness over strictness).
        """
        calendar_path = self._calendar_path(market)
        if not calendar_path or not os.path.exists(calendar_path):
            return None
        try:
            with open(calendar_path) as f:
                lines = f.read().strip().splitlines()
        except Exception as e:
            logger.warning("Failed to read calendar for %s: %s", market, e)
            return None
        days: list[date] = []
        for ln in lines:
            s = ln.strip()
            if not s:
                continue
            try:
                days.append(date.fromisoformat(s))
            except ValueError:
                continue
        if not days:
            return None
        days.sort()
        return days

    def _is_trading_day(self, market: str, day: date) -> bool:
        """Return True if ``day`` is a trading day for ``market``.

        Reads the same qlib ``day.txt`` calendar as the freshness check.
        Fail-OPEN: if the calendar is missing / empty / unparseable, return
        True so a calendar outage never blocks predictions entirely.
        """
        days = self._load_trading_days(market)
        if days is None:
            logger.warning(
                "Trading-day calendar unavailable for %s -- failing open "
                "(treating %s as a trading day)", market, day.isoformat(),
            )
            return True
        # Binary membership check against the sorted calendar.
        idx = bisect.bisect_left(days, day)
        return idx < len(days) and days[idx] == day

    def _resolve_prediction_date(self, market: str, today: date) -> date:
        """Align the prediction date to the actual data day used at inference.

        Returns the latest trading day on-or-before ``today`` from the qlib
        calendar so that every written ``stock_predictions`` row carries a
        backfillable trading-day date (decision 10: prevent NEW non-trading
        rows). Fail-OPEN to ``today`` when the calendar is unavailable -- this
        preserves legacy behavior rather than blocking predictions.

        ``write_predictions`` upserts on ``(market, symbol, prediction_date,
        forward_days)`` so re-aligning the date is idempotent and safe.
        """
        days = self._load_trading_days(market)
        if days is None:
            logger.warning(
                "Trading-day calendar unavailable for %s -- using today=%s "
                "as prediction_date (fail-open)", market, today.isoformat(),
            )
            return today
        # Largest calendar day that is <= today.
        idx = bisect.bisect_right(days, today)
        if idx == 0:
            # Every calendar day is after today (clock skew / stale data) --
            # fail-open to today rather than inventing a future date.
            logger.warning(
                "No trading day on-or-before %s for %s -- using today "
                "(fail-open)", today.isoformat(), market,
            )
            return today
        aligned = days[idx - 1]
        if aligned != today:
            logger.info(
                "Aligned prediction_date for %s: %s -> %s (latest trading day)",
                market, today.isoformat(), aligned.isoformat(),
            )
        return aligned

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
