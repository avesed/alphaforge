"""APScheduler-based background scheduler for AlphaForge.

Uses Redis-backed leader election so that only ONE uvicorn worker runs
scheduled jobs when multiple workers are deployed.

A persistent supervisor task continuously attempts to acquire leadership
and rebuilds the scheduler whenever the previous leader dies (Redis
blip, restart, etc.).  This means a single transient heartbeat miss no
longer kills scheduling permanently -- the next supervisor tick picks
the role back up.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import uuid
from datetime import datetime, timedelta
from typing import Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from app.core.redis import get_redis

logger = logging.getLogger(__name__)

# Leader election settings
_LEADER_KEY = "af:scheduler:leader"
_LEADER_TTL = 60       # seconds -- key expires if leader crashes
_HEARTBEAT_INTERVAL = 20  # seconds -- renew well before TTL expires
_SUPERVISOR_INTERVAL = 30  # seconds -- re-elect cadence when not leader

# Unique ID for this worker instance
_INSTANCE_ID = str(uuid.uuid4())

# Module-level state
_scheduler: Optional[AsyncIOScheduler] = None
_is_leader = False
_supervisor_task: Optional[asyncio.Task] = None
_supervisor_stop = asyncio.Event()


async def start_scheduler() -> None:
    """Start the leader-election supervisor.

    The supervisor runs forever, repeatedly trying to acquire leadership.
    When it succeeds it builds a fresh AsyncIOScheduler with the cron
    jobs.  If leadership is lost (heartbeat fails or someone else takes
    the key), the scheduler is torn down and the supervisor goes back to
    polling -- no process restart required.
    """
    global _supervisor_task, _supervisor_stop

    _supervisor_stop = asyncio.Event()
    _supervisor_task = asyncio.create_task(
        _supervisor_loop(), name="scheduler-supervisor",
    )
    logger.info(
        "Scheduler supervisor started (instance=%s)", _INSTANCE_ID[:8],
    )


async def stop_scheduler() -> None:
    """Stop the supervisor, scheduler, and release leadership."""
    global _supervisor_task

    _supervisor_stop.set()
    if _supervisor_task is not None:
        try:
            await asyncio.wait_for(_supervisor_task, timeout=5)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            _supervisor_task.cancel()
        _supervisor_task = None

    await _teardown_scheduler()
    if _is_leader:
        await _release_leadership()


def is_leader() -> bool:
    """Return whether this instance currently holds leadership."""
    return _is_leader


def get_scheduler() -> Optional[AsyncIOScheduler]:
    """Return the scheduler instance (None if not leader or not started)."""
    return _scheduler


# ---------------------------------------------------------------------------
# Supervisor loop
# ---------------------------------------------------------------------------


async def _supervisor_loop() -> None:
    """Continuously elect a leader and (re)build the scheduler.

    Sleeps for _SUPERVISOR_INTERVAL between attempts when not leader.
    When leader, sleeps until either the scheduler dies or stop is
    requested, then loops back to re-elect.
    """
    global _is_leader

    while not _supervisor_stop.is_set():
        try:
            if not _is_leader:
                acquired = await _try_acquire_leadership()
                if acquired:
                    _is_leader = True
                    logger.info(
                        "Scheduler: acquired leadership (instance=%s)",
                        _INSTANCE_ID[:8],
                    )
                    try:
                        _build_scheduler()
                    except Exception:
                        logger.exception(
                            "Scheduler: failed to build, releasing leadership",
                        )
                        await _release_leadership()
                        _is_leader = False
                else:
                    logger.debug(
                        "Scheduler: another worker is leader (instance=%s)",
                        _INSTANCE_ID[:8],
                    )

            # Wait for either stop signal or interval expiry
            try:
                await asyncio.wait_for(
                    _supervisor_stop.wait(), timeout=_SUPERVISOR_INTERVAL,
                )
            except asyncio.TimeoutError:
                pass

            # If we were leader but the scheduler died, treat as lost
            if _is_leader and (_scheduler is None or not _scheduler.running):
                logger.warning(
                    "Scheduler: instance died while leader, will re-elect "
                    "(instance=%s)", _INSTANCE_ID[:8],
                )
                _is_leader = False
                await _release_leadership()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Scheduler supervisor loop error")
            await asyncio.sleep(_SUPERVISOR_INTERVAL)


def _build_scheduler() -> None:
    """Construct a fresh AsyncIOScheduler with all cron jobs and start it."""
    global _scheduler

    _scheduler = AsyncIOScheduler(timezone="UTC")

    # Leadership heartbeat
    _scheduler.add_job(
        _heartbeat,
        IntervalTrigger(seconds=_HEARTBEAT_INTERVAL),
        id="leader_heartbeat",
        name="Leader heartbeat",
        replace_existing=True,
    )

    _job_kwargs = dict(
        misfire_grace_time=600,  # tolerate up to 10min late firing
        coalesce=True,
        max_instances=1,
        replace_existing=True,
    )

    # --- Qlib data sync (via StockPulse API) ---
    _scheduler.add_job(
        _run_qlib_sync, CronTrigger(hour=8, minute=30),
        args=["cn"], id="qlib_sync_cn", name="Qlib sync CN",
        **_job_kwargs,
    )
    _scheduler.add_job(
        _run_qlib_sync, CronTrigger(hour=9, minute=30),
        args=["hk"], id="qlib_sync_hk", name="Qlib sync HK",
        **_job_kwargs,
    )
    _scheduler.add_job(
        _run_qlib_sync, CronTrigger(hour=22, minute=30),
        args=["us"], id="qlib_sync_us", name="Qlib sync US",
        **_job_kwargs,
    )
    _scheduler.add_job(
        _run_qlib_sync, CronTrigger(hour=23, minute=0),
        args=["metal"], id="qlib_sync_metal", name="Qlib sync Metal",
        **_job_kwargs,
    )

    # --- ML prediction (LightGBM training + inference) ---
    _scheduler.add_job(
        _run_prediction, CronTrigger(hour=9, minute=30),
        args=["cn"], id="predict_cn", name="Predict CN",
        **_job_kwargs,
    )
    _scheduler.add_job(
        _run_prediction, CronTrigger(hour=10, minute=30),
        args=["hk"], id="predict_hk", name="Predict HK",
        **_job_kwargs,
    )
    _scheduler.add_job(
        _run_prediction, CronTrigger(hour=23, minute=30),
        args=["us"], id="predict_us", name="Predict US",
        **_job_kwargs,
    )

    # --- Daily maintenance ---
    _scheduler.add_job(
        _run_backfill_returns, CronTrigger(hour=0, minute=30),
        id="backfill_returns", name="Backfill returns",
        **_job_kwargs,
    )
    _scheduler.add_job(
        _run_cleanup_models, CronTrigger(hour=3, minute=0),
        id="cleanup_models", name="Cleanup old models",
        **_job_kwargs,
    )
    _scheduler.add_job(
        _run_check_auto_retrain, CronTrigger(hour=1, minute=0),
        id="check_auto_retrain", name="Check auto-retrain",
        **_job_kwargs,
    )

    _scheduler.start()
    logger.info("Scheduler started with %d jobs", len(_scheduler.get_jobs()))


async def _teardown_scheduler() -> None:
    """Shut down the AsyncIOScheduler instance if running."""
    global _scheduler

    if _scheduler is not None:
        try:
            _scheduler.shutdown(wait=False)
        except Exception:
            pass
        _scheduler = None
        logger.info("Scheduler shut down")


async def request_relinquish() -> None:
    """Force this instance to drop leadership and trigger a re-election.

    Used by the admin /restart endpoint to recover when a worker is
    stuck or wedged.  Safe to call from any worker -- the supervisor
    loop will re-acquire on the next tick.
    """
    global _is_leader

    if _is_leader:
        logger.warning(
            "Scheduler: relinquish requested, tearing down (instance=%s)",
            _INSTANCE_ID[:8],
        )
        await _teardown_scheduler()
        await _release_leadership()
        _is_leader = False


# ---------------------------------------------------------------------------
# Heartbeat
# ---------------------------------------------------------------------------

_RENEW_LEADER_LUA = """
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("expire", KEYS[1], ARGV[2])
end
return 0
"""


async def _heartbeat() -> None:
    """Atomically check ownership and renew the leader key TTL.

    On failure: tear down the scheduler and reset _is_leader.  The
    supervisor will pick things back up on its next tick -- this
    instance no longer commits suicide on a single transient blip.
    """
    global _is_leader

    try:
        r = await get_redis()
        renewed = await r.eval(
            _RENEW_LEADER_LUA, 1, _LEADER_KEY, _INSTANCE_ID, _LEADER_TTL,
        )
        if renewed:
            return
    except Exception as exc:
        logger.warning(
            "Scheduler: heartbeat error (instance=%s): %s -- staying leader",
            _INSTANCE_ID[:8], exc,
        )
        # Fail-open: keep running and try again next tick
        return

    # Lua returned 0 -- key is gone or owned by another instance
    logger.warning(
        "Scheduler: leadership lost during heartbeat (instance=%s) -- "
        "tearing down, supervisor will retry election",
        _INSTANCE_ID[:8],
    )
    _is_leader = False
    await _teardown_scheduler()


# ---------------------------------------------------------------------------
# Leadership helpers
# ---------------------------------------------------------------------------


async def _try_acquire_leadership() -> bool:
    """Try to acquire the leader key via SET NX."""
    try:
        r = await get_redis()
        acquired = await r.set(
            _LEADER_KEY, _INSTANCE_ID, nx=True, ex=_LEADER_TTL,
        )
        return bool(acquired)
    except Exception as exc:
        logger.warning("Failed to acquire leadership: %s", exc)
        return False


async def _release_leadership() -> None:
    """Release the leader key (only if we own it)."""
    try:
        r = await get_redis()
        lua = (
            'if redis.call("get", KEYS[1]) == ARGV[1] then '
            'return redis.call("del", KEYS[1]) end return 0'
        )
        await r.eval(lua, 1, _LEADER_KEY, _INSTANCE_ID)
    except Exception as exc:
        logger.warning("Failed to release leadership: %s", exc)


# ---------------------------------------------------------------------------
# Job handlers
# ---------------------------------------------------------------------------


async def _run_qlib_sync(market: str) -> None:
    """Sync market data to Qlib .bin format via ProcessPoolExecutor.

    DataSyncService.sync_market() is synchronous (designed for
    ProcessPoolExecutor), so we dispatch via run_qlib_background().
    """
    if not _is_leader:
        return
    logger.info("Scheduler: starting Qlib sync for %s", market)
    try:
        from app.executor import run_qlib_background
        from app.services.data_sync import DataSyncService

        from app.services.stockpulse_client import _get_stockpulse_config
        sp_url, sp_key = await _get_stockpulse_config()

        result = await run_qlib_background(
            DataSyncService.sync_market, market, update_only=True,
            sp_url=sp_url, sp_key=sp_key,
        )
        logger.info(
            "Scheduler: Qlib sync for %s complete -- %d symbols in %.1fs",
            market,
            result.get("symbol_count", 0),
            result.get("duration_s", 0),
        )
    except Exception:
        logger.exception("Scheduler: Qlib sync for %s failed", market)


async def _run_prediction(market: str) -> None:
    """Run LightGBM training + inference for a market.

    Supports multi-horizon when PREDICTION_HORIZONS has >1 value.
    Checks for performance decay and forces retrain if needed.
    Waits for the background prediction task to complete so that
    the scheduler knows the full pipeline has finished.
    """
    if not _is_leader:
        return

    from app.config import get_settings
    from app.services.prediction_service import (
        prediction_service,
        _get_prediction_horizons,
    )

    settings = get_settings()
    horizons = _get_prediction_horizons()

    logger.info(
        "Scheduler: starting prediction for %s, horizons=%s", market, horizons,
    )
    try:
        # Check if retraining is needed due to IC decay
        decay_needed = await prediction_service.check_retrain_needed(market)
        if decay_needed:
            logger.info(
                "Scheduler: IC decay detected for %s, forcing retrain", market,
            )

        if len(horizons) > 1:
            task_id = await prediction_service.run_multi_horizon(
                market, force_retrain=decay_needed,
            )
        else:
            task_id = await prediction_service.run_prediction(
                market, force_retrain=decay_needed, forward_days=horizons[0],
            )

        # Wait for the background asyncio.Task to complete so we know the
        # full pipeline (training + inference) has finished before the
        # scheduler considers the job done.
        task_obj = prediction_service._tasks.get(task_id)
        if task_obj and task_obj._asyncio_task:
            await task_obj._asyncio_task

        # Log result metrics
        if task_obj and task_obj.results:
            ic = task_obj.results.get("ic", 0)
            logger.info(
                "Scheduler: prediction %s complete -- IC=%.4f, status=%s",
                market, ic, task_obj.status,
            )
        else:
            logger.info(
                "Scheduler: prediction %s task %s finished (status=%s)",
                market, task_id, task_obj.status if task_obj else "unknown",
            )
    except Exception:
        logger.exception("Scheduler: prediction for %s failed", market)


async def _run_backfill_returns() -> None:
    """Backfill actual returns for past predictions."""
    if not _is_leader:
        return
    logger.info("Scheduler: starting return backfill")
    try:
        from app.services.prediction_service import prediction_service

        result = await prediction_service.backfill_returns()
        logger.info(
            "Scheduler: backfill returns complete -- updated=%s, failed=%s",
            result.get("updated", 0),
            result.get("failed", 0),
        )
    except Exception:
        logger.exception("Scheduler: backfill returns failed")


async def _run_cleanup_models() -> None:
    """Clean up old model files from disk using PredictionService."""
    if not _is_leader:
        return
    logger.info("Scheduler: starting model cleanup")
    try:
        from app.services.prediction_service import prediction_service

        result = await prediction_service.cleanup_old_models()
        logger.info(
            "Scheduler: model cleanup -- deleted=%d, kept=%d, errors=%d",
            result.get("deleted", 0),
            result.get("kept", 0),
            result.get("errors", 0),
        )
    except Exception:
        logger.exception("Scheduler: model cleanup failed")


async def _run_check_auto_retrain() -> None:
    """Check each market for stale models and trigger retraining if needed.

    A model is considered stale when its age exceeds
    PREDICTION_MAX_STALE_DAYS from settings.  Markets with no models
    at all get an initial training run.
    """
    if not _is_leader:
        return
    logger.info("Scheduler: checking auto-retrain")
    try:
        from app.config import get_settings
        from app.services.prediction_store import prediction_store

        settings = get_settings()

        for market in ("us", "cn", "hk"):
            try:
                models = await prediction_store.get_models(market)
            except Exception as exc:
                logger.warning(
                    "Scheduler: failed to fetch models for %s: %s", market, exc,
                )
                continue

            if not models:
                logger.info(
                    "Scheduler: no models for %s, triggering initial training",
                    market,
                )
                await _run_prediction(market)
                continue

            # Models are returned sorted by date desc from StockPulse API
            latest = models[0]
            model_date_str = latest.get("model_date", "")
            if not model_date_str:
                continue

            try:
                last_train = datetime.strptime(model_date_str, "%Y-%m-%d")
                stale_days = (datetime.now() - last_train).days
                if stale_days > settings.PREDICTION_MAX_STALE_DAYS:
                    logger.info(
                        "Scheduler: model for %s is %d days old "
                        "(threshold=%d), retraining",
                        market, stale_days, settings.PREDICTION_MAX_STALE_DAYS,
                    )
                    await _run_prediction(market)
                else:
                    logger.debug(
                        "Scheduler: model for %s is %d days old, still fresh",
                        market, stale_days,
                    )
            except ValueError:
                logger.warning(
                    "Scheduler: invalid model_date format for %s: %s",
                    market, model_date_str,
                )
    except Exception:
        logger.exception("Scheduler: auto-retrain check failed")
