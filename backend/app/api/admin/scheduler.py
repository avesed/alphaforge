"""Admin scheduler status and manual trigger endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from app.core.auth import require_admin
from app.models.user import User
from app.schemas.base import CamelModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/scheduler", tags=["admin-scheduler"])


class JobInfo(CamelModel):
    id: str
    name: str
    next_run_time: str | None = None
    trigger: str | None = None
    # ISO-8601 UTC timestamp of this job's most recent successful run, read
    # from Redis af:scheduler:last_success:{job_id} (Batch D / decision 9).
    last_success: str | None = None


class SlaMarketStatus(CamelModel):
    market: str
    model_age_days: int | None = None
    max_model_age_days: int | None = None
    model_age_breach: bool = False
    prediction_lag_days: int | None = None
    max_prediction_lag_days: int | None = None
    prediction_lag_breach: bool = False
    latest_model_date: str | None = None
    latest_prediction_date: str | None = None
    error: str | None = None


class SlaStatus(CamelModel):
    checked_at: str | None = None
    breach: bool = False
    markets: list[SlaMarketStatus] = []


class SchedulerStatus(CamelModel):
    is_leader: bool
    running: bool
    job_count: int
    jobs: list[JobInfo]
    # Most recent freshness SLA snapshot (Batch D / decision 9). None when the
    # SLA check has not yet run since the last Redis flush.
    sla: SlaStatus | None = None


async def _read_last_success_map(job_ids: list[str]) -> dict[str, str | None]:
    """Read last-success timestamps for the given job ids from Redis.

    Best-effort: returns all-None on any Redis error so the endpoint never
    fails just because last-success tracking is unavailable.
    """
    from app.core.redis import get_redis
    from app.core.scheduler import last_success_key

    out: dict[str, str | None] = {jid: None for jid in job_ids}
    if not job_ids:
        return out
    try:
        r = await get_redis()
        keys = [last_success_key(jid) for jid in job_ids]
        values = await r.mget(keys)
        for jid, val in zip(job_ids, values):
            if val is None:
                continue
            out[jid] = val.decode() if isinstance(val, bytes) else str(val)
    except Exception as exc:
        logger.warning("Failed to read scheduler last-success map: %s", exc)
    return out


async def _read_sla_status() -> SlaStatus | None:
    """Read the most recent SLA snapshot persisted by the SLA check job."""
    import json

    from app.core.redis import get_redis
    from app.core.scheduler import _SLA_STATUS_KEY

    try:
        r = await get_redis()
        raw = await r.get(_SLA_STATUS_KEY)
        if not raw:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode()
        data = json.loads(raw)
        return SlaStatus(
            checked_at=data.get("checked_at"),
            breach=bool(data.get("breach", False)),
            markets=[SlaMarketStatus(**m) for m in data.get("markets", [])],
        )
    except Exception as exc:
        logger.warning("Failed to read scheduler SLA status: %s", exc)
        return None


@router.get("/jobs", response_model=SchedulerStatus)
async def list_jobs(admin: User = Depends(require_admin)):
    """List all scheduled jobs with their next run times + last success."""
    from app.core.scheduler import get_scheduler, is_leader

    scheduler = get_scheduler()
    leader = is_leader()

    raw_jobs = list(scheduler.get_jobs()) if scheduler is not None else []
    last_success_map = await _read_last_success_map([j.id for j in raw_jobs])

    jobs: list[JobInfo] = []
    for job in raw_jobs:
        jobs.append(JobInfo(
            id=job.id,
            name=job.name,
            next_run_time=str(job.next_run_time) if job.next_run_time else None,
            trigger=str(job.trigger),
            last_success=last_success_map.get(job.id),
        ))

    sla = await _read_sla_status()

    return SchedulerStatus(
        is_leader=leader,
        running=scheduler is not None and scheduler.running,
        job_count=len(jobs),
        jobs=jobs,
        sla=sla,
    )


@router.post("/trigger/{job_id}")
async def trigger_job(
    job_id: str,
    admin: User = Depends(require_admin),
):
    """Manually trigger a scheduled job to run immediately."""
    from app.core.scheduler import get_scheduler, is_leader

    if not is_leader():
        raise HTTPException(status_code=409, detail="This instance is not the scheduler leader")

    scheduler = get_scheduler()
    if scheduler is None:
        raise HTTPException(status_code=503, detail="Scheduler not running")

    job = scheduler.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    from datetime import datetime, timezone
    job.modify(next_run_time=datetime.now(timezone.utc))

    logger.info("Admin %s triggered job '%s'", admin.email, job_id)
    return {"status": "triggered", "job_id": job_id, "job_name": job.name}


@router.post("/relinquish")
async def relinquish_leadership(admin: User = Depends(require_admin)):
    """Force this instance to drop scheduler leadership and trigger re-election.

    Used to recover when the scheduler is wedged or to migrate leadership
    between workers without a process restart.
    """
    from app.core.scheduler import is_leader, request_relinquish

    was_leader = is_leader()
    await request_relinquish()

    logger.info(
        "Admin %s requested scheduler relinquish (was_leader=%s)",
        admin.email, was_leader,
    )
    return {
        "status": "relinquished" if was_leader else "noop",
        "was_leader": was_leader,
    }
