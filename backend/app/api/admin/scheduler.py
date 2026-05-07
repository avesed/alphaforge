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


class SchedulerStatus(CamelModel):
    is_leader: bool
    running: bool
    job_count: int
    jobs: list[JobInfo]


@router.get("/jobs", response_model=SchedulerStatus)
async def list_jobs(admin: User = Depends(require_admin)):
    """List all scheduled jobs with their next run times."""
    from app.core.scheduler import get_scheduler, is_leader

    scheduler = get_scheduler()
    leader = is_leader()

    jobs: list[JobInfo] = []
    if scheduler is not None:
        for job in scheduler.get_jobs():
            jobs.append(JobInfo(
                id=job.id,
                name=job.name,
                next_run_time=str(job.next_run_time) if job.next_run_time else None,
                trigger=str(job.trigger),
            ))

    return SchedulerStatus(
        is_leader=leader,
        running=scheduler is not None and scheduler.running,
        job_count=len(jobs),
        jobs=jobs,
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
