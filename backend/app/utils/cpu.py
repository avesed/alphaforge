"""CPU/thread budgeting for ML parallelism, pinned to the cgroup CPU quota.

Why this exists
---------------
Native ML libraries (LightGBM/OpenMP, NumPy/BLAS, qlib's default ``kernels``)
size their thread/worker pools off the *host* CPU count -- ``os.cpu_count()`` /
``nproc`` -- because they do not read the cgroup CPU quota that constrains the
container. In a container limited to e.g. 16 cores on a 32-core host, every
library spins up 32 threads, oversubscribing the 16 available cores. The kernel
then time-slices ~2x more threads than cores, and with several libraries each
doing this independently the contention compounds (observed ~9x slower training).

The fix is a single source of truth for "how many threads ML code may use",
derived from the cgroup CPU quota rather than the host core count, and applied
everywhere (OMP/BLAS env vars at process start, LightGBM ``num_threads``,
qlib ``kernels``).

Resolution order (see ``get_ml_threads``):
  1. ``OMP_NUM_THREADS`` env var (the single source of truth the entrypoint sets).
  2. cgroup v2: ``/sys/fs/cgroup/cpu.max`` -> floor(quota / period).
  3. cgroup v1: ``cpu.cfs_quota_us`` / ``cpu.cfs_period_us`` -> floor(quota / period).
  4. fallback: ``os.cpu_count()`` (preserves legacy "use all host cores" behavior
     when running locally / outside a container with no quota).

The result is always clamped to ``[1, os.cpu_count()]`` and the whole thing is
wrapped in try/except so it can never raise, never return 0 or a negative value.
"""
import functools
import os

_CGROUP_V2_CPU_MAX = "/sys/fs/cgroup/cpu.max"
_CGROUP_V1_QUOTA = "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"
_CGROUP_V1_PERIOD = "/sys/fs/cgroup/cpu/cpu.cfs_period_us"


def _read_first_line(path: str) -> str | None:
    """Return the stripped first line of ``path``, or None if unreadable."""
    try:
        with open(path, "r") as f:
            return f.readline().strip()
    except (OSError, ValueError):
        return None


def _from_cgroup_v2() -> int | None:
    """Parse cgroup v2 ``cpu.max`` -> integer core budget, or None.

    Content is two whitespace-separated fields: "<quota> <period>", where quota
    is the literal string "max" (no limit) or a number of microseconds, and
    period is a number of microseconds. floor(quota / period) is the core budget.
    """
    raw = _read_first_line(_CGROUP_V2_CPU_MAX)
    if not raw:
        return None
    parts = raw.split()
    if len(parts) < 2:
        return None
    quota_s, period_s = parts[0], parts[1]
    if quota_s == "max":
        return None
    try:
        quota = int(quota_s)
        period = int(period_s)
    except ValueError:
        # Malformed cgroup file -> let the caller try the next source.
        return None
    if quota > 0 and period > 0:
        return quota // period
    return None


def _from_cgroup_v1() -> int | None:
    """Parse cgroup v1 cfs quota/period -> integer core budget, or None.

    A quota of -1 means "no limit" -> skip. Otherwise floor(quota / period).
    """
    quota_s = _read_first_line(_CGROUP_V1_QUOTA)
    period_s = _read_first_line(_CGROUP_V1_PERIOD)
    if quota_s is None or period_s is None:
        return None
    try:
        quota = int(quota_s)
        period = int(period_s)
    except ValueError:
        # Malformed cgroup file -> let the caller try the next source.
        return None
    if quota > 0 and period > 0:
        return quota // period
    return None


@functools.lru_cache(maxsize=1)
def get_ml_threads() -> int:
    """Return the ML thread/worker budget for this process (cached, immutable).

    Pins parallelism to the cgroup CPU quota instead of the host core count so
    that ML libraries do not oversubscribe a CPU-limited container. See the
    module docstring for the full rationale and resolution order.

    Guarantees:
      * never raises (any error falls back to ``os.cpu_count() or 1``);
      * always returns an integer in ``[1, os.cpu_count() or 1]``;
      * never returns 0 or a negative value.
    """
    host_cpus = os.cpu_count() or 1
    try:
        # 1. Single source of truth set by the entrypoint.
        env_val = os.environ.get("OMP_NUM_THREADS")
        if env_val:
            try:
                n = int(env_val)
                if n > 0:
                    return max(1, min(n, host_cpus))
            except ValueError:
                pass

        # 2. cgroup v2, then 3. cgroup v1.
        n = _from_cgroup_v2()
        if n is None:
            n = _from_cgroup_v1()

        # 4. fallback to host core count (legacy "all cores" behavior).
        if n is None:
            n = host_cpus

        return max(1, min(n, host_cpus))
    except Exception:
        # Absolutely never let thread budgeting break the caller.
        return max(1, host_cpus)
