#!/bin/bash
set -e

echo "Running database migrations..."
cd /app/backend && alembic upgrade head

# ---------------------------------------------------------------------------
# Pin ML thread pools (OpenMP / OpenBLAS / MKL / NumExpr) to the cgroup CPU
# quota instead of the host core count. Native libs read nproc, not the cgroup
# limit, so on a CPU-limited container they oversubscribe cores (observed ~9x
# slower training). Mirrors backend/app/utils/cpu.py resolution order.
# Any missing file / odd content safely falls back to nproc -- guarded so that
# `set -e` cannot abort the script (no bare `cat` whose failure would exit).
# ---------------------------------------------------------------------------
ml_threads=""

# cgroup v2: /sys/fs/cgroup/cpu.max -> "<quota> <period>"; quota "max" or
# period <= 0 means no usable limit -> skip.
if [ -r /sys/fs/cgroup/cpu.max ]; then
  read -r cg_quota cg_period < /sys/fs/cgroup/cpu.max || true
  # Validate BOTH fields are positive integers (mirrors cpu.py: a non-numeric
  # quota falls through to v1/nproc rather than bash coercing it to 0 -> 1).
  if [ -n "$cg_quota" ] && [ "$cg_quota" != "max" ] \
     && [ "$cg_quota" -gt 0 ] 2>/dev/null \
     && [ -n "$cg_period" ] && [ "$cg_period" -gt 0 ] 2>/dev/null; then
    ml_threads=$(( cg_quota / cg_period ))
  fi
fi

# cgroup v1: cpu.cfs_quota_us / cpu.cfs_period_us; quota <= 0 (e.g. -1) -> skip.
if [ -z "$ml_threads" ] \
   && [ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ] \
   && [ -r /sys/fs/cgroup/cpu/cpu.cfs_period_us ]; then
  read -r cg_quota < /sys/fs/cgroup/cpu/cpu.cfs_quota_us || true
  read -r cg_period < /sys/fs/cgroup/cpu/cpu.cfs_period_us || true
  if [ -n "$cg_quota" ] && [ "$cg_quota" -gt 0 ] 2>/dev/null \
     && [ -n "$cg_period" ] && [ "$cg_period" -gt 0 ] 2>/dev/null; then
    ml_threads=$(( cg_quota / cg_period ))
  fi
fi

# Fallback: host core count (legacy "all cores" behavior outside a quota).
if [ -z "$ml_threads" ]; then
  ml_threads=$(nproc 2>/dev/null || echo 1)
fi

# Never go below 1.
if [ "$ml_threads" -lt 1 ] 2>/dev/null; then
  ml_threads=1
fi

export OMP_NUM_THREADS="$ml_threads"
export OPENBLAS_NUM_THREADS="$ml_threads"
export MKL_NUM_THREADS="$ml_threads"
export NUMEXPR_NUM_THREADS="$ml_threads"
echo "ML thread cap: ${ml_threads} (pinned OMP/BLAS to cgroup quota)"

echo "Starting AlphaForge..."
exec /usr/bin/supervisord -c /etc/supervisor/conf.d/alphaforge.conf
