"""RD-Agent subprocess management.

Manages the RD-Agent automated factor research process as an isolated
subprocess. Uses Redis distributed lock for single-instance enforcement.

Architecture:
- subprocess.Popen launches ``rdagent fin_quant`` CLI in an isolated workdir
- Each run gets a fresh workdir with a .env file that RD-Agent reads via
  load_dotenv() -- env overrides in Popen env dict are NOT read by RD-Agent
- OPENAI_API_BASE points to ai-gateway Docker DNS (not localhost)
- Progress tracked via stdout log parsing (Round X/Y pattern)
- Discovered factors collected from result directory files after process exits

AlphaForge adaptation:
- Universe symbols resolved via get_db_pool() (asyncpg) instead of
  settings_cache.get_universes().
- Index constituents fetched from StockPulse via get_stockpulse_client().
- Redis lock uses get_redis() from app.core.redis.
"""

import asyncio
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from app.config import get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Redis lock key pattern and TTL (24 hours)
_LOCK_KEY_PATTERN = "rdagent:lock:{market}"
_LOCK_TTL = 86400

# Maximum process runtime before forced kill (24 hours)
_MAX_RUNTIME_SECONDS = 86400

# Graceful shutdown timeout before SIGKILL
_SIGTERM_WAIT_SECONDS = 30

# Maximum log lines retained in memory per task
_MAX_LOG_LINES = 100

# Stdout progress pattern -- only parse round progress, NOT factor data
# (factors are in result directory files, not stdout)
_RE_ROUND = re.compile(r"Round\s+(\d+)[/\s](\d+)", re.IGNORECASE)


def _get_index_constituents_sync(index_code: str, market: str) -> list[str]:
    """Synchronous index constituent fetch (for asyncio.to_thread).

    Uses StockPulse client to resolve index constituents.
    """
    from app.services.stockpulse_client import get_stockpulse_client

    client = get_stockpulse_client()
    # StockPulse returns all symbols for a market; for index universes
    # the actual filtering is done by the universe's symbol list
    return client.get_symbols(market)


# ---------------------------------------------------------------------------
# Task dataclass
# ---------------------------------------------------------------------------


@dataclass
class RDAgentTask:
    """In-memory representation of an RD-Agent research task."""

    market: str
    status: str = "idle"  # idle, starting, running, completed, failed, stopped
    current_round: int = 0
    max_rounds: int = 30
    discovered_count: int = 0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    log_lines: list[str] = field(default_factory=list)  # last N lines
    workdir: Optional[str] = None  # path to isolated run directory

    def to_dict(self) -> dict:
        """Serialize to API-compatible dict."""
        return {
            "market": self.market,
            "status": self.status,
            "current_round": self.current_round,
            "max_rounds": self.max_rounds,
            "discovered_count": self.discovered_count,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error,
            "log_tail": self.log_lines[-20:],
            "workdir": self.workdir,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _acquire_lock(market: str) -> bool:
    """Try to acquire the distributed lock for a market. Returns True on success."""
    from app.core.redis import get_redis

    r = await get_redis()
    key = _LOCK_KEY_PATTERN.format(market=market)
    acquired = await r.set(key, "1", nx=True, ex=_LOCK_TTL)
    return bool(acquired)


async def _release_lock(market: str) -> None:
    """Release the distributed lock for a market."""
    from app.core.redis import get_redis

    r = await get_redis()
    key = _LOCK_KEY_PATTERN.format(market=market)
    await r.delete(key)


# ---------------------------------------------------------------------------
# RDAgentRunner
# ---------------------------------------------------------------------------


class RDAgentRunner:
    """Manages RD-Agent subprocess lifecycle.

    Supports one concurrent process per market. Uses Redis SET NX for
    distributed locking so multiple data-processor instances cannot
    start duplicate runs.
    """

    def __init__(self) -> None:
        self._processes: dict[str, subprocess.Popen] = {}
        self._tasks: dict[str, RDAgentTask] = {}
        self._monitors: dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()

    async def start(
        self,
        market: str,
        universe_id: str | None = None,
        max_rounds: int = 30,
    ) -> dict:
        """Start an RD-Agent research loop for the given market.

        Args:
            market: Market code (us, hk, cn).
            universe_id: Optional universe UUID to scope symbols.
            max_rounds: Maximum number of research iterations.

        Returns:
            Task status dict on success, or dict with "error" key on failure.
        """
        async with self._lock:
            # Check if already running
            if market in self._processes and self._processes[market].poll() is None:
                return {"error": f"RD-Agent already running for market {market}"}

            # Try distributed lock
            if not await _acquire_lock(market):
                return {"error": f"RD-Agent locked by another instance for market {market}"}

            # Resolve universe symbols
            symbols = await self._resolve_symbols(market, universe_id)
            if not symbols:
                await _release_lock(market)
                return {"error": f"No symbols found for market {market}"}

            # Create isolated workdir with .env for this run
            try:
                workdir = self._prepare_workdir(market, max_rounds)
            except OSError as e:
                await _release_lock(market)
                logger.error("Failed to prepare workdir for %s: %s", market, e)
                return {"error": f"Workdir preparation failed: {e}"}
            logger.info(
                "RD-Agent workdir: %s (market=%s, max_rounds=%d, symbols=%d)",
                workdir, market, max_rounds, len(symbols),
            )

            # Create task
            task = RDAgentTask(
                market=market,
                status="starting",
                max_rounds=max_rounds,
                started_at=datetime.now(),
                workdir=str(workdir),
            )
            self._tasks[market] = task

            # Launch subprocess in thread (Popen can briefly block)
            try:
                process = await asyncio.to_thread(
                    self._launch_subprocess, workdir,
                )
                self._processes[market] = process
                task.status = "running"
                logger.info(
                    "RD-Agent started: market=%s, pid=%d, symbols=%d, max_rounds=%d",
                    market, process.pid, len(symbols), max_rounds,
                )
            except Exception as e:
                task.status = "failed"
                task.error = str(e)
                await _release_lock(market)
                logger.error("Failed to launch RD-Agent for %s: %s", market, e)
                return {"error": f"Launch failed: {e}"}

            # Start background monitoring
            monitor = asyncio.create_task(
                self._monitor_process(market),
                name=f"rdagent-monitor-{market}",
            )
            self._monitors[market] = monitor

            return task.to_dict()

    def _prepare_workdir(self, market: str, max_rounds: int) -> Path:
        """Create an isolated workdir and write the RD-Agent .env config file.

        RD-Agent calls load_dotenv() to read its configuration from a .env
        file in the current working directory. Custom env vars in the subprocess
        env dict are NOT read. This method writes the correct .env so that:
        - OPENAI_API_BASE points to ai-gateway via Docker DNS (not localhost)
        - CHAT_MODEL / EMBEDDING_MODEL are set from our config
        - QLIB_QUANT_MAX_LOOP controls the number of research rounds

        Args:
            market: Market code (used for workdir naming).
            max_rounds: Maximum research rounds.

        Returns:
            Path to the prepared workdir.
        """
        settings = get_settings()
        workdir = Path(settings.PREDICTION_DATA_DIR) / f"rdagent_{market}_{int(time.time())}"
        workdir.mkdir(parents=True, exist_ok=True)

        # Write .env that RD-Agent will read via load_dotenv()
        env_lines = [
            f"CHAT_MODEL={settings.RDAGENT_CHAT_MODEL}",
            f"EMBEDDING_MODEL={settings.RDAGENT_EMBED_MODEL}",
            # Use Docker DNS name -- localhost would be wrong inside the container
            f"OPENAI_API_BASE={settings.AI_GATEWAY_URL}/v1",
            # AI Gateway handles real auth; RD-Agent requires a non-empty key
            "OPENAI_API_KEY=dummy",
            f"QLIB_QUANT_MAX_LOOP={max_rounds}",
        ]
        env_content = "\n".join(env_lines) + "\n"
        (workdir / ".env").write_text(env_content)
        logger.debug(
            "RD-Agent .env written to %s: OPENAI_API_BASE=%s/v1, "
            "CHAT_MODEL=%s, QLIB_QUANT_MAX_LOOP=%d",
            workdir, settings.AI_GATEWAY_URL,
            settings.RDAGENT_CHAT_MODEL, max_rounds,
        )
        return workdir

    def _launch_subprocess(self, workdir: Path) -> subprocess.Popen:
        """Launch the RD-Agent subprocess in its isolated workdir.

        Runs in a thread via asyncio.to_thread since Popen may briefly block.
        The subprocess inherits the system PATH so ``rdagent`` CLI is found.
        CWD is set to workdir so load_dotenv() picks up the .env we wrote.

        Args:
            workdir: Isolated working directory containing the .env file.

        Returns:
            Popen process handle.
        """
        # Inherit full process env for PATH, PYTHONPATH, etc.
        # RD-Agent configuration comes from the .env file in workdir,
        # not from env vars here (load_dotenv() reads from filesystem).
        env = dict(os.environ)

        process = subprocess.Popen(
            ["rdagent", "fin_quant"],  # Correct CLI: fin_quant = Qlib factor discovery
            cwd=str(workdir),           # load_dotenv() reads .env from here
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return process

    @staticmethod
    async def _graceful_kill(
        process: subprocess.Popen, market: str,
    ) -> None:
        """Terminate a subprocess gracefully: SIGTERM -> wait -> SIGKILL."""
        try:
            process.send_signal(signal.SIGTERM)
        except OSError as e:
            logger.warning("SIGTERM failed for %s: %s", market, e)
        try:
            await asyncio.to_thread(process.wait, timeout=_SIGTERM_WAIT_SECONDS)
        except subprocess.TimeoutExpired:
            logger.warning(
                "RD-Agent %s did not exit after %ds, sending SIGKILL",
                market, _SIGTERM_WAIT_SECONDS,
            )
            process.kill()
            await asyncio.to_thread(process.wait, timeout=5)

    async def _monitor_process(self, market: str) -> None:
        """Monitor the RD-Agent subprocess and collect results on exit.

        Runs as a background asyncio task. Reads stdout line by line for
        progress (Round X/Y) only. On process exit, scans the workdir for
        result JSON files containing discovered factors.

        Factor data is NOT parsed from stdout -- RD-Agent writes results
        to the filesystem (workdir/log/, workdir/storage/, etc.).
        """
        task = self._tasks.get(market)
        process = self._processes.get(market)
        if not task or not process:
            await _release_lock(market)
            return

        start_time = datetime.now()
        _MAX_CONSECUTIVE_NONES = 20  # 10 seconds of empty stdout reads

        try:
            consecutive_nones = 0
            while process.poll() is None:
                # Check for timeout
                elapsed = (datetime.now() - start_time).total_seconds()
                if elapsed > _MAX_RUNTIME_SECONDS:
                    logger.warning(
                        "RD-Agent %s exceeded %ds runtime, terminating",
                        market, _MAX_RUNTIME_SECONDS,
                    )
                    await self._graceful_kill(process, market)
                    task.status = "failed"
                    task.error = "Exceeded maximum runtime (24h)"
                    break

                # Read one line (blocking, in thread)
                line = await asyncio.to_thread(self._readline_safe, process)
                if line is None:
                    consecutive_nones += 1
                    if consecutive_nones >= _MAX_CONSECUTIVE_NONES:
                        logger.info(
                            "RD-Agent %s: stdout closed, waiting for process exit",
                            market,
                        )
                        break
                    await asyncio.sleep(0.5)
                    continue
                consecutive_nones = 0

                line = line.rstrip()
                if not line:
                    continue

                # Store in log buffer
                task.log_lines.append(line)
                if len(task.log_lines) > _MAX_LOG_LINES:
                    task.log_lines = task.log_lines[-_MAX_LOG_LINES:]

                # Parse round progress only
                round_match = _RE_ROUND.search(line)
                if round_match:
                    task.current_round = int(round_match.group(1))
                    logger.debug(
                        "RD-Agent %s: round %d/%d",
                        market, task.current_round, task.max_rounds,
                    )

            # Process exited (or stdout closed) -- wait for return code
            try:
                await asyncio.to_thread(process.wait, timeout=60)
            except subprocess.TimeoutExpired:
                logger.warning(
                    "RD-Agent %s still running after stdout closed, terminating",
                    market,
                )
                await self._graceful_kill(process, market)
            exit_code = process.returncode

            if task.status not in ("failed", "stopped"):
                if exit_code == 0:
                    task.status = "completed"
                    logger.info(
                        "RD-Agent %s completed: rounds=%d",
                        market, task.current_round,
                    )
                else:
                    task.status = "failed"
                    task.error = f"Process exited with code {exit_code}"
                    logger.error(
                        "RD-Agent %s failed: exit_code=%d", market, exit_code,
                    )

            # Collect results from filesystem (not stdout)
            if task.workdir and task.status in ("completed", "failed"):
                await self._collect_results(Path(task.workdir), market, task)
                # Clean up workdir after result collection (keep failed for debugging)
                if task.status == "completed":
                    self._cleanup_workdir(Path(task.workdir))

        except asyncio.CancelledError:
            logger.info("RD-Agent monitor cancelled for %s", market)
            task.status = "stopped"
        except Exception as e:
            logger.exception("RD-Agent monitor error for %s: %s", market, e)
            task.status = "failed"
            task.error = str(e)
        finally:
            task.completed_at = datetime.now()
            await _release_lock(market)
            # Clean up process reference
            self._processes.pop(market, None)
            self._monitors.pop(market, None)

    @staticmethod
    def _readline_safe(process: subprocess.Popen) -> Optional[str]:
        """Read a line from process stdout, returning None on EOF/error."""
        try:
            if process.stdout is None:
                return None
            line = process.stdout.readline()
            return line if line else None
        except (ValueError, OSError):
            return None

    async def _collect_results(
        self, workdir: Path, market: str, task: RDAgentTask,
    ) -> None:
        """Scan the workdir for factor result files and register them.

        RD-Agent writes results to the filesystem (not stdout). The exact
        directory structure depends on the rdagent version and scenario.
        We search broadly for JSON files that look like factor results.

        Args:
            workdir: The isolated run directory to scan.
            market: Market code for factor registration.
            task: Task object to update discovered_count.
        """
        # Search for JSON files that may contain factor results
        # RD-Agent typically writes to: log/, storage/, results/, or root
        candidate_patterns = [
            "**/factor*.json",
            "**/result*.json",
            "**/factors.json",
            "**/discovered*.json",
        ]

        if not workdir.exists():
            logger.warning(
                "RD-Agent %s: workdir no longer exists at collection time: %s",
                market, workdir,
            )
            return

        result_files: list[Path] = []
        for pattern in candidate_patterns:
            result_files.extend(workdir.glob(pattern))

        if not result_files:
            logger.info(
                "RD-Agent %s: no result JSON files found in %s. "
                "If this is the first run, check the workdir structure to update "
                "_collect_results() parsing logic.",
                market, workdir,
            )
            # Log all files in workdir for debugging
            all_files = list(workdir.rglob("*"))
            logger.debug(
                "RD-Agent workdir contents (%d files): %s",
                len(all_files),
                [str(f.relative_to(workdir)) for f in all_files[:30]],
            )
            return

        discovered_factors: list[dict] = []
        for result_file in result_files:
            try:
                data = json.loads(result_file.read_text())
                factors = self._parse_factor_data(data)
                if factors:
                    discovered_factors.extend(factors)
                    logger.info(
                        "RD-Agent %s: parsed %d factors from %s",
                        market, len(factors), result_file.name,
                    )
            except json.JSONDecodeError as e:
                logger.warning(
                    "RD-Agent %s: invalid JSON in %s: %s", market, result_file, e,
                )
            except Exception as e:
                logger.warning(
                    "RD-Agent %s: failed to parse %s: %s", market, result_file, e,
                )

        if discovered_factors:
            task.discovered_count = len(discovered_factors)
            await self._register_factors(market, discovered_factors)
        else:
            logger.info(
                "RD-Agent %s: result files found but no parseable factors. "
                "Inspect workdir %s and update _parse_factor_data() to match "
                "the actual JSON structure.",
                market, workdir,
            )

    @staticmethod
    def _parse_factor_data(data: object) -> list[dict]:
        """Extract factor dicts from a parsed JSON result file.

        This method needs to be updated after the first successful RD-Agent run
        to match the actual JSON output format. The structure varies by rdagent
        version and scenario configuration.

        Current implementation handles common patterns:
        - List of factor objects: [{"name": ..., "expression": ..., "ic": ...}, ...]
        - Dict with "factors" key: {"factors": [...]}
        - Single factor object: {"name": ..., "expression": ...}

        Args:
            data: Parsed JSON data (any type).

        Returns:
            List of factor dicts with keys: name, expression, ic, icir.
            Empty list if no recognizable factor structure found.
        """
        factors = []

        def _extract_factor(obj: dict) -> Optional[dict]:
            """Try to extract a factor dict from a JSON object."""
            if not isinstance(obj, dict):
                return None
            # Must have at least a name or expression
            name = obj.get("name") or obj.get("factor_name") or obj.get("id")
            expression = (
                obj.get("expression")
                or obj.get("formula")
                or obj.get("code")
                or obj.get("factor_expression")
            )
            if not (name or expression):
                return None
            return {
                "name": str(name or expression),
                "expression": str(expression or name),
                "ic": float(obj.get("ic", 0.0) or 0.0),
                "icir": float(obj.get("icir", 0.0) or 0.0),
            }

        if isinstance(data, list):
            for item in data:
                f = _extract_factor(item)
                if f:
                    factors.append(f)
        elif isinstance(data, dict):
            # Try "factors" key first
            if "factors" in data and isinstance(data["factors"], list):
                for item in data["factors"]:
                    f = _extract_factor(item)
                    if f:
                        factors.append(f)
            else:
                # Try as single factor
                f = _extract_factor(data)
                if f:
                    factors.append(f)

        return factors

    @staticmethod
    async def _register_factors(market: str, factors: list[dict]) -> None:
        """Register discovered factors via the factor registry."""
        from app.services.factor_registry import factor_registry

        try:
            count = await factor_registry.register_batch(
                factors=factors,
                market=market,
                universe_id=None,
            )
            logger.info(
                "Registered %d/%d discovered factors for market=%s",
                count, len(factors), market,
            )
        except Exception as e:
            logger.error(
                "Failed to register factors for %s: %s", market, e,
            )

    @staticmethod
    async def _resolve_symbols(
        market: str,
        universe_id: str | None,
    ) -> list[str]:
        """Resolve the symbol list for an RD-Agent run.

        Reads prediction_universes from AlphaForge's local DB via
        get_db_pool(). Falls back to StockPulse for symbol lists.

        Args:
            market: Market code.
            universe_id: Optional specific universe UUID.

        Returns:
            List of stock symbols, empty if none found.
        """
        try:
            from app.core.database import get_db_pool

            pool = get_db_pool()

            # Query prediction_universes for this market
            if universe_id:
                row = await pool.fetchrow(
                    "SELECT id, name, universe_type, index_code, symbols "
                    "FROM prediction_universes WHERE id = $1::uuid",
                    universe_id,
                )
                if row and row["symbols"]:
                    symbols = list(row["symbols"])
                    logger.info(
                        "Resolved %d symbols from universe %s (%s)",
                        len(symbols), row["name"], universe_id,
                    )
                    return symbols
            else:
                # Find default universe for this market
                rows = await pool.fetch(
                    "SELECT id, name, universe_type, index_code, symbols, is_default "
                    "FROM prediction_universes WHERE market = $1 "
                    "ORDER BY is_default DESC, created_at ASC",
                    market,
                )
                for row in rows:
                    if row["symbols"]:
                        symbols = list(row["symbols"])
                        logger.info(
                            "Resolved %d symbols from universe %s (default=%s)",
                            len(symbols), row["name"], row["is_default"],
                        )
                        return symbols
        except Exception as e:
            logger.warning("Failed to resolve symbols from DB: %s", e)

        # Fallback: get all symbols from StockPulse
        try:
            symbols = await asyncio.to_thread(
                _get_index_constituents_sync, "", market,
            )
            if symbols:
                logger.info(
                    "Resolved %d symbols from StockPulse for market=%s",
                    len(symbols), market,
                )
                return symbols
        except Exception as e:
            logger.warning(
                "StockPulse symbol resolution failed for %s: %s", market, e,
            )

        logger.warning(
            "No symbols resolved for market=%s (universe_id=%s)",
            market, universe_id,
        )
        return []

    async def stop(self, market: str) -> dict:
        """Stop a running RD-Agent process.

        Sends SIGTERM, waits up to 30 seconds, then SIGKILL if needed.

        Args:
            market: Market code.

        Returns:
            Updated task status dict, or dict with "error" key.
        """
        async with self._lock:
            process = self._processes.get(market)
            task = self._tasks.get(market)

            if not process or process.poll() is not None:
                return {"error": f"No running RD-Agent process for market {market}"}

            logger.info("Stopping RD-Agent for %s (pid=%d)", market, process.pid)

            if task:
                task.status = "stopped"

        # Graceful shutdown: SIGTERM -> wait -> SIGKILL (outside lock)
        await self._graceful_kill(process, market)

        # Cancel monitor task
        monitor = self._monitors.get(market)
        if monitor and not monitor.done():
            monitor.cancel()

        # Ensure lock is released
        await _release_lock(market)

        if task:
            task.completed_at = datetime.now()
            return task.to_dict()

        return {"market": market, "status": "stopped"}

    async def get_status(self, market: str) -> dict:
        """Get the current status of an RD-Agent task.

        Args:
            market: Market code.

        Returns:
            Task status dict, or idle status if no task exists.
        """
        task = self._tasks.get(market)
        if task:
            return task.to_dict()

        return {
            "market": market,
            "status": "idle",
            "current_round": 0,
            "max_rounds": 0,
            "discovered_count": 0,
            "started_at": None,
            "completed_at": None,
            "error": None,
            "log_tail": [],
            "workdir": None,
        }

    @staticmethod
    def _cleanup_workdir(workdir: Path) -> None:
        """Remove a completed run's workdir to prevent accumulation."""
        try:
            shutil.rmtree(workdir, ignore_errors=True)
            logger.debug("Cleaned up RD-Agent workdir: %s", workdir)
        except Exception as e:
            logger.warning("Failed to clean up workdir %s: %s", workdir, e)

    @staticmethod
    def cleanup_old_workdirs(max_age_days: int = 7) -> int:
        """Remove RD-Agent workdirs older than max_age_days.

        Called by the scheduled model cleanup job.
        Returns count of directories removed.
        """
        from app.config import get_settings
        base_dir = Path(get_settings().PREDICTION_DATA_DIR)
        cutoff = time.time() - max_age_days * 86400
        removed = 0
        for d in base_dir.glob("rdagent_*"):
            if d.is_dir() and d.stat().st_mtime < cutoff:
                try:
                    shutil.rmtree(d)
                    removed += 1
                except Exception as e:
                    logger.warning("Failed to remove old workdir %s: %s", d, e)
        if removed:
            logger.info("Cleaned up %d old RD-Agent workdirs", removed)
        return removed

    def shutdown(self) -> None:
        """Kill all running RD-Agent processes.

        Called during application shutdown to ensure no orphaned processes.
        """
        for market, process in list(self._processes.items()):
            if process.poll() is None:
                logger.info("Killing RD-Agent process for %s (pid=%d)", market, process.pid)
                try:
                    process.kill()
                    process.wait(timeout=5)
                except Exception as e:
                    logger.warning("Error killing RD-Agent %s: %s", market, e)

        # Cancel all monitor tasks
        for market, monitor in list(self._monitors.items()):
            if not monitor.done():
                monitor.cancel()

        self._processes.clear()
        self._monitors.clear()
        logger.info("RDAgentRunner shut down")


# Module singleton
rdagent_runner = RDAgentRunner()
