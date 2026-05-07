"""Data synchronization service for Qlib .bin format.

Fetches market data from StockPulse API (PostgreSQL daily bars)
and converts to Qlib binary format. All data flows through StockPulse --
there is no direct download from external providers.

Designed to run in ProcessPoolExecutor via run_qlib_background().
"""
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd

from app.config import get_settings
from app.context import QlibContext
from app.utils.bin_writer import (
    batch_update_instruments,
    build_calendar_index,
    dataframe_to_bin,
    read_calendar,
    update_calendar,
    update_instruments,
)
from app.utils.symbol_mapping import stockpulse_to_qlib

logger = logging.getLogger(__name__)


def _write_sync_progress(data_dir: str, market: str, info: dict) -> None:
    """Write sync progress to a shared JSON file (IPC with main process)."""
    progress_path = os.path.join(data_dir, "sync_progress.json")
    try:
        progress = {}
        if os.path.exists(progress_path):
            with open(progress_path) as f:
                progress = json.load(f)
        progress[market] = info
        # Atomic write via temp file
        tmp_path = progress_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(progress, f)
        os.replace(tmp_path, progress_path)
    except Exception as e:
        logger.warning("Failed to write sync progress: %s", e)


def _clear_sync_progress(data_dir: str, market: str) -> None:
    """Clear progress for a market after completion or failure."""
    progress_path = os.path.join(data_dir, "sync_progress.json")
    try:
        if not os.path.exists(progress_path):
            return
        with open(progress_path) as f:
            progress = json.load(f)
        progress.pop(market, None)
        tmp_path = progress_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(progress, f)
        os.replace(tmp_path, progress_path)
    except Exception as e:
        logger.warning("Failed to clear sync progress: %s", e)


def get_sync_progress(data_dir: str) -> dict:
    """Read current sync progress for all markets."""
    progress_path = os.path.join(data_dir, "sync_progress.json")
    if not os.path.exists(progress_path):
        return {}
    try:
        with open(progress_path) as f:
            return json.load(f)
    except Exception:
        return {}


class DataSyncService:
    """Market data synchronization to Qlib .bin format.

    All methods are synchronous -- designed to run in ProcessPoolExecutor
    via run_qlib_background().
    """

    @staticmethod
    def sync_market(
        market: str,
        data_dir: Optional[str] = None,
        symbols: Optional[List[str]] = None,
        update_only: bool = True,
        sp_url: str | None = None,
        sp_key: str | None = None,
    ) -> Dict[str, Any]:
        """Synchronize market data to Qlib .bin format.

        All data is fetched from StockPulse API.

        Args:
            market: Market code (us, hk, cn, metal).
            data_dir: Override base Qlib data directory.
            symbols: Specific symbols to sync. None = full default universe.
            update_only: If True, only fetch dates after the last calendar entry.

        Returns:
            Dict with keys: market, symbol_count, new_symbols, errors, duration_s.

        Raises:
            RuntimeError: If StockPulse is unreachable or returns no data.
            ValueError: If market code is invalid.
        """
        settings = get_settings()
        data_dir = data_dir or settings.QLIB_DATA_DIR
        os.makedirs(data_dir, exist_ok=True)

        valid_markets = {"us", "hk", "cn", "sh", "sz", "metal"}
        if market not in valid_markets:
            raise ValueError(
                f"Unknown market: {market}. Valid: {sorted(valid_markets)}"
            )

        # Normalize cn/sh/sz to a common market key for StockPulse API
        api_market = market if market not in ("sh", "sz") else "cn"

        logger.info(
            "Starting data sync for market=%s, update_only=%s", market, update_only
        )
        start_time = time.monotonic()

        try:
            result = DataSyncService._sync_from_stockpulse(
                api_market, data_dir, symbols, update_only,
                sp_url=sp_url, sp_key=sp_key,
            )
        except Exception:
            _clear_sync_progress(data_dir, api_market)
            raise
        # After success, clear progress
        _clear_sync_progress(data_dir, api_market)

        elapsed = time.monotonic() - start_time
        result["duration_s"] = round(elapsed, 2)
        result["market"] = market
        DataSyncService._save_metadata(data_dir, market, result)

        logger.info(
            "Data sync complete for market=%s: %d symbols, %d errors in %.1fs",
            market,
            result["symbol_count"],
            len(result.get("errors", [])),
            elapsed,
        )
        return result

    @staticmethod
    def get_sync_status(data_dir: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """Read sync status for all markets.

        Returns:
            Dict keyed by market code, each containing last_sync, symbol_count,
            date_range, status, and data_exists fields.
        """
        settings = get_settings()
        data_dir = data_dir or settings.QLIB_DATA_DIR

        # Read metadata file
        meta_path = os.path.join(data_dir, "sync_metadata.json")
        metadata: Dict[str, Any] = {}
        if os.path.exists(meta_path):
            try:
                with open(meta_path) as f:
                    metadata = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to read sync_metadata.json: %s", e)

        markets: Dict[str, Dict[str, Any]] = {}
        for market, subdir in QlibContext.STATUS_MARKETS.items():
            market_dir = os.path.join(data_dir, subdir)
            data_exists = os.path.isdir(market_dir) and bool(os.listdir(market_dir))

            # Read calendar date range
            date_range = DataSyncService._get_date_range(market_dir)

            # Read instrument count
            instrument_count = DataSyncService._count_instruments(market_dir)

            market_meta = metadata.get(market, {})
            markets[market] = {
                "last_sync": market_meta.get("last_sync"),
                "symbol_count": market_meta.get("symbol_count", instrument_count),
                "date_range": date_range,
                "data_exists": data_exists,
                "status": "idle",
            }

        return markets

    # ------------------------------------------------------------------ #
    # StockPulse data source sync
    # ------------------------------------------------------------------ #

    @staticmethod
    def _sync_from_stockpulse(
        market: str,
        data_dir: str,
        symbols: Optional[List[str]],
        update_only: bool,
        sp_url: str | None = None,
        sp_key: str | None = None,
    ) -> Dict[str, Any]:
        """Sync market data from StockPulse API.

        Two-phase approach to prevent .bin overwrites during incremental sync:
          Phase 1: Collect all batch data into memory + accumulate new dates
          Phase 2: Update calendar FIRST, then write .bin files with merge

        IMPORTANT: This method is NOT safe for concurrent execution on the same
        market. Calendar and .bin writes are not atomic across calls. Safety is
        ensured by ProcessPoolExecutor(max_workers=1) in executor.py.

        Raises:
            RuntimeError: If StockPulse returns no symbols or all batches fail.
        """
        from app.services.stockpulse_client import StockPulseClient

        # Determine market subdirectory
        market_subdirs = {
            "us": "us_data",
            "hk": "hk_data",
            "cn": "cn_data",
            "metal": "metal_data",
        }
        market_dir = os.path.join(
            data_dir, market_subdirs.get(market, f"{market}_data")
        )
        os.makedirs(market_dir, exist_ok=True)

        client = StockPulseClient(base_url=sp_url, api_key=sp_key)

        # 1. Get symbols from StockPulse if not provided
        if symbols is None:
            symbols = client.get_symbols(market)
            if not symbols:
                raise RuntimeError(
                    f"StockPulse returned empty symbol list for market={market}"
                )

        # 2. Determine start_date for incremental sync
        start_date: Optional[str] = None
        if update_only:
            start_date = DataSyncService._resolve_start_date(
                market_dir, update_only, default="2000-01-01"
            )

        logger.info(
            "Syncing %d %s symbols from StockPulse (start_date=%s)",
            len(symbols), market, start_date,
        )

        # -- Phase 1: Collect all data into memory --
        phase1_start = time.monotonic()
        sync_started_at = datetime.now().isoformat()
        collected: Dict[str, pd.DataFrame] = {}
        all_dates: Set[str] = set()
        errors: List[str] = []
        batch_size = 30
        total_batches = (len(symbols) + batch_size - 1) // batch_size

        # Write initial progress immediately to prevent duplicate triggers
        _write_sync_progress(data_dir, market, {
            "status": "syncing",
            "phase": "collecting",
            "current": 0,
            "total": total_batches,
            "percent": 0,
            "started_at": sync_started_at,
        })

        for i in range(0, len(symbols), batch_size):
            batch = symbols[i : i + batch_size]
            batch_num = i // batch_size + 1
            logger.info(
                "  StockPulse batch %d/%d (%d symbols)",
                batch_num, total_batches, len(batch),
            )

            try:
                data = client.get_daily_bars(
                    symbols=batch,
                    market=market,
                    start_date=start_date,
                )
            except Exception as e:
                msg = f"batch_{batch_num}: {e}"
                errors.append(msg)
                logger.warning("  Batch %d/%d failed: %s", batch_num, total_batches, e)
                continue

            if not data:
                logger.debug("  Batch %d/%d: no data for date range (market=%s)", batch_num, total_batches, market)
                continue

            empty_syms = []
            for sym in batch:
                sym_data = data.get(sym)
                if not sym_data or not sym_data.get("dates"):
                    empty_syms.append(sym)
                    continue

                try:
                    df = pd.DataFrame(
                        {
                            "open": sym_data["open"],
                            "high": sym_data["high"],
                            "low": sym_data["low"],
                            "close": sym_data["close"],
                            "volume": sym_data["volume"],
                        },
                        index=pd.to_datetime(sym_data["dates"]),
                    )

                    if df.empty:
                        logger.debug("  %s: DataFrame empty after construction, skipping", sym)
                        continue

                    qlib_sym = stockpulse_to_qlib(sym, market)
                    dates = [d.strftime("%Y-%m-%d") for d in df.index]
                    all_dates.update(dates)
                    collected[qlib_sym] = df
                except Exception as e:
                    logger.warning("  Failed to process %s: %s", sym, e)
                    errors.append(f"{sym}: {e}")

            if empty_syms:
                logger.debug(
                    "  Batch %d/%d: %d symbols had no/empty data: %s",
                    batch_num, total_batches, len(empty_syms), empty_syms[:10],
                )

            _write_sync_progress(data_dir, market, {
                "status": "syncing",
                "phase": "collecting",
                "current": batch_num,
                "total": total_batches,
                "percent": round(batch_num / total_batches * 50),  # Phase 1 = 0-50%
                "started_at": sync_started_at,
            })

        if not collected:
            if errors:
                raise RuntimeError(
                    f"No data collected for market={market} "
                    f"(start_date={start_date}): "
                    f"{len(errors)} errors, 0 symbols succeeded. "
                    f"Errors: {errors[:5]}"
                )
            # All batches returned empty -- data is already up to date
            logger.info(
                "No new data for market=%s since %s -- already up to date (%.1fs)",
                market, start_date, time.monotonic() - phase1_start,
            )
            _clear_sync_progress(data_dir, market)
            return {
                "symbol_count": 0,
                "new_symbols": 0,
                "errors": [],
                "up_to_date": True,
            }

        phase1_elapsed = time.monotonic() - phase1_start
        logger.info(
            "Phase 1 complete: market=%s, collected %d symbols, "
            "%d new dates (%.1fs)",
            market, len(collected), len(all_dates), phase1_elapsed,
        )

        # -- Phase 2: Update calendar, then write .bin files --
        phase2_start = time.monotonic()

        # 2a. Update calendar FIRST so .bin files can be aligned
        logger.info(
            "Phase 2: updating calendar with %d new dates for market=%s",
            len(all_dates), market,
        )
        full_calendar: List[str] = []
        if all_dates:
            full_calendar = update_calendar(market_dir, sorted(all_dates))
        else:
            full_calendar = read_calendar(market_dir)
        logger.info(
            "Calendar updated: %d total dates for market=%s",
            len(full_calendar), market,
        )

        # 2b. Write .bin files with merge for incremental sync
        success_count = 0
        use_merge = update_only and bool(full_calendar)

        # Pre-compute calendar index once (avoids rebuilding per-symbol)
        cal_index = build_calendar_index(full_calendar) if full_calendar else None

        logger.info(
            "Phase 2: writing .bin files for %d symbols "
            "(merge=%s, calendar_size=%d)",
            len(collected), use_merge, len(full_calendar),
        )

        # Collect instrument updates for batch write at the end
        instrument_updates = []

        for qlib_sym, df in collected.items():
            try:
                ok = dataframe_to_bin(
                    df,
                    qlib_sym,
                    market_dir,
                    calendar=full_calendar if full_calendar else None,
                    merge=use_merge,
                    cal_index=cal_index,
                )
                if ok:
                    dates = sorted(d.strftime("%Y-%m-%d") for d in df.index)
                    instrument_updates.append((qlib_sym, dates[0], dates[-1]))
                    success_count += 1
                    if success_count % 500 == 0 or success_count == len(collected):
                        _write_sync_progress(data_dir, market, {
                            "status": "syncing",
                            "phase": "writing",
                            "current": success_count,
                            "total": len(collected),
                            "percent": 50 + round(success_count / len(collected) * 50),  # Phase 2 = 50-100%
                            "started_at": sync_started_at,
                        })
            except Exception as e:
                logger.warning("  Failed to write .bin for %s: %s", qlib_sym, e)
                errors.append(f"{qlib_sym}: bin write error: {e}")

        # Batch update instruments file (single read-modify-write)
        if instrument_updates:
            batch_update_instruments(market_dir, instrument_updates)

        phase2_elapsed = time.monotonic() - phase2_start
        logger.info(
            "Phase 2 complete: market=%s, wrote %d/%d symbols, "
            "%d errors (%.1fs)",
            market, success_count, len(collected), len(errors), phase2_elapsed,
        )

        return {
            "symbol_count": success_count,
            "new_symbols": success_count,
            "errors": errors,
        }

    # ------------------------------------------------------------------ #
    # Helper methods
    # ------------------------------------------------------------------ #

    @staticmethod
    def _resolve_start_date(
        market_dir: str,
        update_only: bool,
        default: str = "2000-01-01",
        date_format: str = "%Y-%m-%d",
    ) -> str:
        """Determine start date: last calendar entry if update_only, else default.

        Uses the last calendar date (inclusive) rather than last+1, because
        a previous sync may have run before daily bar collection finished,
        leaving some symbols missing for the last calendar date. Re-fetching
        it is safe since Phase 2 uses merge mode for incremental syncs.
        """
        if not update_only:
            return default

        cal_path = os.path.join(market_dir, "calendars", "day.txt")
        if os.path.exists(cal_path):
            try:
                lines = Path(cal_path).read_text().strip().split("\n")
                if lines:
                    last_date = sorted(lines)[-1].strip()
                    logger.info(
                        "Update mode: starting from %s (last calendar date, inclusive)",
                        last_date,
                    )
                    return last_date
            except Exception as e:
                logger.warning(
                    "Failed to read calendar for update_only, using default: %s", e
                )

        return default

    @staticmethod
    def _get_date_range(market_dir: str) -> Optional[Dict[str, str]]:
        """Read calendar file and return {start, end} date range."""
        cal_path = os.path.join(market_dir, "calendars", "day.txt")
        if not os.path.exists(cal_path):
            return None

        try:
            lines = Path(cal_path).read_text().strip().split("\n")
            lines = [l.strip() for l in lines if l.strip()]
            if lines:
                sorted_dates = sorted(lines)
                return {"start": sorted_dates[0], "end": sorted_dates[-1]}
        except Exception as e:
            logger.warning("Failed to read calendar at %s: %s", cal_path, e)

        return None

    @staticmethod
    def _count_instruments(market_dir: str) -> int:
        """Count instruments in instruments/all.txt."""
        inst_path = os.path.join(market_dir, "instruments", "all.txt")
        if not os.path.exists(inst_path):
            return 0

        try:
            lines = Path(inst_path).read_text().strip().split("\n")
            return len([l for l in lines if l.strip()])
        except Exception as e:
            logger.debug("Failed to count instruments at %s: %s", inst_path, e)
            return 0

    @staticmethod
    def _save_metadata(
        data_dir: str, market: str, result: Dict[str, Any]
    ) -> None:
        """Update sync_metadata.json with latest sync result."""
        meta_path = os.path.join(data_dir, "sync_metadata.json")
        metadata: Dict[str, Any] = {}

        if os.path.exists(meta_path):
            try:
                with open(meta_path) as f:
                    metadata = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to read sync_metadata.json, starting fresh: %s", e)

        # Determine market data dir for date range
        subdir = QlibContext.MARKET_TO_DATA_DIR.get(market, f"{market}_data")
        market_dir = os.path.join(data_dir, subdir)
        date_range = DataSyncService._get_date_range(market_dir)

        metadata[market] = {
            "last_sync": datetime.now().isoformat(),
            "symbol_count": result.get("symbol_count", 0),
            "error_count": len(result.get("errors", [])),
            "duration_s": result.get("duration_s", 0),
            "date_range": date_range,
        }

        try:
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)
        except (OSError, TypeError) as e:
            logger.warning("Failed to save sync metadata for market=%s: %s", market, e)
