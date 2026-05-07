"""HTTP clients for fetching market data from StockPulse.

Two clients:
1. StockPulseClient (synchronous) -- for ProcessPoolExecutor (DataSyncService)
   where all I/O must be synchronous.
2. StockPulseAsyncClient (async) -- for ML services (feature_service,
   prediction_service, direction_service, market_features_service) where
   all data access goes through StockPulse HTTP API.

Both authenticate via X-API-Key header.

StockPulse API endpoints consumed:
    Sync:
        GET  /api/v1/data/internal/symbols/{market}
        POST /api/v1/data/internal/history/batch
    Async (ML data):
        POST /api/v1/data/ml/batch/financials
        POST /api/v1/data/ml/batch/analyst
        POST /api/v1/data/ml/batch/earnings
        POST /api/v1/data/ml/batch/options
        POST /api/v1/data/ml/batch/insider
        POST /api/v1/data/ml/batch/sectors
        GET  /api/v1/data/ml/market/breadth/{market}
        GET  /api/v1/data/ml/market/volume/{market}
        GET  /api/v1/data/ml/market/sector-returns/{market}
        GET  /api/v1/data/history/{symbol}

    Prediction write/read methods are temporarily disabled pending
    migration to local DB storage.
"""

import logging
from datetime import datetime
from typing import Any

import httpx
from httpx import HTTPTransport

from app.config import get_settings

logger = logging.getLogger(__name__)


class StockPulseClient:
    """Synchronous HTTP client for fetching market data from StockPulse.

    Uses X-API-Key authentication. Designed for use in ProcessPoolExecutor
    (DataSyncService) where async I/O is not available.
    """

    def __init__(self, base_url: str | None = None, api_key: str | None = None) -> None:
        settings = get_settings()
        self.base_url = base_url or settings.STOCKPULSE_URL
        self.api_key = api_key or settings.STOCKPULSE_API_KEY
        # Transport retries handle transient connection errors (refused, reset)
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(120.0, connect=10.0),
            headers={"X-API-Key": self.api_key},
            transport=HTTPTransport(retries=2),
        )
        logger.info(
            "StockPulseClient initialized: base_url=%s, key_configured=%s",
            self.base_url, bool(self.api_key),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_symbols(self, market: str) -> list[str]:
        """Fetch symbol list for a market from StockPulse.

        Raises RuntimeError on failure.
        """
        try:
            resp = self._client.get(f"/api/v1/data/internal/symbols/{market}")
            resp.raise_for_status()
            data = resp.json()
            symbols = data.get("symbols", [])
            logger.info(
                "Fetched %d symbols for market=%s from StockPulse",
                len(symbols),
                market,
            )
            return symbols
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"StockPulse returned HTTP {e.response.status_code} "
                f"for symbols/{market}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to fetch symbols for market={market}: "
                f"[{type(e).__name__}] {e}"
            ) from e

    def get_daily_bars(
        self,
        symbols: list[str],
        market: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict:
        """Fetch daily bars for a batch of symbols from StockPulse.

        Returns columnar format::

            {"AAPL": {"dates": [...], "open": [...], "high": [...], ...}}

        Raises RuntimeError on failure.
        """
        payload: dict = {
            "symbols": symbols,
            "market": market,
        }
        if start_date:
            payload["startDate"] = start_date
        if end_date:
            payload["endDate"] = end_date

        try:
            resp = self._client.post(
                "/api/v1/data/internal/history/batch",
                json=payload,
                timeout=httpx.Timeout(190.0, connect=10.0),
            )
            resp.raise_for_status()
            data = resp.json().get("data", {})
            total_bars = sum(len(v.get("dates", [])) for v in data.values())
            logger.info(
                "Fetched daily bars: market=%s, requested=%d, received=%d symbols, %d bars",
                market, len(symbols), len(data), total_bars,
            )
            if len(data) < len(symbols):
                missing = set(symbols) - set(data.keys())
                logger.warning(
                    "StockPulse returned data for %d/%d symbols (market=%s), missing: %s",
                    len(data), len(symbols), market, list(missing)[:10],
                )
            return data
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"StockPulse returned HTTP {e.response.status_code} "
                f"for history batch ({len(symbols)} symbols, market={market})"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to fetch daily bars for {len(symbols)} symbols "
                f"(market={market}): [{type(e).__name__}] {e}"
            ) from e

    def is_available(self) -> bool:
        """Check if StockPulse API is reachable."""
        if not self.api_key:
            logger.debug("StockPulseClient: no STOCKPULSE_API_KEY configured")
            return False
        try:
            resp = self._client.get("/health")
            return resp.status_code == 200
        except Exception as e:
            logger.debug("StockPulse health check failed: %s", e)
            return False

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()


# ------------------------------------------------------------------
# Module-level singleton
# ------------------------------------------------------------------

_client: StockPulseClient | None = None


def get_stockpulse_client() -> StockPulseClient:
    """Get or create a singleton StockPulseClient."""
    global _client
    if _client is None:
        _client = StockPulseClient()
    return _client


def reset_stockpulse_client() -> None:
    """Reset the singleton. Must be called after fork in child processes."""
    global _client
    if _client is not None:
        _client.close()
    _client = None


# ======================================================================
# Async client for ML services
# ======================================================================


async def _get_stockpulse_config() -> tuple[str, str]:
    """Read StockPulse URL and API key from system_settings DB, with env fallback.

    Returns (base_url, api_key). Catches all exceptions so callers always
    get a usable pair even if the DB is not yet ready.
    """
    settings = get_settings()
    base_url = settings.STOCKPULSE_URL
    api_key = settings.STOCKPULSE_API_KEY

    try:
        from sqlalchemy import select
        from app.core.orm import get_session_factory
        from app.models.system_setting import SystemSetting

        factory = get_session_factory()
        async with factory() as session:
            url_result = await session.execute(
                select(SystemSetting).where(SystemSetting.key == "stockpulse_url")
            )
            url_setting = url_result.scalar_one_or_none()
            if url_setting and url_setting.value:
                base_url = url_setting.value

            key_result = await session.execute(
                select(SystemSetting).where(SystemSetting.key == "stockpulse_api_key")
            )
            key_setting = key_result.scalar_one_or_none()
            if key_setting and key_setting.value:
                api_key = key_setting.value
    except Exception as e:
        logger.debug(
            "Could not read StockPulse config from DB, using env vars: %s", e,
        )

    return base_url.rstrip("/"), api_key


class StockPulseAsyncClient:
    """Async HTTP client for ML data access via StockPulse API.

    Used by feature_service, prediction_service, direction_service, and
    market_features_service. All ML data (fundamentals, bars, analyst,
    sectors, predictions, models, factors) is accessed through this client
    -- AlphaForge never queries StockPulse's PostgreSQL directly.
    """

    def __init__(self) -> None:
        self._base_url: str | None = None
        self._api_key: str | None = None
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._base_url, self._api_key = await _get_stockpulse_config()
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers={"X-API-Key": self._api_key},
                timeout=120.0,
            )
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    @staticmethod
    def _date_range_to_days(start_date: str, end_date: str) -> int:
        """Convert a start_date/end_date pair to a ``days`` count.

        The new StockPulse market-feature endpoints accept a ``days``
        query parameter instead of start_date/end_date. This helper
        computes the number of calendar days between the two dates,
        with a minimum of 1.
        """
        try:
            d_start = datetime.strptime(start_date, "%Y-%m-%d")
            d_end = datetime.strptime(end_date, "%Y-%m-%d")
            delta = (d_end - d_start).days
            return max(delta, 1)
        except (ValueError, TypeError):
            logger.warning(
                "_date_range_to_days: could not parse dates "
                "start_date=%r end_date=%r, defaulting to 90",
                start_date, end_date,
            )
            return 90

    # ------------------------------------------------------------------
    # READ methods (from StockPulse DB tables)
    # ------------------------------------------------------------------

    async def get_fundamentals_batch(
        self,
        symbols: list[str],
        market: str,
        start_date: str,
        end_date: str,
    ) -> list[dict]:
        """Fetch fundamental data for multiple symbols over date range.

        Calls POST /api/v1/data/ml/batch/financials. The response nests
        data under ``data.symbols`` keyed by symbol, each containing
        ``sec_filings`` and ``valuation`` arrays. This method flattens
        the nested structure into a flat list of dicts compatible with
        the downstream feature builder.
        """
        client = await self._get_client()
        body: dict[str, Any] = {"symbols": symbols, "market": market}
        if start_date:
            body["start_date"] = start_date
        if end_date:
            body["end_date"] = end_date
        try:
            resp = await client.post(
                "/api/v1/data/ml/batch/financials", json=body,
            )
            resp.raise_for_status()
        except Exception as e:
            logger.warning("get_fundamentals_batch failed: %s", e)
            return []

        symbols_data = resp.json().get("data", {}).get("symbols", {})
        if not isinstance(symbols_data, dict):
            logger.warning(
                "get_fundamentals_batch: unexpected response format, "
                "expected dict under data.symbols, got %s",
                type(symbols_data).__name__,
            )
            return []

        result: list[dict] = []
        for symbol, payload in symbols_data.items():
            if not isinstance(payload, dict):
                continue

            # --- Valuation array: emit ALL records per symbol ---
            # Fundamentals are quarterly; the downstream merge does a
            # left join on (symbol, date) then forward-fills so each
            # trading day gets the latest available value.
            valuation_entries = payload.get("valuation", [])
            if not isinstance(valuation_entries, list):
                valuation_entries = []

            for entry in valuation_entries:
                if not isinstance(entry, dict) or not entry.get("date"):
                    continue
                row: dict[str, Any] = {"symbol": symbol}
                row["date"] = entry.get("date", "")
                # Direct valuation fields
                row["pe_ratio"] = entry.get("pe_ratio")
                row["pb_ratio"] = entry.get("pb_ratio")
                row["ps_ratio"] = entry.get("ps_ratio")
                row["roe"] = entry.get("roe")
                row["roa"] = entry.get("roa")
                row["eps"] = entry.get("eps")
                row["market_cap"] = entry.get("market_cap")
                row["book_value"] = entry.get("book_value")
                row["debt_to_equity"] = entry.get("debt_to_equity")
                row["current_ratio"] = entry.get("current_ratio")
                row["payout_ratio"] = entry.get("payout_ratio")
                row["operating_margin"] = entry.get("operating_margin")
                row["gross_margin"] = entry.get("gross_margin")
                # Map net_margin -> profit_margin if profit_margin not present
                row["profit_margin"] = entry.get("profit_margin") or entry.get("net_margin")
                # Optional fields (may be absent in valuation)
                row["dividend_yield"] = entry.get("dividend_yield")
                row["dividend_rate"] = entry.get("dividend_rate")
                row["forward_pe"] = entry.get("forward_pe")
                row["revenue_growth_yoy"] = entry.get("revenue_growth_yoy")
                row["eps_growth"] = entry.get("eps_growth")
                row["net_cash_ratio"] = entry.get("net_cash_ratio")
                row["short_pct_float"] = entry.get("short_pct_float")
                row["short_ratio"] = entry.get("short_ratio")
                result.append(row)

            # --- sec_filings: merge additional fields not in valuation ---
            for entry in payload.get("sec_filings", []):
                if not isinstance(entry, dict):
                    continue
                row = {"symbol": symbol}
                row["date"] = entry.get("date", entry.get("filed_date", ""))
                row["revenue"] = entry.get("revenue")
                row["net_income"] = entry.get("net_income")
                row["eps"] = entry.get("eps")
                row["total_assets"] = entry.get("total_assets")
                row["total_debt"] = entry.get("total_debt")
                row["free_cash_flow"] = entry.get("free_cash_flow")
                result.append(row)
        logger.info(
            "get_fundamentals_batch: %d symbols requested, %d rows produced",
            len(symbols), len(result),
        )
        return result

    async def get_analyst_batch(
        self,
        symbols: list[str],
        market: str,
        start_date: str,
        end_date: str,
    ) -> list[dict]:
        """Fetch analyst data batch.

        Calls POST /api/v1/data/ml/batch/analyst. Response nests data
        under ``data.symbols`` with ``recommendations`` and
        ``upgrades_downgrades`` per symbol. Flattened to a list of dicts
        with computed ``analyst_buy_ratio`` and ``analyst_net_score``.
        """
        client = await self._get_client()
        body: dict[str, Any] = {"symbols": symbols, "market": market}
        if start_date:
            body["start_date"] = start_date
        if end_date:
            body["end_date"] = end_date
        try:
            resp = await client.post(
                "/api/v1/data/ml/batch/analyst",
                json=body,
            )
            resp.raise_for_status()
        except Exception as e:
            logger.warning("get_analyst_batch failed: %s", e)
            return []

        symbols_data = resp.json().get("data", {}).get("symbols", {})
        if not isinstance(symbols_data, dict):
            logger.warning(
                "get_analyst_batch: unexpected response format, "
                "expected dict under data.symbols, got %s",
                type(symbols_data).__name__,
            )
            return []

        result: list[dict] = []
        for symbol, payload in symbols_data.items():
            if not isinstance(payload, dict):
                continue
            for rec in payload.get("recommendations", []):
                if not isinstance(rec, dict):
                    continue
                strong_buy = rec.get("strong_buy", 0) or 0
                buy = rec.get("buy", 0) or 0
                hold = rec.get("hold", 0) or 0
                sell = rec.get("sell", 0) or 0
                strong_sell = rec.get("strong_sell", 0) or 0
                total = strong_buy + buy + hold + sell + strong_sell
                buy_ratio = (strong_buy + buy) / total if total > 0 else 0.0
                # Net score: +2 strong_buy, +1 buy, 0 hold, -1 sell, -2 strong_sell
                net_score = (
                    (2 * strong_buy + buy - sell - 2 * strong_sell) / total
                    if total > 0 else 0.0
                )
                result.append({
                    "symbol": symbol,
                    "date": rec.get("period", ""),
                    "analyst_buy_ratio": round(buy_ratio, 4),
                    "analyst_net_score": round(net_score, 4),
                    "strong_buy": strong_buy,
                    "buy": buy,
                    "hold": hold,
                    "sell": sell,
                    "strong_sell": strong_sell,
                    "total_analysts": total,
                })
            for ud in payload.get("upgrades_downgrades", []):
                if not isinstance(ud, dict):
                    continue
                result.append({
                    "symbol": symbol,
                    "date": ud.get("date", ""),
                    "firm": ud.get("firm"),
                    "to_grade": ud.get("to_grade"),
                    "from_grade": ud.get("from_grade"),
                    "action": ud.get("action"),
                })
        logger.info(
            "get_analyst_batch: %d symbols requested, %d rows produced",
            len(symbols), len(result),
        )
        return result

    async def get_earnings_batch(
        self,
        symbols: list[str],
        market: str,
        start_date: str,
        end_date: str,
    ) -> list[dict]:
        """Fetch earnings event data batch.

        Calls POST /api/v1/data/ml/batch/earnings. Response nests data
        under ``data.symbols`` with ``surprises`` and ``calendar`` per
        symbol. Flattened to a flat list of dicts.
        """
        client = await self._get_client()
        body: dict[str, Any] = {"symbols": symbols, "market": market}
        if start_date:
            body["start_date"] = start_date
        if end_date:
            body["end_date"] = end_date
        try:
            resp = await client.post(
                "/api/v1/data/ml/batch/earnings",
                json=body,
            )
            resp.raise_for_status()
        except Exception as e:
            logger.warning("get_earnings_batch failed: %s", e)
            return []

        symbols_data = resp.json().get("data", {}).get("symbols", {})
        if not isinstance(symbols_data, dict):
            logger.warning(
                "get_earnings_batch: unexpected response format, "
                "expected dict under data.symbols, got %s",
                type(symbols_data).__name__,
            )
            return []

        result: list[dict] = []
        for symbol, payload in symbols_data.items():
            if not isinstance(payload, dict):
                continue
            for surprise in payload.get("surprises", []):
                if not isinstance(surprise, dict):
                    continue
                row = {"symbol": symbol, **surprise}
                result.append(row)
            for cal in payload.get("calendar", []):
                if not isinstance(cal, dict):
                    continue
                row = {"symbol": symbol, **cal}
                result.append(row)
        logger.info(
            "get_earnings_batch: %d symbols requested, %d rows produced",
            len(symbols), len(result),
        )
        return result

    async def get_options_batch(
        self,
        symbols: list[str],
        market: str,
        start_date: str,
        end_date: str,
    ) -> list[dict]:
        """Fetch options put/call ratio data batch.

        Calls POST /api/v1/data/ml/batch/options. Response nests data
        under ``data.symbols`` with per-symbol ``data`` array. Flattened
        to a list of dicts with symbol, date, put_call_ratio.
        """
        client = await self._get_client()
        body: dict[str, Any] = {"symbols": symbols, "market": market}
        if start_date:
            body["start_date"] = start_date
        if end_date:
            body["end_date"] = end_date
        try:
            resp = await client.post(
                "/api/v1/data/ml/batch/options",
                json=body,
            )
            resp.raise_for_status()
        except Exception as e:
            logger.warning("get_options_batch failed: %s", e)
            return []

        symbols_data = resp.json().get("data", {}).get("symbols", {})
        if not isinstance(symbols_data, dict):
            logger.warning(
                "get_options_batch: unexpected response format, "
                "expected dict under data.symbols, got %s",
                type(symbols_data).__name__,
            )
            return []

        result: list[dict] = []
        for symbol, payload in symbols_data.items():
            if not isinstance(payload, dict):
                continue
            entries = payload.get("data", [])
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                result.append({
                    "symbol": symbol,
                    "date": entry.get("date", ""),
                    "put_call_ratio": entry.get("put_call_ratio"),
                    "put_volume": entry.get("put_volume"),
                    "call_volume": entry.get("call_volume"),
                })
        logger.info(
            "get_options_batch: %d symbols requested, %d rows produced",
            len(symbols), len(result),
        )
        return result

    async def get_insider_batch(
        self,
        symbols: list[str],
        market: str,
        start_date: str,
        end_date: str,
    ) -> list[dict]:
        """Fetch insider trading data batch.

        Calls POST /api/v1/data/ml/batch/insider. Response nests data
        under ``data.symbols`` with ``transactions`` and ``sentiment``
        per symbol. Flattened to a flat list of dicts.
        """
        client = await self._get_client()
        body: dict[str, Any] = {"symbols": symbols, "market": market}
        if start_date:
            body["start_date"] = start_date
        if end_date:
            body["end_date"] = end_date
        try:
            resp = await client.post(
                "/api/v1/data/ml/batch/insider",
                json=body,
            )
            resp.raise_for_status()
        except Exception as e:
            logger.warning("get_insider_batch failed: %s", e)
            return []

        symbols_data = resp.json().get("data", {}).get("symbols", {})
        if not isinstance(symbols_data, dict):
            logger.warning(
                "get_insider_batch: unexpected response format, "
                "expected dict under data.symbols, got %s",
                type(symbols_data).__name__,
            )
            return []

        result: list[dict] = []
        for symbol, payload in symbols_data.items():
            if not isinstance(payload, dict):
                continue

            transactions = payload.get("transactions", [])
            sentiment_list = payload.get("sentiment", [])

            # Group transactions by month and compute net_shares_pct
            monthly_txns: dict[str, dict] = {}
            for txn in transactions:
                if not isinstance(txn, dict):
                    continue
                txn_date = txn.get("date", "")
                if not txn_date or len(txn_date) < 7:
                    continue
                month_key = txn_date[:7]  # "YYYY-MM"
                if month_key not in monthly_txns:
                    monthly_txns[month_key] = {"buy_shares": 0.0, "sell_shares": 0.0}
                shares = abs(float(txn.get("shares", 0) or 0))
                txn_type = (txn.get("transaction_type") or "").lower()
                if "buy" in txn_type or "purchase" in txn_type:
                    monthly_txns[month_key]["buy_shares"] += shares
                elif "sell" in txn_type or "sale" in txn_type:
                    monthly_txns[month_key]["sell_shares"] += shares

            # Build a map of month -> mspr from sentiment data
            sentiment_by_month: dict[str, float] = {}
            for sent in sentiment_list:
                if not isinstance(sent, dict):
                    continue
                month_key = sent.get("month", "")
                if month_key:
                    sentiment_by_month[month_key] = float(sent.get("mspr", 0) or 0)

            # Emit one row per (symbol, month)
            all_months = set(monthly_txns.keys()) | set(sentiment_by_month.keys())
            for month_key in sorted(all_months):
                agg = monthly_txns.get(month_key, {"buy_shares": 0.0, "sell_shares": 0.0})
                net_shares = agg["buy_shares"] - agg["sell_shares"]
                total_shares = agg["buy_shares"] + agg["sell_shares"]
                net_shares_pct = (
                    net_shares / total_shares if total_shares > 0 else 0.0
                )
                # Use mspr (monthly share purchase ratio) as insider_ownership_pct proxy
                insider_ownership_pct = sentiment_by_month.get(month_key, 0.0)

                result.append({
                    "symbol": symbol,
                    "date": f"{month_key}-01",
                    "net_shares_pct": net_shares_pct,
                    "insider_ownership_pct": insider_ownership_pct,
                })
        logger.info(
            "get_insider_batch: %d symbols requested, %d rows produced",
            len(symbols), len(result),
        )
        return result

    async def get_sentiment_batch(
        self,
        symbols: list[str],
        market: str,
        start_date: str,
        end_date: str,
    ) -> list[dict]:
        """Fetch news sentiment feature data batch.

        NOTE: The sentiment batch endpoint has been removed from
        StockPulse. This method returns an empty list for backward
        compatibility. Sentiment data should be sourced from NewsForge
        in a future integration.
        """
        logger.warning(
            "get_sentiment_batch called but StockPulse no longer provides "
            "a sentiment batch endpoint. Returning empty list. "
            "symbols=%d, market=%s",
            len(symbols), market,
        )
        return []

    async def get_sectors(
        self, market: str, symbols: list[str] | None = None,
    ) -> list[dict]:
        """Fetch sector mappings for a market.

        Calls POST /api/v1/data/ml/batch/sectors which requires a
        symbols list. If no symbols are provided, returns an empty list
        since the endpoint requires them.
        """
        if not symbols:
            logger.debug(
                "get_sectors called without symbols for market=%s, "
                "returning empty list (batch/sectors requires symbols)",
                market,
            )
            return []

        client = await self._get_client()
        try:
            resp = await client.post(
                "/api/v1/data/ml/batch/sectors",
                json={"symbols": symbols, "market": market},
            )
            resp.raise_for_status()
        except Exception as e:
            logger.warning("get_sectors failed: %s", e)
            return []

        symbols_data = resp.json().get("data", {}).get("symbols", {})
        if not isinstance(symbols_data, dict):
            logger.warning(
                "get_sectors: unexpected response format, "
                "expected dict under data.symbols, got %s",
                type(symbols_data).__name__,
            )
            return []

        result: list[dict] = []
        for symbol, payload in symbols_data.items():
            if not isinstance(payload, dict):
                continue
            row: dict[str, Any] = {
                "symbol": symbol,
                "sector": payload.get("sector", ""),
                "industry": payload.get("industry", ""),
            }
            if payload.get("market_cap") is not None:
                row["market_cap"] = float(payload["market_cap"])
            result.append(row)
        return result

    async def get_sector_map(
        self, market: str, symbols: list[str] | None = None,
    ) -> dict[str, str]:
        """Fetch sector map {symbol: sector} for a market.

        Convenience wrapper over get_sectors() that converts the list
        format to a dict for direct use in feature ranking/neutralization.

        The underlying batch/sectors endpoint requires a symbols list.
        When called without symbols, returns an empty dict.
        """
        if not symbols:
            logger.debug(
                "get_sector_map called without symbols for market=%s, "
                "returning empty dict",
                market,
            )
            return {}
        sectors = await self.get_sectors(market, symbols=symbols)
        return {s["symbol"]: s["sector"] for s in sectors if s.get("sector")}

    async def get_market_caps(
        self, market: str, symbols: list[str],
    ) -> dict[str, float]:
        """Get market cap for symbols. Returns {symbol: market_cap_millions}."""
        sectors_data = await self.get_sectors(market, symbols=symbols)
        return {
            row["symbol"]: row["market_cap"]
            for row in sectors_data
            if row.get("market_cap") and row["market_cap"] > 0
        }

    async def get_universes(self) -> list[dict]:
        """Fetch all prediction universes.

        NOTE: Universe endpoints no longer exist in StockPulse. Returns
        an empty list for backward compatibility. Universe management
        will be handled locally in a future release.
        """
        logger.debug("get_universes: endpoint not available in StockPulse, returning []")
        return []

    async def get_universe(self, universe_id: str) -> dict | None:
        """Fetch a specific prediction universe by ID.

        NOTE: Universe endpoints no longer exist in StockPulse. Returns
        None for backward compatibility.
        """
        logger.debug(
            "get_universe(%s): endpoint not available in StockPulse, returning None",
            universe_id,
        )
        return None

    async def get_universe_symbols(self, market: str) -> list[str]:
        """Get default universe symbols for a market."""
        client = await self._get_client()
        resp = await client.get(f"/api/v1/data/internal/symbols/{market}")
        resp.raise_for_status()
        data = resp.json()
        return data.get("symbols", data.get("data", []))

    async def get_bars_batch_async(
        self,
        symbols: list[str],
        market: str,
        start_date: str,
        end_date: str,
    ) -> dict:
        """Async version of get_daily_bars for ML services.

        Returns columnar format:
            {"AAPL": {"dates": [...], "open": [...], "high": [...], ...}}
        """
        client = await self._get_client()
        resp = await client.post(
            "/api/v1/data/internal/history/batch",
            json={
                "symbols": symbols,
                "market": market,
                "startDate": start_date,
                "endDate": end_date,
            },
            timeout=300.0,
        )
        resp.raise_for_status()
        return resp.json().get("data", {})

    async def get_index_history(
        self,
        symbol: str,
        market: str,
        start_date: str,
        end_date: str,
    ) -> list[dict]:
        """Fetch index history (for market-level features).

        Uses GET /api/v1/data/history/{symbol} with date range params.
        StockPulse returns ``{"data": {"symbol": "...", "bars": [...]}}``.
        This method extracts the ``bars`` array and returns a list of
        ``{date, open, high, low, close, volume}`` dicts.
        """
        client = await self._get_client()
        try:
            resp = await client.get(
                f"/api/v1/data/history/{symbol}",
                params={
                    "start_date": start_date,
                    "end_date": end_date,
                },
                timeout=60.0,
            )
            resp.raise_for_status()
        except Exception as e:
            logger.warning("get_index_history failed for %s: %s", symbol, e)
            return []
        outer = resp.json().get("data") or {}
        if isinstance(outer, dict):
            return outer.get("bars", []) or []
        return []

    async def get_market_breadth(
        self,
        market: str,
        start_date: str,
        end_date: str,
    ) -> list[dict]:
        """Fetch market breadth data (advancers, decliners per date).

        Calls GET /api/v1/data/ml/market/breadth/{market}?days=N.
        The ``days`` param is derived from the date range.
        """
        days = self._date_range_to_days(start_date, end_date)
        client = await self._get_client()
        try:
            resp = await client.get(
                f"/api/v1/data/ml/market/breadth/{market}",
                params={"days": days},
            )
            resp.raise_for_status()
        except Exception as e:
            logger.warning("get_market_breadth failed for %s: %s", market, e)
            return []
        return resp.json().get("data", {}).get("data", [])

    async def get_market_volume(
        self,
        market: str,
        start_date: str,
        end_date: str,
    ) -> list[dict]:
        """Fetch market-wide volume data per date.

        Calls GET /api/v1/data/ml/market/volume/{market}?days=N.
        """
        days = self._date_range_to_days(start_date, end_date)
        client = await self._get_client()
        try:
            resp = await client.get(
                f"/api/v1/data/ml/market/volume/{market}",
                params={"days": days},
            )
            resp.raise_for_status()
        except Exception as e:
            logger.warning("get_market_volume failed for %s: %s", market, e)
            return []
        return resp.json().get("data", {}).get("data", [])

    async def get_sector_returns(
        self,
        market: str,
        start_date: str,
        end_date: str,
    ) -> list[dict]:
        """Fetch per-sector return data.

        Calls GET /api/v1/data/ml/market/sector-returns/{market}?days=N.
        """
        days = self._date_range_to_days(start_date, end_date)
        client = await self._get_client()
        try:
            resp = await client.get(
                f"/api/v1/data/ml/market/sector-returns/{market}",
                params={"days": days},
                timeout=60.0,
            )
            resp.raise_for_status()
        except Exception as e:
            logger.warning("get_sector_returns failed for %s: %s", market, e)
            return []
        return resp.json().get("data", {}).get("data", [])

    # ------------------------------------------------------------------
    # WRITE/READ methods — prediction storage (PENDING LOCAL DB MIGRATION)
    #
    # These methods previously wrote to / read from StockPulse's
    # prediction tables via /api/v1/ml/* endpoints. Those endpoints have
    # been removed from StockPulse. Prediction storage is being migrated
    # to AlphaForge's local database. Until that migration is complete,
    # these methods raise NotImplementedError to prevent silent failures.
    # ------------------------------------------------------------------

    async def write_predictions_batch(self, predictions: list[dict]) -> dict:
        """Bulk write prediction scores to stock_predictions."""
        raise NotImplementedError(
            "Prediction storage is being migrated to local DB. "
            "write_predictions_batch is temporarily unavailable."
        )

    async def write_model(self, model_data: dict) -> dict:
        """Write model metadata to prediction_models."""
        raise NotImplementedError(
            "Prediction storage is being migrated to local DB. "
            "write_model is temporarily unavailable."
        )

    async def write_factors(self, factors: list[dict]) -> dict:
        """Write discovered factors."""
        raise NotImplementedError(
            "Prediction storage is being migrated to local DB. "
            "write_factors is temporarily unavailable."
        )

    async def backfill_returns(self, updates: list[dict]) -> dict:
        """Backfill actual_return in stock_predictions."""
        raise NotImplementedError(
            "Prediction storage is being migrated to local DB. "
            "backfill_returns is temporarily unavailable."
        )

    async def get_predictions_for_backfill(
        self, market: str, days: int = 30,
    ) -> list[dict]:
        """Get predictions that need actual_return backfill."""
        raise NotImplementedError(
            "Prediction storage is being migrated to local DB. "
            "get_predictions_for_backfill is temporarily unavailable."
        )

    async def get_latest_predictions(
        self, market: str, top_n: int = 50,
    ) -> list[dict]:
        """Get latest predictions for a market."""
        raise NotImplementedError(
            "Prediction storage is being migrated to local DB. "
            "get_latest_predictions is temporarily unavailable."
        )

    async def get_models(self, market: str | None = None) -> list[dict]:
        """List trained models, optionally filtered by market."""
        raise NotImplementedError(
            "Prediction storage is being migrated to local DB. "
            "get_models is temporarily unavailable."
        )

    async def get_factors(self, market: str | None = None) -> list[dict]:
        """List discovered factors, optionally filtered by market."""
        raise NotImplementedError(
            "Prediction storage is being migrated to local DB. "
            "get_factors is temporarily unavailable."
        )

    async def get_prediction_history(
        self, market: str, days: int = 30,
    ) -> list[dict]:
        """Get prediction history for the last N days."""
        raise NotImplementedError(
            "Prediction storage is being migrated to local DB. "
            "get_prediction_history is temporarily unavailable."
        )

    async def get_performance_metrics(
        self, market: str, days: int = 90,
    ) -> list[dict]:
        """Get performance metrics data (predictions with actual returns)."""
        raise NotImplementedError(
            "Prediction storage is being migrated to local DB. "
            "get_performance_metrics is temporarily unavailable."
        )

    async def update_up_probabilities(
        self,
        market: str,
        prediction_date: str,
        forward_days: int,
        updates: list[dict],
    ) -> dict:
        """Batch update up_probability on stock_predictions rows."""
        raise NotImplementedError(
            "Prediction storage is being migrated to local DB. "
            "update_up_probabilities is temporarily unavailable."
        )

    async def get_macro_batch(
        self,
        tickers: list[str],
        start_date: str,
        end_date: str,
    ) -> dict[str, list[dict]]:
        """Fetch macro indicator data (VIX, DXY, TNX) from StockPulse.

        Calls ``GET /api/v1/data/ml/macro/{ticker}`` for each ticker.

        Returns:
            ``{ticker: [{date, value}, ...]}``
        """
        client = await self._get_client()
        result: dict[str, list[dict]] = {}
        for ticker in tickers:
            try:
                resp = await client.get(
                    f"/api/v1/data/ml/macro/{ticker}",
                    params={"start_date": start_date, "end_date": end_date},
                    timeout=60.0,
                )
                resp.raise_for_status()
                outer = resp.json().get("data") or {}
                if isinstance(outer, dict):
                    result[ticker] = outer.get("data", []) or []
                else:
                    result[ticker] = []
            except Exception as e:
                logger.warning("Macro fetch failed for %s: %s", ticker, e)
                result[ticker] = []
        return result

    async def get_short_interest_batch(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
    ) -> list[dict]:
        """Fetch short interest data for multiple symbols from StockPulse.

        Calls ``POST /api/v1/data/ml/batch/short-interest`` which accepts
        up to 3000 symbols at once.

        Returns:
            Flat list of dicts, each with a ``symbol`` field added.
        """
        client = await self._get_client()
        body: dict[str, Any] = {"symbols": symbols, "market": "us"}
        if start_date:
            body["start_date"] = start_date
        if end_date:
            body["end_date"] = end_date
        try:
            resp = await client.post(
                "/api/v1/data/ml/batch/short-interest",
                json=body,
                timeout=120.0,
            )
            resp.raise_for_status()
        except Exception as e:
            logger.warning("get_short_interest_batch failed: %s", e)
            return []

        symbols_data = resp.json().get("data", {}).get("symbols", {})
        result: list[dict] = []
        for symbol, payload in symbols_data.items():
            if isinstance(payload, dict):
                rows = payload.get("data", [])
            elif isinstance(payload, list):
                rows = payload
            else:
                continue
            if not isinstance(rows, list):
                continue
            for row in rows:
                if isinstance(row, dict):
                    row["symbol"] = symbol
                    result.append(row)
        logger.info(
            "get_short_interest_batch: %d symbols requested, %d rows produced",
            len(symbols), len(result),
        )
        return result

    async def get_earnings_calendar(
        self,
        start_date: str,
        end_date: str,
    ) -> list[dict]:
        """Fetch earnings calendar from StockPulse.

        Calls ``GET /api/v1/data/ml/earnings-calendar``.

        Returns:
            List of ``{symbol, earnings_date}`` dicts.
        """
        client = await self._get_client()
        resp = await client.get(
            "/api/v1/data/ml/earnings-calendar",
            params={"start_date": start_date, "end_date": end_date},
            timeout=60.0,
        )
        resp.raise_for_status()
        outer = resp.json().get("data") or {}
        return outer.get("data", []) if isinstance(outer, dict) else []

    async def get_market_cap_latest(self, market: str) -> dict[str, float]:
        """Get latest market cap per symbol for return attribution.

        NOTE: Market cap endpoint is not available in StockPulse.
        Returns empty dict for backward compatibility. Market cap data
        can be obtained from fundamentals batch if needed.
        """
        logger.debug(
            "get_market_cap_latest(%s): endpoint not available in StockPulse, "
            "returning empty dict",
            market,
        )
        return {}


# ------------------------------------------------------------------
# Async singleton
# ------------------------------------------------------------------

_async_client: StockPulseAsyncClient | None = None


async def get_stockpulse_async_client() -> StockPulseAsyncClient:
    """Get or create a singleton StockPulseAsyncClient."""
    global _async_client
    if _async_client is None:
        _async_client = StockPulseAsyncClient()
    return _async_client


async def close_stockpulse_async_client() -> None:
    """Close the async singleton. Call during app shutdown."""
    global _async_client
    if _async_client is not None:
        await _async_client.close()
        _async_client = None
