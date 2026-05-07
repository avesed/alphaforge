"""Unified feature pipeline for ML prediction.

Merges three feature sources into a single DataFrame:
1. Alpha158 (65 OHLCV-based features from Qlib D.features())
2. Fundamental data (~19 financial metrics, forward-filled)
3. News sentiment (~11 rolling aggregates)

The merged matrix is rank-transformed (cross-sectional percentile)
for each date, which normalizes features for LightGBM ranking.

AlphaForge adaptation:
- All external data (fundamentals, sentiment, analyst, earnings, options,
  insider, sectors) is fetched via StockPulseAsyncClient HTTP API.
- Qlib Alpha158 features are computed locally (same as data-processor).
- No direct database queries.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import pandas as pd

from app.config import get_settings
from app.services.market_config import MarketConfig, get_market_config
from app.services.factor_service import FEATURE_NAMES
from app.services.stockpulse_client import get_stockpulse_async_client

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Feature column definitions
# -----------------------------------------------------------------------

# 86 Alpha158 OHLCV-based features (from factor_service.py)
ALPHA158_FEATURES: list[str] = list(FEATURE_NAMES)

# Fundamental financial metrics from StockPulse.
# Direct from valuation: pe_ratio..payout_ratio.
# Derived from sec_filings: revenue_growth_yoy, eps_growth, dividend_yield, net_cash_ratio.
# forward_pe/dividend_rate require analyst estimates — not available.
# short_pct_float/short_ratio come from the short interest endpoint, not here.
FUNDAMENTAL_FEATURES: list[str] = [
    "pe_ratio", "pb_ratio", "ps_ratio", "roe", "roa",
    "profit_margin", "gross_margin", "eps",
    "debt_to_equity", "current_ratio", "market_cap", "book_value",
    "operating_margin", "payout_ratio",
    "revenue_growth_yoy", "eps_growth", "dividend_yield", "net_cash_ratio",
]

# ~11 sentiment rolling features
SENTIMENT_FEATURES: list[str] = [
    "sentiment_avg",
    "article_count",
    "bullish_ratio",
    "content_score_avg",
    "sentiment_7d_ma",
    "sentiment_30d_ma",
    "article_count_7d",
    "bullish_ratio_7d",
    "sentiment_volatility_7d",
    "has_news_7d",
    "has_news_30d",
]

# EPS surprise excluded from training (too sparse, see CLAUDE.md)
EARNINGS_FEATURES: list[str] = []

# Analyst snapshots
ANALYST_FEATURES: list[str] = [
    "analyst_buy_ratio", "analyst_net_score",
    "eps_revision_score", "target_premium", "growth_est_next_y",
]

# Insider activity
INSIDER_FEATURES: list[str] = ["net_shares_pct", "insider_ownership_pct"]

# Options put/call ratio (US only) — also serves as sentiment proxy
OPTIONS_FEATURES: list[str] = ["put_call_ratio", "put_call_oi_ratio"]

# Earnings calendar features
EARNINGS_CALENDAR_FEATURES: list[str] = ["days_to_earnings"]

# Cross-feature interactions
INTERACTION_FEATURES: list[str] = [
    "momentum_vol_ratio",
    "volume_price_confirm",
    "momentum_divergence",
    "drawdown_recovery",
    "trend_vol_interaction",
    "price_ma_volume",
    "volatility_acceleration",
    "momentum_acceleration",
    "value_momentum",
    "yield_vol_adj",
]

# ThreadPoolExecutor for synchronous Qlib calls (D.features)
_qlib_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="feature-qlib")


class FeatureService:
    """Build unified feature matrices for ML training and inference.

    Merges Alpha158 technical features (Qlib), fundamental financial
    metrics, and news sentiment aggregates. Applies cross-sectional
    rank normalization to produce LightGBM-ready input.

    Data access: all non-Qlib data is fetched via StockPulseAsyncClient.
    """

    async def build_feature_matrix(
        self,
        market: str,
        symbols: list[str],
        start_date: str,
        end_date: str,
        include_fundamental: bool = True,
        include_sentiment: bool = True,
        config_override: MarketConfig | None = None,
    ) -> pd.DataFrame:
        """Build a merged, rank-normalized feature matrix.

        Args:
            market: Market code (us, hk, cn, etc.).
            symbols: List of stock symbols.
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            include_fundamental: Whether to include fundamental features.
            include_sentiment: Whether to include sentiment features.

        Returns:
            DataFrame with columns: symbol, date, + feature columns.
            All feature values are rank-transformed to [0, 1] percentiles
            cross-sectionally (per date). Empty DataFrame on total failure.
        """
        market = market.lower()

        if not symbols:
            logger.warning("build_feature_matrix called with empty symbol list")
            return pd.DataFrame()

        logger.info(
            "Building feature matrix: market=%s, symbols=%d, %s~%s, "
            "fundamental=%s, sentiment=%s",
            market, len(symbols), start_date, end_date,
            include_fundamental, include_sentiment,
        )

        # Step 1: Alpha158 technical features (synchronous, needs thread pool)
        loop = asyncio.get_running_loop()
        try:
            alpha_df = await loop.run_in_executor(
                _qlib_executor,
                self._get_alpha158_sync,
                market, symbols, start_date, end_date,
            )
        except Exception as e:
            logger.error("Alpha158 feature extraction failed: %s", e)
            alpha_df = pd.DataFrame()

        if alpha_df.empty:
            logger.warning(
                "Alpha158 returned empty DataFrame; cannot build feature matrix"
            )
            return pd.DataFrame()

        logger.info(
            "Alpha158 features: %d rows x %d columns",
            len(alpha_df), len(alpha_df.columns) - 2,
        )

        # Step 2: RD-Agent discovered factors (Qlib expressions, synchronous)
        rdagent_df = pd.DataFrame()
        try:
            from app.services.factor_registry import factor_registry
            active_factors = await factor_registry.get_active_factors(market)
            if active_factors:
                rdagent_df = await loop.run_in_executor(
                    _qlib_executor,
                    self._get_rdagent_factors_sync,
                    market, symbols, start_date, end_date, active_factors,
                )
                if not rdagent_df.empty:
                    logger.info(
                        "RD-Agent factors: %d rows x %d columns",
                        len(rdagent_df), len(rdagent_df.columns) - 2,
                    )
        except Exception as e:
            logger.warning("RD-Agent factor retrieval failed: %s", e)

        # Step 3 & 4: Fetch auxiliary features via StockPulse API (parallel)
        tasks: list[asyncio.Task] = []
        task_names: list[str] = []

        fundamental_df = pd.DataFrame()
        sentiment_df = pd.DataFrame()
        earnings_df = pd.DataFrame()
        analyst_df = pd.DataFrame()
        insider_df = pd.DataFrame()
        options_df = pd.DataFrame()
        short_interest_df = pd.DataFrame()
        earnings_calendar_df = pd.DataFrame()

        if include_fundamental:
            tasks.append(
                asyncio.create_task(
                    self._safe_get_fundamentals(symbols, start_date, end_date, market),
                )
            )
            task_names.append("fundamentals")

        # Sentiment: StockPulse endpoint deprecated; skip to avoid empty NaN columns.
        # Future: source from NewsForge integration.

        if EARNINGS_FEATURES:
            tasks.append(asyncio.create_task(
                self._safe_get_earnings(symbols, start_date, end_date, market),
            ))
            task_names.append("earnings")

        tasks.append(asyncio.create_task(
            self._safe_get_analyst(symbols, start_date, end_date, market),
        ))
        task_names.append("analyst")

        tasks.append(asyncio.create_task(
            self._safe_get_insider(symbols, start_date, end_date, market),
        ))
        task_names.append("insider")

        tasks.append(asyncio.create_task(
            self._safe_get_options(symbols, market, start_date, end_date),
        ))
        task_names.append("options")

        # Short interest (US market only)
        if market == "us":
            tasks.append(asyncio.create_task(
                self._safe_get_short_interest(symbols, start_date, end_date),
            ))
            task_names.append("short_interest")

        # Earnings calendar
        tasks.append(asyncio.create_task(
            self._safe_get_earnings_calendar(symbols, start_date, end_date),
        ))
        task_names.append("earnings_calendar")

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            failed_sources: list[str] = []
            for name, res in zip(task_names, results):
                if isinstance(res, pd.DataFrame):
                    if name == "fundamentals":
                        fundamental_df = res
                    elif name == "sentiment":
                        sentiment_df = res
                    elif name == "earnings":
                        earnings_df = res
                    elif name == "analyst":
                        analyst_df = res
                    elif name == "insider":
                        insider_df = res
                    elif name == "options":
                        options_df = res
                    elif name == "short_interest":
                        short_interest_df = res
                    elif name == "earnings_calendar":
                        earnings_calendar_df = res
                elif isinstance(res, Exception):
                    failed_sources.append(name)
                    logger.warning("%s feature fetch failed: %s", name, res)

            if failed_sources:
                logger.error(
                    "Feature sources failed (%d/%d): %s. "
                    "Feature matrix may be degraded.",
                    len(failed_sources), len(task_names), failed_sources,
                )

        # Step 4: Merge -- Alpha158 as the base, left-join everything else
        merged = alpha_df.copy()
        merged["date"] = pd.to_datetime(merged["date"])

        if not fundamental_df.empty:
            fundamental_df["date"] = pd.to_datetime(fundamental_df["date"].astype(str).str[:10], errors="coerce")
            fund_feature_cols = [
                c for c in FUNDAMENTAL_FEATURES if c in fundamental_df.columns
            ]
            fund_cols = ["symbol", "date"] + fund_feature_cols
            # Fundamentals are quarterly — merge then forward-fill so each
            # trading day inherits the latest available fundamental values.
            merged = merged.merge(
                fundamental_df[fund_cols],
                on=["symbol", "date"],
                how="left",
            )
            if fund_feature_cols:
                merged = merged.sort_values(["symbol", "date"])
                merged[fund_feature_cols] = (
                    merged.groupby("symbol")[fund_feature_cols]
                    .ffill()
                )
                merged = merged.reset_index(drop=True)
            logger.info(
                "Merged fundamental features: %d columns added (with forward-fill)",
                len(fund_cols) - 2,
            )

        if not sentiment_df.empty:
            sentiment_df["date"] = pd.to_datetime(sentiment_df["date"].astype(str).str[:10], errors="coerce")
            sent_cols = ["symbol", "date"] + [
                c for c in SENTIMENT_FEATURES if c in sentiment_df.columns
            ]
            merged = merged.merge(
                sentiment_df[sent_cols],
                on=["symbol", "date"],
                how="left",
            )
            logger.info(
                "Merged sentiment features: %d columns added",
                len(sent_cols) - 2,
            )

        if not earnings_df.empty:
            earnings_df["date"] = pd.to_datetime(earnings_df["date"].astype(str).str[:10], errors="coerce")
            earn_cols = ["symbol", "date"] + [
                c for c in EARNINGS_FEATURES if c in earnings_df.columns
            ]
            merged = merged.merge(earnings_df[earn_cols], on=["symbol", "date"], how="left")
            logger.info("Merged earnings features: %d columns added", len(earn_cols) - 2)

        if not analyst_df.empty:
            analyst_df["date"] = pd.to_datetime(analyst_df["date"].astype(str).str[:10], errors="coerce")
            ana_feature_cols = [
                c for c in ANALYST_FEATURES if c in analyst_df.columns
            ]
            ana_cols = ["symbol", "date"] + ana_feature_cols
            merged = merged.merge(analyst_df[ana_cols], on=["symbol", "date"], how="left")
            if ana_feature_cols:
                merged = merged.sort_values(["symbol", "date"])
                merged[ana_feature_cols] = merged.groupby("symbol")[ana_feature_cols].ffill()
                merged = merged.reset_index(drop=True)
            logger.info("Merged analyst features: %d columns added (with forward-fill)", len(ana_cols) - 2)

        if not insider_df.empty:
            insider_df["date"] = pd.to_datetime(insider_df["date"].astype(str).str[:10], errors="coerce")
            ins_feature_cols = [
                c for c in INSIDER_FEATURES if c in insider_df.columns
            ]
            ins_cols = ["symbol", "date"] + ins_feature_cols
            merged = merged.merge(insider_df[ins_cols], on=["symbol", "date"], how="left")
            if ins_feature_cols:
                merged = merged.sort_values(["symbol", "date"])
                merged[ins_feature_cols] = merged.groupby("symbol")[ins_feature_cols].ffill()
                merged = merged.reset_index(drop=True)
            logger.info("Merged insider features: %d columns added (with forward-fill)", len(ins_cols) - 2)

        if not options_df.empty:
            options_df["date"] = pd.to_datetime(options_df["date"].astype(str).str[:10], errors="coerce")
            opt_feature_cols = [
                c for c in OPTIONS_FEATURES if c in options_df.columns
            ]
            opt_cols = ["symbol", "date"] + opt_feature_cols
            merged = merged.merge(options_df[opt_cols], on=["symbol", "date"], how="left")
            if opt_feature_cols:
                merged = merged.sort_values(["symbol", "date"])
                merged[opt_feature_cols] = merged.groupby("symbol")[opt_feature_cols].ffill()
                merged = merged.reset_index(drop=True)
            logger.info("Merged options features: %d columns added (with forward-fill)", len(opt_cols) - 2)

        if not short_interest_df.empty:
            short_interest_df["date"] = pd.to_datetime(short_interest_df["date"].astype(str).str[:10], errors="coerce")
            si_cols = ["symbol", "date"] + [
                c for c in short_interest_df.columns
                if c not in ("symbol", "date")
            ]
            # Use fillna strategy: short_ratio/short_pct_float may already
            # exist from fundamentals -- prefer existing non-null values.
            si_new = short_interest_df[si_cols]
            overlap_cols = [
                c for c in si_new.columns
                if c in merged.columns and c not in ("symbol", "date")
            ]
            if overlap_cols:
                si_unique = [
                    c for c in si_new.columns
                    if c not in overlap_cols or c in ("symbol", "date")
                ]
                if len(si_unique) > 2:
                    merged = merged.merge(
                        si_new[si_unique], on=["symbol", "date"], how="left",
                    )
                # Fill NaN in existing columns from short interest data
                si_fill = si_new[["symbol", "date"] + overlap_cols]
                merged = merged.merge(
                    si_fill, on=["symbol", "date"], how="left", suffixes=("", "_si"),
                )
                for col in overlap_cols:
                    si_col = f"{col}_si"
                    if si_col in merged.columns:
                        merged[col] = merged[col].fillna(merged[si_col])
                        merged = merged.drop(columns=[si_col])
            else:
                merged = merged.merge(si_new, on=["symbol", "date"], how="left")
            si_feature_cols = [
                c for c in merged.columns
                if c in short_interest_df.columns and c not in ("symbol", "date")
            ]
            if si_feature_cols:
                merged = merged.sort_values(["symbol", "date"])
                merged[si_feature_cols] = merged.groupby("symbol")[si_feature_cols].ffill()
                merged = merged.reset_index(drop=True)
            logger.info(
                "Merged short interest features: %d columns (with forward-fill)",
                len(si_cols) - 2,
            )

        if not earnings_calendar_df.empty:
            earnings_calendar_df["date"] = pd.to_datetime(
                earnings_calendar_df["date"]
            )
            ec_cols = ["symbol", "date"] + [
                c for c in EARNINGS_CALENDAR_FEATURES
                if c in earnings_calendar_df.columns
            ]
            merged = merged.merge(
                earnings_calendar_df[ec_cols],
                on=["symbol", "date"],
                how="left",
            )
            logger.info(
                "Merged earnings calendar features: %d columns added",
                len(ec_cols) - 2,
            )

        if not rdagent_df.empty:
            rdagent_df["date"] = pd.to_datetime(rdagent_df["date"])
            rd_cols = ["symbol", "date"] + [
                c for c in rdagent_df.columns if c not in ("symbol", "date")
            ]
            merged = merged.merge(
                rdagent_df[rd_cols],
                on=["symbol", "date"],
                how="left",
            )
            logger.info(
                "Merged RD-Agent discovered factors: %d columns added",
                len(rd_cols) - 2,
            )

        # Ensure all feature columns are float64
        feature_cols = [c for c in merged.columns if c not in ("symbol", "date")]
        for col in feature_cols:
            merged[col] = pd.to_numeric(merged[col], errors="coerce").astype("float64")

        # Step 5: Feature interactions (before rank transform)
        cfg = config_override or get_market_config(market)
        if cfg.use_interactions:
            merged = self._compute_interaction_features(merged)
        else:
            logger.info(
                "Skipping interaction features for market=%s "
                "(MarketConfig.use_interactions=False)",
                market,
            )

        # Step 5.5: Drop sparse features
        merged, dropped = self._drop_sparse_features(merged, max_nan_ratio=cfg.nan_threshold)

        # Step 6: Rank transform (cross-sectional percentile per date)
        if cfg.use_sector_rank:
            client = await get_stockpulse_async_client()
            sector_map = await client.get_sector_map(market, symbols=symbols)
            logger.info(
                "Sector-adjusted feature ranking enabled for market=%s", market,
            )
        else:
            sector_map = {}
            logger.info(
                "Sector-adjusted feature ranking disabled for market=%s "
                "(MarketConfig.use_sector_rank=False)",
                market,
            )
        merged = self._rank_transform(merged, sector_map=sector_map or None)

        remaining = len([c for c in merged.columns if c not in ("symbol", "date")])
        logger.info(
            "Feature matrix built: %d rows x %d feature columns (dropped %d sparse)",
            len(merged), remaining, dropped,
        )
        return merged

    # ------------------------------------------------------------------
    # Alpha158 (synchronous, runs in ThreadPoolExecutor)
    # ------------------------------------------------------------------

    def _get_alpha158_sync(
        self,
        market: str,
        symbols: list[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Compute Alpha158 features via Qlib D.features().

        Synchronous method designed for executor use. Initializes Qlib
        for the target market and queries D.features() with the 65
        single-stock expression list.

        Returns:
            Flat DataFrame with columns: symbol, date, + 65 feature names.
        """
        from app.context import QlibContext
        from app.services.factor_service import SINGLE_STOCK_FEATURES
        from app.utils.symbol_mapping import (
            normalize_symbol_for_qlib,
            qlib_to_stockpulse,
        )

        settings = get_settings()

        try:
            QlibContext.ensure_init(market, settings.QLIB_DATA_DIR)
        except Exception as e:
            logger.error("Qlib init failed for market=%s: %s", market, e)
            return pd.DataFrame()

        from qlib.data import D

        qlib_symbols = [normalize_symbol_for_qlib(s, market) for s in symbols]
        qlib_to_ws = {}
        for ws_sym, q_sym in zip(symbols, qlib_symbols):
            qlib_to_ws[q_sym] = ws_sym

        logger.info(
            "D.features() call: %d symbols, %s~%s, %d features",
            len(qlib_symbols), start_date, end_date, len(SINGLE_STOCK_FEATURES),
        )

        try:
            df = D.features(
                instruments=qlib_symbols,
                fields=SINGLE_STOCK_FEATURES,
                start_time=start_date,
                end_time=end_date,
            )
        except Exception as e:
            logger.error("D.features() failed: %s", e)
            return pd.DataFrame()

        if df.empty:
            logger.warning("D.features() returned empty DataFrame")
            return pd.DataFrame()

        df.columns = FEATURE_NAMES[: len(df.columns)]

        if hasattr(df.index, "levels") and len(df.index.levels) == 2:
            df = df.reset_index()
            df.columns = ["qlib_symbol", "date"] + list(df.columns[2:])
            df["symbol"] = df["qlib_symbol"].map(
                lambda s: qlib_to_ws.get(s, qlib_to_stockpulse(s, market))
            )
            df = df.drop(columns=["qlib_symbol"])
        else:
            df = df.reset_index()
            df.columns = ["date"] + list(df.columns[1:])
            df["symbol"] = symbols[0] if len(symbols) == 1 else "UNKNOWN"

        df["date"] = pd.to_datetime(df["date"])
        feature_cols = [c for c in df.columns if c not in ("symbol", "date")]
        df = df[["symbol", "date"] + feature_cols]

        logger.info(
            "Alpha158 extraction complete: %d rows, %d symbols",
            len(df), df["symbol"].nunique(),
        )
        return df

    # ------------------------------------------------------------------
    # RD-Agent discovered factors (synchronous, runs in ThreadPoolExecutor)
    # ------------------------------------------------------------------

    def _get_rdagent_factors_sync(
        self,
        market: str,
        symbols: list[str],
        start_date: str,
        end_date: str,
        active_factors: list[dict],
    ) -> pd.DataFrame:
        """Compute RD-Agent discovered factor features via Qlib D.features()."""
        from app.context import QlibContext
        from app.utils.symbol_mapping import (
            normalize_symbol_for_qlib,
            qlib_to_stockpulse,
        )

        settings = get_settings()
        try:
            QlibContext.ensure_init(market, settings.QLIB_DATA_DIR)
        except Exception as e:
            logger.error("Qlib init failed for RD-Agent factors: %s", e)
            return pd.DataFrame()

        from qlib.data import D

        expressions = [f["expression"] for f in active_factors]
        factor_names = [f["name"] for f in active_factors]

        qlib_symbols = [normalize_symbol_for_qlib(s, market) for s in symbols]
        qlib_to_ws = {q: w for w, q in zip(symbols, qlib_symbols)}

        logger.info(
            "D.features() for %d RD-Agent factors, %d symbols",
            len(expressions), len(qlib_symbols),
        )

        try:
            df = D.features(
                instruments=qlib_symbols,
                fields=expressions,
                start_time=start_date,
                end_time=end_date,
            )
        except Exception as e:
            logger.error("D.features() failed for RD-Agent factors: %s", e)
            return pd.DataFrame()

        if df.empty:
            return pd.DataFrame()

        df.columns = factor_names[: len(df.columns)]

        if hasattr(df.index, "levels") and len(df.index.levels) == 2:
            df = df.reset_index()
            df.columns = ["qlib_symbol", "date"] + list(df.columns[2:])
            df["symbol"] = df["qlib_symbol"].map(
                lambda s: qlib_to_ws.get(s, qlib_to_stockpulse(s, market))
            )
            df = df.drop(columns=["qlib_symbol"])
        else:
            df = df.reset_index()
            df.columns = ["date"] + list(df.columns[1:])
            df["symbol"] = symbols[0] if len(symbols) == 1 else "UNKNOWN"

        df["date"] = pd.to_datetime(df["date"])
        feat_cols = [c for c in df.columns if c not in ("symbol", "date")]
        return df[["symbol", "date"] + feat_cols]

    # ------------------------------------------------------------------
    # Safe wrappers (fetch from StockPulse API with error isolation)
    # ------------------------------------------------------------------

    @staticmethod
    async def _safe_get_fundamentals(
        symbols: list[str], start_date: str, end_date: str, market: str,
    ) -> pd.DataFrame:
        """Fetch fundamentals via StockPulse API with error isolation.

        Requests an extra 400 days before start_date so derived YoY
        features (revenue_growth_yoy, eps_growth) have prior-year
        comparisons. Batches symbols to avoid overloading StockPulse.
        """
        try:
            from datetime import date as _date, timedelta
            extended_start = (
                _date.fromisoformat(start_date) - timedelta(days=400)
            ).isoformat()
            client = await get_stockpulse_async_client()
            all_data: list[dict] = []
            batch_size = 200
            for i in range(0, len(symbols), batch_size):
                batch = symbols[i : i + batch_size]
                rows = await client.get_fundamentals_batch(
                    batch, market, extended_start, end_date,
                )
                all_data.extend(rows)
            if not all_data:
                return pd.DataFrame()
            return pd.DataFrame(all_data)
        except Exception as e:
            logger.warning("Fundamental feature retrieval failed: %s", e)
            return pd.DataFrame()

    @staticmethod
    async def _safe_get_sentiment(
        symbols: list[str], start_date: str, end_date: str, market: str,
    ) -> pd.DataFrame:
        """Fetch sentiment features via StockPulse API with error isolation."""
        try:
            client = await get_stockpulse_async_client()
            data = await client.get_sentiment_batch(symbols, market, start_date, end_date)
            if not data:
                return pd.DataFrame()
            return pd.DataFrame(data)
        except Exception as e:
            logger.warning("Sentiment feature retrieval failed: %s", e)
            return pd.DataFrame()

    @staticmethod
    async def _safe_get_earnings(
        symbols: list[str], start_date: str, end_date: str, market: str,
    ) -> pd.DataFrame:
        """Fetch EPS surprise features via StockPulse API with error isolation."""
        try:
            client = await get_stockpulse_async_client()
            data = await client.get_earnings_batch(symbols, market, start_date, end_date)
            if not data:
                return pd.DataFrame()
            return pd.DataFrame(data)
        except Exception as e:
            logger.warning("Earnings feature retrieval failed: %s", e)
            return pd.DataFrame()

    @staticmethod
    async def _safe_get_analyst(
        symbols: list[str], start_date: str, end_date: str, market: str,
    ) -> pd.DataFrame:
        """Fetch analyst snapshot features via StockPulse API with error isolation."""
        try:
            client = await get_stockpulse_async_client()
            data = await client.get_analyst_batch(symbols, market, start_date, end_date)
            if not data:
                return pd.DataFrame()
            return pd.DataFrame(data)
        except Exception as e:
            logger.warning("Analyst feature retrieval failed: %s", e)
            return pd.DataFrame()

    @staticmethod
    async def _safe_get_insider(
        symbols: list[str], start_date: str, end_date: str, market: str,
    ) -> pd.DataFrame:
        """Fetch insider activity features via StockPulse API with error isolation."""
        try:
            client = await get_stockpulse_async_client()
            data = await client.get_insider_batch(symbols, market, start_date, end_date)
            if not data:
                return pd.DataFrame()
            return pd.DataFrame(data)
        except Exception as e:
            logger.warning("Insider feature retrieval failed: %s", e)
            return pd.DataFrame()

    @staticmethod
    async def _safe_get_options(
        symbols: list[str], market: str, start_date: str, end_date: str,
    ) -> pd.DataFrame:
        """Fetch options put/call ratio features via StockPulse API with error isolation."""
        try:
            client = await get_stockpulse_async_client()
            data = await client.get_options_batch(symbols, market, start_date, end_date)
            if not data:
                return pd.DataFrame()
            return pd.DataFrame(data)
        except Exception as e:
            logger.warning("Options feature retrieval failed: %s", e)
            return pd.DataFrame()

    @staticmethod
    async def _safe_get_short_interest(
        symbols: list[str], start_date: str, end_date: str,
    ) -> pd.DataFrame:
        """Fetch short interest data via StockPulse API (US market only).

        Returns DataFrame with columns: symbol, date, short_ratio,
        short_pct_float (among others). Error-isolated.
        """
        try:
            client = await get_stockpulse_async_client()
            data = await client.get_short_interest_batch(
                symbols, start_date, end_date,
            )
            if not data:
                return pd.DataFrame()
            return pd.DataFrame(data)
        except Exception as e:
            logger.warning("Short interest feature retrieval failed: %s", e)
            return pd.DataFrame()

    @staticmethod
    async def _safe_get_earnings_calendar(
        symbols: list[str], start_date: str, end_date: str,
    ) -> pd.DataFrame:
        """Fetch earnings calendar and compute days_to_earnings feature.

        For each (symbol, date) pair, ``days_to_earnings`` is:
        - positive: number of days until the next earnings date
        - negative: number of days since the last earnings date

        Error-isolated: returns empty DataFrame on failure.
        """
        try:
            client = await get_stockpulse_async_client()
            calendar = await client.get_earnings_calendar(start_date, end_date)
            if not calendar:
                return pd.DataFrame()

            import numpy as np

            # Build a map of symbol -> sorted list of earnings dates
            from collections import defaultdict

            earnings_map: dict[str, list[pd.Timestamp]] = defaultdict(list)
            for row in calendar:
                sym = row.get("symbol")
                edate = row.get("earnings_date")
                if sym and edate:
                    earnings_map[sym].append(pd.Timestamp(edate))

            for sym in earnings_map:
                earnings_map[sym].sort()

            # Filter to requested symbols
            target_symbols = set(symbols) & set(earnings_map.keys())
            if not target_symbols:
                return pd.DataFrame()

            # Generate (symbol, date) grid for all trading dates in range
            date_range = pd.bdate_range(start=start_date, end=end_date)
            records = []
            for sym in target_symbols:
                edates = earnings_map[sym]
                for dt in date_range:
                    # Find closest earnings date (next or previous)
                    diffs = [(ed - dt).days for ed in edates]
                    future = [d for d in diffs if d >= 0]
                    past = [d for d in diffs if d < 0]

                    if future:
                        days_val = min(future)  # days until next
                    elif past:
                        days_val = max(past)  # days since last (negative)
                    else:
                        continue

                    records.append({
                        "symbol": sym,
                        "date": dt,
                        "days_to_earnings": days_val,
                    })

            if not records:
                return pd.DataFrame()

            return pd.DataFrame(records)
        except Exception as e:
            logger.warning("Earnings calendar feature retrieval failed: %s", e)
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # Feature interactions
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
        """Compute cross-feature interaction columns from raw merged values.

        Called after all source features are merged but before rank transform,
        so interactions use raw (unranked) values -- preserving multiplicative
        and ratio relationships that tree models struggle to learn via
        axis-aligned splits.

        Uses pd.concat to avoid DataFrame fragmentation warnings.
        """
        import numpy as np

        cols = set(df.columns)
        _EPS = 1e-8
        new_cols: dict[str, "pd.Series"] = {}

        # --- Pure technical interactions (Alpha158 x Alpha158) ---

        if {"ret20", "std20"} <= cols:
            new_cols["momentum_vol_ratio"] = df["ret20"] / (df["std20"].abs() + _EPS)

        if {"vol_ratio5", "ret5"} <= cols:
            new_cols["volume_price_confirm"] = df["vol_ratio5"] * df["ret5"].abs()

        if {"ret5", "ret60"} <= cols:
            new_cols["momentum_divergence"] = df["ret5"] - df["ret60"]

        if {"drawdown20", "vol_ratio5"} <= cols:
            new_cols["drawdown_recovery"] = df["drawdown20"] * df["vol_ratio5"]

        if {"rsi14", "std20"} <= cols:
            new_cols["trend_vol_interaction"] = df["rsi14"] * df["std20"]

        if {"close_ma20_ratio", "vol_ratio5"} <= cols:
            new_cols["price_ma_volume"] = df["close_ma20_ratio"] * df["vol_ratio5"]

        if {"return_vol20", "return_vol60"} <= cols:
            new_cols["volatility_acceleration"] = df["return_vol20"] - df["return_vol60"]

        if {"ret5", "ret20"} <= cols:
            new_cols["momentum_acceleration"] = df["ret5"] - df["ret20"]

        # --- Cross-category interactions (fundamental x technical) ---

        if {"pb_ratio", "ret20"} <= cols:
            new_cols["value_momentum"] = df["pb_ratio"] * df["ret20"]

        if {"dividend_yield", "std20"} <= cols:
            new_cols["yield_vol_adj"] = df["dividend_yield"] / (df["std20"].abs() + _EPS)

        if new_cols:
            import pandas as _pd
            df = _pd.concat([df, _pd.DataFrame(new_cols, index=df.index)], axis=1)
            logger.info("Computed %d interaction features", len(new_cols))

        return df

    # ------------------------------------------------------------------
    # Feature quality control
    # ------------------------------------------------------------------

    @staticmethod
    def _drop_sparse_features(
        df: pd.DataFrame, max_nan_ratio: float = 0.90,
    ) -> tuple[pd.DataFrame, int]:
        """Drop feature columns with excessively high NaN rates."""
        feature_cols = [c for c in df.columns if c not in ("symbol", "date")]
        if not feature_cols:
            return df, 0

        nan_ratios = df[feature_cols].isna().mean()
        sparse_cols = nan_ratios[nan_ratios > max_nan_ratio].index.tolist()

        if sparse_cols:
            logger.warning(
                "Dropping %d sparse features (>%.0f%% NaN): %s",
                len(sparse_cols),
                max_nan_ratio * 100,
                sparse_cols[:10],
            )
            df = df.drop(columns=sparse_cols)

        return df, len(sparse_cols)

    # ------------------------------------------------------------------
    # Rank transform
    # ------------------------------------------------------------------

    @staticmethod
    def _cross_sectional_normalize(df: pd.DataFrame) -> pd.DataFrame:
        """Apply cross-sectional MAD-based robust normalization per date.

        Preserved for reference but NOT used for lambdarank training
        (rank transform gives better IC: 0.017 vs 0.006).
        """
        import numpy as np

        feature_cols = [c for c in df.columns if c not in ("symbol", "date")]
        if not feature_cols:
            return df

        result = df.copy()

        for col in feature_cols:
            def _robust_norm(x: "pd.Series") -> "pd.Series":
                median = x.median()
                mad = (x - median).abs().median()
                scale = mad * 1.4826
                if scale < 1e-10:
                    return x - median
                normed = (x - median) / scale
                return normed.clip(-3.0, 3.0)

            result[col] = result.groupby("date")[col].transform(_robust_norm)

        return result

    @staticmethod
    def _rank_transform(
        df: pd.DataFrame,
        sector_map: dict[str, str] | None = None,
    ) -> pd.DataFrame:
        """Apply cross-sectional percentile ranking per date.

        Preferred over _cross_sectional_normalize() for lambdarank:
        uniform [0,1] distribution maximizes split information per tree,
        and rank inputs naturally match the ranking objective.

        If sector_map is provided with sufficient coverage (>=30%),
        valuation features are ranked within sectors instead of
        cross-sectionally.
        """
        feature_cols = [c for c in df.columns if c not in ("symbol", "date")]
        if not feature_cols:
            return df

        _SECTOR_RANK_FEATURES = {
            "pe_ratio", "pb_ratio", "ps_ratio", "forward_pe",
            "ev_ebitda", "dividend_yield", "dividend_rate",
            "roe", "roa", "profit_margin", "gross_margin",
            "operating_margin", "payout_ratio", "debt_to_equity",
            "current_ratio", "eps_growth", "revenue_growth_yoy",
        }

        use_sector_rank = False
        if sector_map:
            df["_sector"] = df["symbol"].map(sector_map)
            coverage = df["_sector"].notna().mean()
            if coverage >= 0.3:
                use_sector_rank = True
                logger.info(
                    "Sector-adjusted ranking enabled for %d valuation features "
                    "(%.0f%% sector coverage)",
                    len(_SECTOR_RANK_FEATURES & set(feature_cols)),
                    coverage * 100,
                )
            else:
                logger.info(
                    "Sector coverage too low (%.0f%%) for feature ranking, "
                    "using cross-sectional",
                    coverage * 100,
                )

        result = df.copy()
        for col in feature_cols:
            if use_sector_rank and col in _SECTOR_RANK_FEATURES:
                has_sector = result["_sector"].notna()
                result.loc[has_sector, col] = (
                    result.loc[has_sector]
                    .groupby(["date", "_sector"])[col]
                    .rank(pct=True)
                )
                if (~has_sector).any():
                    result.loc[~has_sector, col] = (
                        result.loc[~has_sector]
                        .groupby("date")[col]
                        .rank(pct=True)
                    )
            else:
                result[col] = result.groupby("date")[col].rank(pct=True)

        if "_sector" in result.columns:
            result = result.drop(columns=["_sector"])

        return result

    # ------------------------------------------------------------------
    # Feature name helpers
    # ------------------------------------------------------------------

    def get_feature_names(
        self,
        include_fundamental: bool = True,
        include_sentiment: bool = True,
    ) -> list[str]:
        """Return the full list of feature column names."""
        names = list(ALPHA158_FEATURES)
        if include_fundamental:
            names.extend(FUNDAMENTAL_FEATURES)
        if include_sentiment:
            names.extend(SENTIMENT_FEATURES)
        return names

    def get_feature_count(
        self,
        include_fundamental: bool = True,
        include_sentiment: bool = True,
    ) -> int:
        """Return total number of features for the given configuration."""
        return len(self.get_feature_names(include_fundamental, include_sentiment))

    # ------------------------------------------------------------------
    # Feature drift detection (PSI)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_feature_psi(
        train_df: pd.DataFrame,
        inference_df: pd.DataFrame,
        feature_cols: list[str],
        bins: int = 10,
    ) -> dict[str, float]:
        """Compute Population Stability Index (PSI) for each feature.

        PSI measures distribution shift between training and inference data.
        Values:
        - < 0.1: insignificant change
        - 0.1-0.2: moderate change, monitor
        - > 0.2: significant change, model may be unreliable
        """
        import numpy as np

        psi_scores: dict[str, float] = {}
        _EPS = 1e-6

        for col in feature_cols:
            if col not in train_df.columns or col not in inference_df.columns:
                continue

            train_vals = train_df[col].dropna().values
            infer_vals = inference_df[col].dropna().values

            if len(train_vals) < bins * 2 or len(infer_vals) < bins * 2:
                continue

            bin_edges = np.percentile(train_vals, np.linspace(0, 100, bins + 1))
            bin_edges[0] = -np.inf
            bin_edges[-1] = np.inf

            train_hist = np.histogram(train_vals, bins=bin_edges)[0]
            infer_hist = np.histogram(infer_vals, bins=bin_edges)[0]

            train_prop = train_hist / len(train_vals) + _EPS
            infer_prop = infer_hist / len(infer_vals) + _EPS

            psi = float(np.sum(
                (infer_prop - train_prop) * np.log(infer_prop / train_prop)
            ))
            psi_scores[col] = round(psi, 6)

        return psi_scores


# Module singleton
feature_service = FeatureService()
