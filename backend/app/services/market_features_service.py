"""Market-level features for direction prediction.

Builds 7+ features that are identical for all stocks on a given trading date:

    index_return_5d   - 5-day rolling return of the market benchmark index
    index_return_10d  - 10-day rolling return
    index_return_20d  - 20-day rolling return
    market_breadth    - fraction of stocks with positive daily return
    market_volatility - 20-day rolling std-dev of daily index returns
    sector_momentum   - mean 5-day return of top-3 performing sectors
    volume_ma10       - 10-day MA of market-wide mean daily volume
    up_ratio_5d       - fraction of recent 5 days with positive index return
    up_ratio_20d      - fraction of recent 20 days with positive index return
    breadth_momentum_5d - 5-day change in market breadth

AlphaForge adaptation:
- All data fetched via StockPulseAsyncClient HTTP API (no direct DB queries).
- Index prices, breadth, volume, sector returns all come from StockPulse.
- Graceful degradation: each feature group is independent, NaN on failure.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from app.services.market_config import get_market_config
from app.services.stockpulse_client import get_stockpulse_async_client

logger = logging.getLogger(__name__)

# Feature column names exported for use by feature_service / prediction_service
MARKET_FEATURE_COLUMNS: list[str] = [
    "index_return_5d",
    "index_return_10d",
    "index_return_20d",
    "market_breadth",
    "market_volatility",
    "sector_momentum_mean",
    "volume_ma10",
    # Trend/regime indicators for direction prediction
    "up_ratio_5d",
    "up_ratio_20d",
    "breadth_momentum_5d",
    # Macro indicators
    "vix_level",
    "vix_change_5d",
    "dxy_level",
    "dxy_change_5d",
    "tnx_level",
    "tnx_change_5d",
]

# Extra calendar-day lookback to ensure rolling windows have enough history.
_LOOKBACK_BUFFER_DAYS = 60


@dataclass
class MarketFeatures:
    """Market-level features for a single date."""

    date: date
    index_return_5d: Optional[float] = None
    index_return_10d: Optional[float] = None
    index_return_20d: Optional[float] = None
    market_breadth: Optional[float] = None
    market_volatility: Optional[float] = None
    sector_momentum_mean: Optional[float] = None
    volume_ma10: Optional[float] = None
    up_ratio_5d: Optional[float] = None
    up_ratio_20d: Optional[float] = None
    breadth_momentum_5d: Optional[float] = None
    vix_level: Optional[float] = None
    vix_change_5d: Optional[float] = None
    dxy_level: Optional[float] = None
    dxy_change_5d: Optional[float] = None
    tnx_level: Optional[float] = None
    tnx_change_5d: Optional[float] = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def build_market_features(
    market: str,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """Build market-level features for the given date range.

    Returns a DataFrame with columns ``[date] + MARKET_FEATURE_COLUMNS``.
    All feature values are z-score normalised (zero mean, unit variance).
    Dates without sufficient data have NaN for the affected features.

    Args:
        market: Market code (``us``, ``cn``, ``hk``).
        start_date: First date of the requested output range.
        end_date: Last date of the requested output range.

    Returns:
        DataFrame ready for left-join onto per-stock features on ``date``.
        Empty DataFrame (with correct columns) on total failure.
    """
    market = market.lower()
    cfg = get_market_config(market)

    # Extend lookback to fill rolling windows at the start of the range.
    fetch_start = start_date - timedelta(days=_LOOKBACK_BUFFER_DAYS)

    logger.info(
        "Building market features: market=%s, range=%s~%s, index=%s, "
        "fetch_start=%s (buffer=%d days)",
        market, start_date, end_date, cfg.index_symbol,
        fetch_start, _LOOKBACK_BUFFER_DAYS,
    )

    # Fetch all data sources (sequential to avoid overwhelming StockPulse)
    index_df = await _fetch_index_prices(cfg.index_symbol, market, fetch_start, end_date)
    breadth_df = await _fetch_breadth(market, fetch_start, end_date)
    volume_df = await _fetch_volume(market, fetch_start, end_date)
    sector_df = await _fetch_sector_momentum(market, fetch_start, end_date)
    macro_df = await _fetch_macro_indicators(fetch_start, end_date)

    # ---- Compute rolling features ----

    result_frames: list[pd.DataFrame] = []

    # 1. Index-based features (returns + volatility)
    idx_features = _compute_index_features(index_df)
    if idx_features is not None:
        result_frames.append(idx_features)

    # 2. Market breadth
    if breadth_df is not None:
        result_frames.append(breadth_df)

    # 3. Volume MA
    vol_features = _compute_volume_ma(volume_df)
    if vol_features is not None:
        result_frames.append(vol_features)

    # 4. Sector momentum
    if sector_df is not None:
        result_frames.append(sector_df)

    # 5. Macro indicators
    if macro_df is not None:
        result_frames.append(macro_df)

    # ---- Merge all feature groups ----

    if not result_frames:
        logger.warning(
            "No market features could be computed for market=%s", market,
        )
        return _empty_result()

    merged = result_frames[0]
    for frame in result_frames[1:]:
        merged = merged.merge(frame, on="date", how="outer")

    # Breadth momentum: 5-day change in market breadth.
    if "market_breadth" in merged.columns:
        merged = merged.sort_values("date")
        merged["breadth_momentum_5d"] = merged["market_breadth"].diff(5)

    # Trim to requested date range
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    merged = merged[(merged["date"] >= start_ts) & (merged["date"] <= end_ts)]

    if merged.empty:
        logger.warning(
            "Market features empty after date trim for market=%s", market,
        )
        return _empty_result()

    # Ensure all expected columns exist
    for col in MARKET_FEATURE_COLUMNS:
        if col not in merged.columns:
            merged[col] = np.nan

    # ---- Z-score normalisation ----
    for col in MARKET_FEATURE_COLUMNS:
        series = merged[col].astype("float64")
        non_null = series.notna().sum()
        if non_null == 0:
            continue
        mean = series.mean()
        std = series.std()
        if std > 0:
            merged[col] = (series - mean) / std
        else:
            merged[col] = 0.0

    merged = merged[["date"] + MARKET_FEATURE_COLUMNS].reset_index(drop=True)

    logger.info(
        "Market features built: market=%s, %d dates, non-null counts: %s",
        market,
        len(merged),
        {col: int(merged[col].notna().sum()) for col in MARKET_FEATURE_COLUMNS},
    )

    return merged


# ---------------------------------------------------------------------------
# Data fetchers (via StockPulse API)
# ---------------------------------------------------------------------------


async def _fetch_index_prices(
    index_symbol: str,
    market: str,
    start_date: date,
    end_date: date,
) -> Optional[pd.DataFrame]:
    """Fetch daily close prices for the benchmark index via StockPulse.

    Returns a DataFrame with columns ``[date, close]`` sorted by date,
    or None on failure.
    """
    try:
        client = await get_stockpulse_async_client()
        bars = await client.get_index_history(
            symbol=index_symbol,
            market=market,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
        )
    except Exception as e:
        logger.warning(
            "Failed to fetch index prices for %s (market=%s): %s",
            index_symbol, market, e,
        )
        return None

    if not bars:
        logger.warning(
            "No index data returned for %s (market=%s)", index_symbol, market,
        )
        return None

    df = pd.DataFrame(bars)
    if "date" not in df.columns or "close" not in df.columns:
        logger.warning(
            "Incomplete index data for %s: columns=%s",
            index_symbol, list(df.columns),
        )
        return None

    # Extract just the date part (tz-naive)
    df["date"] = pd.to_datetime(df["date"].astype(str).str[:10])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df[["date", "close"]].dropna(subset=["close"]).sort_values("date").reset_index(drop=True)

    logger.info(
        "Index prices fetched: %s, %d bars, %s ~ %s",
        index_symbol, len(df),
        df["date"].min().date() if len(df) > 0 else "N/A",
        df["date"].max().date() if len(df) > 0 else "N/A",
    )

    return df


async def _fetch_breadth(
    market: str,
    start_date: date,
    end_date: date,
) -> Optional[pd.DataFrame]:
    """Fetch market breadth data from StockPulse.

    Returns a DataFrame with columns ``[date, market_breadth]``.
    """
    try:
        client = await get_stockpulse_async_client()
        data = await client.get_market_breadth(
            market, start_date.isoformat(), end_date.isoformat(),
        )
    except Exception as e:
        logger.warning("Market breadth fetch failed for market=%s: %s", market, e)
        return None

    if not data:
        logger.warning("No breadth data for market=%s", market)
        return None

    records = []
    for row in data:
        total = row.get("total_issues", 0)
        if total and int(total) > 0:
            records.append({
                "date": pd.Timestamp(row["date"]),
                "market_breadth": float(row.get("advancers", 0)) / float(total),
            })

    if not records:
        return None

    df = pd.DataFrame(records)
    logger.info("Market breadth computed: market=%s, %d dates", market, len(df))
    return df


async def _fetch_volume(
    market: str,
    start_date: date,
    end_date: date,
) -> Optional[pd.DataFrame]:
    """Fetch market-wide average daily volume from StockPulse.

    Returns a DataFrame with columns ``[date, avg_volume]``.
    """
    try:
        client = await get_stockpulse_async_client()
        data = await client.get_market_volume(
            market, start_date.isoformat(), end_date.isoformat(),
        )
    except Exception as e:
        logger.warning("Volume fetch failed for market=%s: %s", market, e)
        return None

    if not data:
        logger.warning("No volume data for market=%s", market)
        return None

    df = pd.DataFrame([
        {"date": pd.Timestamp(row["date"]), "avg_volume": float(row.get("total_volume", 0))}
        for row in data
        if row.get("total_volume") is not None
    ])

    if df.empty:
        return None

    logger.info("Volume data fetched: market=%s, %d dates", market, len(df))
    return df


async def _fetch_sector_momentum(
    market: str,
    start_date: date,
    end_date: date,
) -> Optional[pd.DataFrame]:
    """Compute sector momentum from StockPulse sector return data.

    For each date, computes the mean 5-day return of the top-3 performing
    sectors. Returns a DataFrame with columns ``[date, sector_momentum_mean]``.
    """
    try:
        client = await get_stockpulse_async_client()
        data = await client.get_sector_returns(
            market, start_date.isoformat(), end_date.isoformat(),
        )
    except Exception as e:
        logger.warning(
            "Sector momentum fetch failed for market=%s: %s", market, e,
        )
        return None

    if not data:
        logger.info("No sector return data for market=%s", market)
        return None

    # StockPulse returns pre-aggregated per-sector data:
    #   {"date": "...", "sector": "Technology", "avg_return": "-0.0075", "num_stocks": 85}
    # Group by date, sort sectors by avg_return descending, take top 3.
    date_sectors: dict[pd.Timestamp, list[float]] = defaultdict(list)
    for row in data:
        ret = row.get("avg_return")
        if ret is not None:
            try:
                date_sectors[pd.Timestamp(str(row["date"])[:10])].append(float(ret))
            except (ValueError, TypeError, KeyError):
                continue

    records = []
    for dt in sorted(date_sectors.keys()):
        returns = date_sectors[dt]
        returns.sort(reverse=True)
        top3 = returns[:3]
        if top3:
            records.append({
                "date": dt,
                "sector_momentum_mean": sum(top3) / len(top3),
            })

    if not records:
        return None

    df = pd.DataFrame(records)
    logger.info("Sector momentum computed: market=%s, %d dates", market, len(df))
    return df


async def _fetch_macro_indicators(
    start_date: date,
    end_date: date,
) -> Optional[pd.DataFrame]:
    """Fetch macro indicators (VIX, DXY, TNX) from StockPulse and compute features.

    Returns a DataFrame with columns:
        ``[date, vix_level, vix_change_5d, dxy_level, dxy_change_5d,
          tnx_level, tnx_change_5d]``
    or None on failure.
    """
    try:
        client = await get_stockpulse_async_client()
        macro_data = await client.get_macro_batch(
            tickers=["^VIX", "DX-Y.NYB", "^TNX"],
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
        )
    except Exception as e:
        logger.warning("Macro indicator fetch failed: %s", e)
        return None

    frames: list[pd.DataFrame] = []
    ticker_col_map = {
        "^VIX": ("vix_level", "vix_change_5d"),
        "DX-Y.NYB": ("dxy_level", "dxy_change_5d"),
        "^TNX": ("tnx_level", "tnx_change_5d"),
    }

    for ticker, (level_col, change_col) in ticker_col_map.items():
        rows = macro_data.get(ticker, [])
        if not rows:
            continue

        df = pd.DataFrame(rows)
        if "date" not in df.columns:
            continue
        val_col = "close" if "close" in df.columns else "value"
        if val_col not in df.columns:
            continue

        df["date"] = pd.to_datetime(df["date"].astype(str).str[:10])
        df["_val"] = pd.to_numeric(df[val_col], errors="coerce")
        df = df.dropna(subset=["_val"]).sort_values("date").reset_index(drop=True)

        if df.empty:
            continue

        df[level_col] = df["_val"]
        df[change_col] = df["_val"].pct_change(periods=5)
        frames.append(df[["date", level_col, change_col]])

    if not frames:
        logger.warning("No macro indicator data available")
        return None

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="date", how="outer")

    merged = merged.sort_values("date").reset_index(drop=True)
    logger.info("Macro indicators computed: %d dates", len(merged))
    return merged


# ---------------------------------------------------------------------------
# Feature computation helpers
# ---------------------------------------------------------------------------


def _compute_index_features(index_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Compute index return and volatility features from daily close prices."""
    if index_df is None or len(index_df) < 2:
        logger.warning("Insufficient index data for feature computation")
        return None

    df = index_df.copy()
    df = df.sort_values("date").reset_index(drop=True)

    df["daily_return"] = df["close"].pct_change()

    for window in (5, 10, 20):
        col = f"index_return_{window}d"
        df[col] = df["close"].pct_change(periods=window)

    df["market_volatility"] = df["daily_return"].rolling(window=20, min_periods=10).std()

    daily_up = (df["daily_return"] > 0).astype(float)
    df["up_ratio_5d"] = daily_up.rolling(5, min_periods=3).mean()
    df["up_ratio_20d"] = daily_up.rolling(20, min_periods=10).mean()

    result = df[["date", "index_return_5d", "index_return_10d",
                 "index_return_20d", "market_volatility",
                 "up_ratio_5d", "up_ratio_20d"]].copy()

    non_null = result.dropna(how="all", subset=[
        "index_return_5d", "index_return_10d",
        "index_return_20d", "market_volatility",
        "up_ratio_5d", "up_ratio_20d",
    ])

    if non_null.empty:
        logger.warning("All index features are NaN after rolling computation")
        return None

    logger.info(
        "Index features computed: %d dates with data",
        len(non_null),
    )
    return result


def _compute_volume_ma(volume_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Compute 10-day moving average of market-wide mean volume."""
    if volume_df is None or volume_df.empty:
        return None

    df = volume_df.copy().sort_values("date").reset_index(drop=True)
    df["volume_ma10"] = df["avg_volume"].rolling(window=10, min_periods=5).mean()
    result = df[["date", "volume_ma10"]].dropna(subset=["volume_ma10"])

    if result.empty:
        logger.warning("Volume MA10 all NaN after rolling computation")
        return None

    logger.info("Volume MA10 computed: %d dates", len(result))
    return result


def _empty_result() -> pd.DataFrame:
    """Return an empty DataFrame with the correct column schema."""
    return pd.DataFrame(columns=["date"] + MARKET_FEATURE_COLUMNS)
