"""Technical indicator computation engine using pure pandas/numpy.

Computes 12 technical indicator types from OHLCV bar data without any
external TA library dependency. All calculations use standard pandas
rolling/ewm operations and numpy vectorized math.

Supported indicators:
    SMA, EMA, RSI, MACD, Bollinger Bands, ATR, OBV, KDJ, Williams %R,
    CCI, VWAP, Parabolic SAR.

Output format is identical to the backend's indicator_service.py so that
consumers (API routes, chat skills) can swap implementations transparently.
"""

import logging
import time
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _format_time(value: Any, intraday: bool = False) -> str:
    """Format a time value for indicator data points.

    Args:
        value: A datetime object, pd.Timestamp, or ISO datetime string.
        intraday: When True return ``YYYY-MM-DD HH:MM:SS``, otherwise
            return only the date portion ``YYYY-MM-DD``.

    Returns:
        Formatted time string.
    """
    if intraday:
        if isinstance(value, (datetime, pd.Timestamp)):
            return value.strftime("%Y-%m-%d %H:%M:%S")
        return str(value)

    # Daily mode: extract date portion only
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.strftime("%Y-%m-%d")
    s = str(value)
    # Extract YYYY-MM-DD from ISO datetime strings
    # e.g. "2025-02-11T00:00:00-05:00" -> "2025-02-11"
    if len(s) >= 10 and s[4] == "-" and s[7] == "-":
        return s[:10]
    return s


def _series_to_points(
    dates: pd.Index,
    values: pd.Series,
    intraday: bool = False,
) -> list[dict[str, Any]]:
    """Convert a pandas Series to a list of ``{time, value}`` dicts.

    NaN values are dropped and numeric values are rounded to 4 decimal
    places, matching the output contract of the backend indicator service.

    Args:
        dates: Index of date/datetime values aligned with *values*.
        values: Numeric series to serialize.
        intraday: Forwarded to :func:`_format_time`.

    Returns:
        List of data-point dicts suitable for JSON serialization.
    """
    points: list[dict[str, Any]] = []
    for date, val in zip(dates, values):
        if pd.notna(val):
            points.append({
                "time": _format_time(date, intraday=intraday),
                "value": round(float(val), 4),
            })
    return points


# ---------------------------------------------------------------------------
# Individual indicator computations
# ---------------------------------------------------------------------------

def _compute_sma(
    dates: pd.Index,
    close: pd.Series,
    period: int,
    num_bars: int,
    intraday: bool,
    result: dict[str, Any],
    warnings: list[str],
) -> None:
    """Simple Moving Average for a single period."""
    key = f"sma_{period}"
    if num_bars < period:
        warnings.append(f"SMA {period} needs {period} bars, only have {num_bars}")
        return
    try:
        series = close.rolling(window=period).mean()
        points = _series_to_points(dates, series, intraday=intraday)
        if points:
            result[key] = {
                "series": points,
                "metadata": {"period": period, "type": "sma"},
            }
        else:
            warnings.append(f"SMA {period} produced no valid data points")
    except Exception as e:
        logger.error("Error computing SMA %d: %s", period, e)
        warnings.append(f"SMA {period} computation failed: {e}")


def _compute_ema(
    dates: pd.Index,
    close: pd.Series,
    period: int,
    num_bars: int,
    intraday: bool,
    result: dict[str, Any],
    warnings: list[str],
) -> None:
    """Exponential Moving Average for a single period."""
    key = f"ema_{period}"
    if num_bars < period:
        warnings.append(f"EMA {period} needs {period} bars, only have {num_bars}")
        return
    try:
        series = close.ewm(span=period, adjust=False).mean()
        points = _series_to_points(dates, series, intraday=intraday)
        if points:
            result[key] = {
                "series": points,
                "metadata": {"period": period, "type": "ema"},
            }
        else:
            warnings.append(f"EMA {period} produced no valid data points")
    except Exception as e:
        logger.error("Error computing EMA %d: %s", period, e)
        warnings.append(f"EMA {period} computation failed: {e}")


def _compute_rsi(
    dates: pd.Index,
    close: pd.Series,
    period: int,
    num_bars: int,
    intraday: bool,
    result: dict[str, Any],
    warnings: list[str],
) -> None:
    """Relative Strength Index using Wilder's smoothing method.

    Uses ``ewm(alpha=1/period)`` which is equivalent to Wilder's original
    exponential smoothing and matches the ``ta`` library's RSI output.
    """
    if num_bars < period + 1:
        warnings.append(
            f"RSI {period} needs at least {period + 1} bars, "
            f"only have {num_bars}"
        )
        return
    try:
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)

        avg_gain = gain.ewm(
            alpha=1.0 / period, min_periods=period, adjust=False
        ).mean()
        avg_loss = loss.ewm(
            alpha=1.0 / period, min_periods=period, adjust=False
        ).mean()

        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))

        points = _series_to_points(dates, rsi, intraday=intraday)
        if points:
            result["rsi"] = {
                "series": points,
                "metadata": {"period": period},
            }
        else:
            warnings.append("RSI produced no valid data points")
    except Exception as e:
        logger.error("Error computing RSI: %s", e)
        warnings.append(f"RSI computation failed: {e}")


def _compute_macd(
    dates: pd.Index,
    close: pd.Series,
    fast: int,
    slow: int,
    signal_period: int,
    num_bars: int,
    intraday: bool,
    result: dict[str, Any],
    warnings: list[str],
) -> None:
    """Moving Average Convergence Divergence (MACD)."""
    min_bars_needed = slow + signal_period
    if num_bars < min_bars_needed:
        warnings.append(
            f"MACD needs at least {min_bars_needed} bars, only have {num_bars}"
        )
        return
    try:
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line

        result["macd"] = {
            "macd_line": _series_to_points(dates, macd_line, intraday=intraday),
            "signal_line": _series_to_points(dates, signal_line, intraday=intraday),
            "histogram": _series_to_points(dates, histogram, intraday=intraday),
            "metadata": {
                "fast": fast,
                "slow": slow,
                "signal": signal_period,
            },
        }
    except Exception as e:
        logger.error("Error computing MACD: %s", e)
        warnings.append(f"MACD computation failed: {e}")


def _compute_bb(
    dates: pd.Index,
    close: pd.Series,
    period: int,
    std_dev: float,
    num_bars: int,
    intraday: bool,
    result: dict[str, Any],
    warnings: list[str],
) -> None:
    """Bollinger Bands (upper, middle, lower).

    Uses population standard deviation (``ddof=0``) to match the ``ta``
    library's default behaviour.
    """
    if num_bars < period:
        warnings.append(
            f"Bollinger Bands need {period} bars, only have {num_bars}"
        )
        return
    try:
        middle = close.rolling(window=period).mean()
        std = close.rolling(window=period).std(ddof=0)
        upper = middle + std_dev * std
        lower = middle - std_dev * std

        result["bb"] = {
            "upper": _series_to_points(dates, upper, intraday=intraday),
            "middle": _series_to_points(dates, middle, intraday=intraday),
            "lower": _series_to_points(dates, lower, intraday=intraday),
            "metadata": {"period": period, "std_dev": std_dev},
        }
    except Exception as e:
        logger.error("Error computing Bollinger Bands: %s", e)
        warnings.append(f"Bollinger Bands computation failed: {e}")


def _compute_atr(
    dates: pd.Index,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int,
    num_bars: int,
    intraday: bool,
    result: dict[str, Any],
    warnings: list[str],
) -> None:
    """Average True Range (ATR) using Wilder's smoothing.

    True Range is ``max(H-L, |H-Cprev|, |L-Cprev|)``.  The ATR is
    smoothed with Wilder's EMA (``alpha=1/period``) for consistency with
    the RSI implementation.
    """
    if num_bars < period + 1:
        warnings.append(
            f"ATR {period} needs at least {period + 1} bars, "
            f"only have {num_bars}"
        )
        return
    try:
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.ewm(
            alpha=1.0 / period, min_periods=period, adjust=False
        ).mean()

        points = _series_to_points(dates, atr, intraday=intraday)
        if points:
            result["atr"] = {
                "series": points,
                "metadata": {"period": period},
            }
        else:
            warnings.append("ATR produced no valid data points")
    except Exception as e:
        logger.error("Error computing ATR: %s", e)
        warnings.append(f"ATR computation failed: {e}")


def _compute_obv(
    dates: pd.Index,
    close: pd.Series,
    volume: pd.Series,
    num_bars: int,
    intraday: bool,
    result: dict[str, Any],
    warnings: list[str],
) -> None:
    """On-Balance Volume (OBV).

    Cumulative sum of signed volume where the sign is determined by the
    direction of the close price change.
    """
    if num_bars < 2:
        warnings.append(f"OBV needs at least 2 bars, only have {num_bars}")
        return
    try:
        sign = np.where(
            close > close.shift(1), 1,
            np.where(close < close.shift(1), -1, 0),
        )
        obv = pd.Series((sign * volume.values).cumsum(), index=close.index)

        points = _series_to_points(dates, obv, intraday=intraday)
        if points:
            result["obv"] = {
                "series": points,
                "metadata": {},
            }
        else:
            warnings.append("OBV produced no valid data points")
    except Exception as e:
        logger.error("Error computing OBV: %s", e)
        warnings.append(f"OBV computation failed: {e}")


def _compute_kdj(
    dates: pd.Index,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int,
    d_period: int,
    num_bars: int,
    intraday: bool,
    result: dict[str, Any],
    warnings: list[str],
) -> None:
    """KDJ Stochastic Oscillator (Chinese variant).

    Uses ``ewm(com=d_period-1)`` (equivalent to ``span=2*d_period-1``)
    for the K and D smoothing, which is the standard Chinese KDJ
    formulation and differs from the Western %K/%D SMA approach.

    J = 3K - 2D, and can exceed the 0-100 range.
    """
    min_bars = k_period + d_period
    if num_bars < min_bars:
        warnings.append(
            f"KDJ needs at least {min_bars} bars, only have {num_bars}"
        )
        return
    try:
        low_n = low.rolling(window=k_period).min()
        high_n = high.rolling(window=k_period).max()
        rsv = (close - low_n) / (high_n - low_n + 1e-10) * 100.0

        # com = d_period - 1 gives the Chinese-standard EMA smoothing
        k = rsv.ewm(com=d_period - 1, adjust=False).mean()
        d = k.ewm(com=d_period - 1, adjust=False).mean()
        j = 3.0 * k - 2.0 * d

        result["kdj"] = {
            "k_line": _series_to_points(dates, k, intraday=intraday),
            "d_line": _series_to_points(dates, d, intraday=intraday),
            "j_line": _series_to_points(dates, j, intraday=intraday),
            "metadata": {"k_period": k_period, "d_period": d_period},
        }
    except Exception as e:
        logger.error("Error computing KDJ: %s", e)
        warnings.append(f"KDJ computation failed: {e}")


def _compute_williams_r(
    dates: pd.Index,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int,
    num_bars: int,
    intraday: bool,
    result: dict[str, Any],
    warnings: list[str],
) -> None:
    """Williams %R oscillator.

    Ranges from -100 (oversold) to 0 (overbought).
    """
    if num_bars < period:
        warnings.append(
            f"Williams %R needs {period} bars, only have {num_bars}"
        )
        return
    try:
        high_n = high.rolling(window=period).max()
        low_n = low.rolling(window=period).min()
        wr = (high_n - close) / (high_n - low_n + 1e-10) * (-100.0)

        points = _series_to_points(dates, wr, intraday=intraday)
        if points:
            result["williams_r"] = {
                "series": points,
                "metadata": {"period": period},
            }
        else:
            warnings.append("Williams %R produced no valid data points")
    except Exception as e:
        logger.error("Error computing Williams %%R: %s", e)
        warnings.append(f"Williams %R computation failed: {e}")


def _compute_cci(
    dates: pd.Index,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int,
    num_bars: int,
    intraday: bool,
    result: dict[str, Any],
    warnings: list[str],
) -> None:
    """Commodity Channel Index (CCI).

    Uses the standard 0.015 constant for mean absolute deviation scaling.
    """
    if num_bars < period:
        warnings.append(
            f"CCI needs {period} bars, only have {num_bars}"
        )
        return
    try:
        tp = (high + low + close) / 3.0
        sma_tp = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(
            lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
        )
        cci = (tp - sma_tp) / (0.015 * mad + 1e-10)

        points = _series_to_points(dates, cci, intraday=intraday)
        if points:
            result["cci"] = {
                "series": points,
                "metadata": {"period": period},
            }
        else:
            warnings.append("CCI produced no valid data points")
    except Exception as e:
        logger.error("Error computing CCI: %s", e)
        warnings.append(f"CCI computation failed: {e}")


def _compute_vwap(
    dates: pd.Index,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    num_bars: int,
    intraday: bool,
    result: dict[str, Any],
    warnings: list[str],
) -> None:
    """Volume Weighted Average Price (VWAP).

    For intraday data the cumulative sums reset at each new trading day
    (detected from the date portion of timestamps).  For daily data the
    accumulation runs from the first bar without resetting.
    """
    if num_bars < 1:
        warnings.append(f"VWAP needs at least 1 bar, only have {num_bars}")
        return
    try:
        tp = (high + low + close) / 3.0

        if intraday:
            # Group by the date portion of each timestamp to detect day
            # boundaries and reset cumulative sums at each new day.
            day_groups = dates.map(lambda d: str(d)[:10])
            vwap = pd.Series(dtype=float, index=dates.index)
            for _, idx in dates.groupby(day_groups):
                group_idx = idx.index
                group_vol = volume.loc[group_idx]
                cum_vol = group_vol.cumsum()
                cum_tp_vol = (tp.loc[group_idx] * group_vol).cumsum()
                vwap.loc[group_idx] = cum_tp_vol / (cum_vol + 1e-10)
        else:
            cum_vol = volume.cumsum()
            cum_tp_vol = (tp * volume).cumsum()
            vwap = cum_tp_vol / (cum_vol + 1e-10)

        points = _series_to_points(dates, vwap, intraday=intraday)
        if points:
            result["vwap"] = {
                "series": points,
                "metadata": {},
            }
        else:
            warnings.append("VWAP produced no valid data points")
    except Exception as e:
        logger.error("Error computing VWAP: %s", e)
        warnings.append(f"VWAP computation failed: {e}")


def _compute_sar(
    dates: pd.Index,
    high: pd.Series,
    low: pd.Series,
    num_bars: int,
    af_start: float,
    af_step: float,
    af_max: float,
    intraday: bool,
    result: dict[str, Any],
    warnings: list[str],
) -> None:
    """Parabolic Stop and Reverse (SAR).

    Implements the classic Welles Wilder algorithm which is inherently
    sequential (each bar depends on the previous bar's SAR and extreme
    point).  Runs as a pure-Python loop over numpy arrays for speed.

    Args:
        af_start: Initial acceleration factor.
        af_step: AF increment on each new extreme.
        af_max: Maximum acceleration factor.
    """
    if num_bars < 2:
        warnings.append(f"SAR needs at least 2 bars, only have {num_bars}")
        return
    try:
        high_arr = high.values.astype(float)
        low_arr = low.values.astype(float)
        n = len(high_arr)
        sar_arr = np.empty(n, dtype=float)

        # Initialize: assume uptrend if bar 1 high > bar 0 high
        is_long = bool(high_arr[1] >= high_arr[0])

        if is_long:
            sar_arr[0] = low_arr[0]
            ep = high_arr[0]  # extreme point
        else:
            sar_arr[0] = high_arr[0]
            ep = low_arr[0]

        af = af_start

        for i in range(1, n):
            prev_sar = sar_arr[i - 1]

            # Compute candidate SAR
            sar_candidate = prev_sar + af * (ep - prev_sar)

            if is_long:
                # SAR must not be above the two prior lows
                if i >= 2:
                    sar_candidate = min(
                        sar_candidate, low_arr[i - 1], low_arr[i - 2]
                    )
                else:
                    sar_candidate = min(sar_candidate, low_arr[i - 1])

                if low_arr[i] < sar_candidate:
                    # Reversal to short
                    is_long = False
                    sar_arr[i] = ep  # SAR flips to the extreme point
                    ep = low_arr[i]
                    af = af_start
                else:
                    sar_arr[i] = sar_candidate
                    if high_arr[i] > ep:
                        ep = high_arr[i]
                        af = min(af + af_step, af_max)
            else:
                # SAR must not be below the two prior highs
                if i >= 2:
                    sar_candidate = max(
                        sar_candidate, high_arr[i - 1], high_arr[i - 2]
                    )
                else:
                    sar_candidate = max(sar_candidate, high_arr[i - 1])

                if high_arr[i] > sar_candidate:
                    # Reversal to long
                    is_long = True
                    sar_arr[i] = ep
                    ep = high_arr[i]
                    af = af_start
                else:
                    sar_arr[i] = sar_candidate
                    if low_arr[i] < ep:
                        ep = low_arr[i]
                        af = min(af + af_step, af_max)

        sar_series = pd.Series(sar_arr, index=high.index)
        points = _series_to_points(dates, sar_series, intraday=intraday)
        if points:
            result["sar"] = {
                "series": points,
                "metadata": {
                    "af_start": af_start,
                    "af_step": af_step,
                    "af_max": af_max,
                },
            }
        else:
            warnings.append("SAR produced no valid data points")
    except Exception as e:
        logger.error("Error computing SAR: %s", e)
        warnings.append(f"SAR computation failed: {e}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_indicator_series(
    bars: list[dict],
    indicator_types: list[str],
    ma_periods: list[int] | None = None,
    rsi_period: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    bb_period: int = 20,
    bb_std: float = 2.0,
    atr_period: int = 14,
    kdj_k_period: int = 9,
    kdj_d_period: int = 3,
    williams_r_period: int = 14,
    cci_period: int = 20,
    sar_af_start: float = 0.02,
    sar_af_step: float = 0.02,
    sar_af_max: float = 0.2,
    intraday: bool = False,
) -> dict:
    """Compute technical indicator series from OHLCV bars.

    This is the primary entry point.  It builds a pandas DataFrame from
    the incoming bar dicts, then dispatches to the individual indicator
    functions based on *indicator_types*.

    Args:
        bars: List of bar dicts.  Each dict must contain at minimum
            ``date`` and ``close``; indicators that need OHLCV will also
            require ``open``, ``high``, ``low``, and ``volume``.
        indicator_types: Which indicators to compute.  Valid values:
            ``"sma"``, ``"ema"``, ``"rsi"``, ``"macd"``, ``"bb"``,
            ``"atr"``, ``"obv"``, ``"kdj"``, ``"williams_r"``,
            ``"cci"``, ``"vwap"``, ``"sar"``.
        ma_periods: Periods for SMA/EMA.  Defaults to ``[20, 50, 200]``.
        rsi_period: Period for RSI.  Defaults to 14.
        macd_fast: Fast EMA period for MACD.  Defaults to 12.
        macd_slow: Slow EMA period for MACD.  Defaults to 26.
        macd_signal: Signal line period for MACD.  Defaults to 9.
        bb_period: Period for Bollinger Bands.  Defaults to 20.
        bb_std: Std-dev multiplier for Bollinger Bands.  Defaults to 2.0.
        atr_period: Period for ATR.  Defaults to 14.
        kdj_k_period: K lookback period for KDJ.  Defaults to 9.
        kdj_d_period: D smoothing period for KDJ.  Defaults to 3.
        williams_r_period: Lookback period for Williams %R.  Defaults to 14.
        cci_period: Period for CCI.  Defaults to 20.
        sar_af_start: Initial acceleration factor for SAR.  Defaults to 0.02.
        sar_af_step: AF increment for SAR.  Defaults to 0.02.
        sar_af_max: Maximum AF for SAR.  Defaults to 0.2.
        intraday: When True, preserve full timestamps in output points.

    Returns:
        Dictionary mapping indicator keys to their series/metadata, plus a
        ``"warnings"`` list of non-fatal messages (e.g. insufficient data).
    """
    if ma_periods is None:
        ma_periods = [20, 50, 200]

    start_time = time.monotonic()
    result: dict[str, Any] = {}
    warnings: list[str] = []
    num_bars = len(bars)

    logger.info(
        "Computing indicators %s for %d bars (ma_periods=%s, intraday=%s)",
        indicator_types, num_bars, ma_periods, intraday,
    )

    if num_bars < 2:
        warnings.append(f"Insufficient data: only {num_bars} bar(s) provided")
        result["warnings"] = warnings
        return result

    # ------------------------------------------------------------------
    # Build DataFrame from bars
    # ------------------------------------------------------------------
    try:
        df = pd.DataFrame(bars)
        if "close" not in df.columns or "date" not in df.columns:
            warnings.append(
                "Bar data missing required 'close' or 'date' columns"
            )
            result["warnings"] = warnings
            return result

        close = df["close"].astype(float)
        dates = df["date"]

        # Parse OHLCV columns that some indicators need.  Missing columns
        # are created as NaN so that indicator helpers can detect the gap
        # and emit a warning rather than crashing at DataFrame construction.
        high = df["high"].astype(float) if "high" in df.columns else pd.Series(
            np.nan, index=df.index
        )
        low = df["low"].astype(float) if "low" in df.columns else pd.Series(
            np.nan, index=df.index
        )
        volume = df["volume"].astype(float) if "volume" in df.columns else pd.Series(
            np.nan, index=df.index
        )

    except (KeyError, ValueError, TypeError) as e:
        logger.error("Invalid bar data structure: %s", e)
        warnings.append(f"Invalid bar data: {e}")
        result["warnings"] = warnings
        return result

    # Helper to check that an OHLCV column is actually available
    def _require_column(name: str, indicator_name: str) -> bool:
        """Return True if *name* column has real values, else warn."""
        series_map = {"high": high, "low": low, "volume": volume}
        s = series_map.get(name)
        if s is None or s.isna().all():
            warnings.append(
                f"{indicator_name} requires '{name}' data which is missing"
            )
            return False
        return True

    # ------------------------------------------------------------------
    # Dispatch to individual indicator computations
    # ------------------------------------------------------------------
    indicator_set = set(indicator_types)

    # SMA
    if "sma" in indicator_set:
        for period in ma_periods:
            _compute_sma(dates, close, period, num_bars, intraday, result, warnings)

    # EMA
    if "ema" in indicator_set:
        for period in ma_periods:
            _compute_ema(dates, close, period, num_bars, intraday, result, warnings)

    # RSI
    if "rsi" in indicator_set:
        _compute_rsi(dates, close, rsi_period, num_bars, intraday, result, warnings)

    # MACD
    if "macd" in indicator_set:
        _compute_macd(
            dates, close, macd_fast, macd_slow, macd_signal,
            num_bars, intraday, result, warnings,
        )

    # Bollinger Bands
    if "bb" in indicator_set:
        _compute_bb(
            dates, close, bb_period, bb_std, num_bars, intraday, result, warnings,
        )

    # ATR (requires high, low, close)
    if "atr" in indicator_set:
        if _require_column("high", "ATR") and _require_column("low", "ATR"):
            _compute_atr(
                dates, high, low, close, atr_period,
                num_bars, intraday, result, warnings,
            )

    # OBV (requires volume)
    if "obv" in indicator_set:
        if _require_column("volume", "OBV"):
            _compute_obv(
                dates, close, volume, num_bars, intraday, result, warnings,
            )

    # KDJ (requires high, low)
    if "kdj" in indicator_set:
        if _require_column("high", "KDJ") and _require_column("low", "KDJ"):
            _compute_kdj(
                dates, high, low, close, kdj_k_period, kdj_d_period,
                num_bars, intraday, result, warnings,
            )

    # Williams %R (requires high, low)
    if "williams_r" in indicator_set:
        if (
            _require_column("high", "Williams %R")
            and _require_column("low", "Williams %R")
        ):
            _compute_williams_r(
                dates, high, low, close, williams_r_period,
                num_bars, intraday, result, warnings,
            )

    # CCI (requires high, low)
    if "cci" in indicator_set:
        if _require_column("high", "CCI") and _require_column("low", "CCI"):
            _compute_cci(
                dates, high, low, close, cci_period,
                num_bars, intraday, result, warnings,
            )

    # VWAP (requires high, low, volume)
    if "vwap" in indicator_set:
        if (
            _require_column("high", "VWAP")
            and _require_column("low", "VWAP")
            and _require_column("volume", "VWAP")
        ):
            _compute_vwap(
                dates, high, low, close, volume,
                num_bars, intraday, result, warnings,
            )

    # Parabolic SAR (requires high, low)
    if "sar" in indicator_set:
        if _require_column("high", "SAR") and _require_column("low", "SAR"):
            _compute_sar(
                dates, high, low, num_bars,
                sar_af_start, sar_af_step, sar_af_max,
                intraday, result, warnings,
            )

    # ------------------------------------------------------------------
    # Finalize
    # ------------------------------------------------------------------
    result["warnings"] = warnings

    elapsed_ms = (time.monotonic() - start_time) * 1000
    indicator_count = len([k for k in result if k != "warnings"])
    logger.info(
        "Indicator computation completed in %.1fms: %d indicator(s), %d warning(s)",
        elapsed_ms, indicator_count, len(warnings),
    )

    return result
