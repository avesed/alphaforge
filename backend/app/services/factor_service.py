"""Alpha158 factor computation using Qlib D.features().

Computes time-series quantitative features for single-stock analysis.
All methods are synchronous -- designed to run in ThreadPoolExecutor
via run_qlib_quick().

The feature set is a curated subset of Qlib's Alpha158 that works for
single-stock analysis (no cross-sectional data required). Features span:
price dynamics, moving averages, volatility, volume, correlation, trend,
momentum, range, EMA, skewness/kurtosis, MACD-like, Bollinger-like.
"""
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Alpha158 time-series features (usable for single-stock analysis, ~70 features)
# These do NOT require cross-sectional data
SINGLE_STOCK_FEATURES = [
    # Price dynamics
    "$open/$close", "$high/$close", "$low/$close",
    "$close-Ref($close,1)", "($close-Ref($close,1))/Ref($close,1)",
    # Moving averages
    "Mean($close,5)", "Mean($close,10)", "Mean($close,20)", "Mean($close,30)", "Mean($close,60)",
    "$close/Mean($close,5)", "$close/Mean($close,10)", "$close/Mean($close,20)",
    # Volatility
    "Std($close,5)", "Std($close,10)", "Std($close,20)", "Std($close,60)",
    "Std($close,5)/Mean($close,5)", "Std($close,10)/Mean($close,10)",
    # Volume features
    "Mean($volume,5)", "Mean($volume,10)", "Mean($volume,20)", "Mean($volume,60)",
    "$volume/Mean($volume,5)", "$volume/Mean($volume,10)", "$volume/Mean($volume,20)",
    "Std($volume,5)", "Std($volume,10)",
    # Price-volume correlation
    "Corr($close,$volume,5)", "Corr($close,$volume,10)", "Corr($close,$volume,20)",
    # Trend
    "Slope($close,5)/Std($close,5)", "Slope($close,10)/Std($close,10)", "Slope($close,20)/Std($close,20)",
    "Rsquare($close,5)", "Rsquare($close,10)", "Rsquare($close,20)",
    # Momentum
    "Ref($close,1)/$close", "Ref($close,5)/$close", "Ref($close,10)/$close", "Ref($close,20)/$close",
    # Min/Max
    "Min($low,5)/$close", "Min($low,10)/$close", "Min($low,20)/$close",
    "Max($high,5)/$close", "Max($high,10)/$close", "Max($high,20)/$close",
    # Range
    "($high-$low)/$close", "($close-$low)/($high-$low+1e-8)",
    # Volume momentum
    "Ref($volume,1)/$volume", "Ref($volume,5)/$volume", "Ref($volume,10)/$volume",
    # EMA
    "EMA($close,5)", "EMA($close,10)", "EMA($close,20)",
    "$close/EMA($close,5)", "$close/EMA($close,10)", "$close/EMA($close,20)",
    # Skewness/Kurtosis
    "Skew($close,20)", "Kurt($close,20)", "Skew($volume,20)",
    # MACD-like
    "EMA($close,12)-EMA($close,26)", "(EMA($close,12)-EMA($close,26))/$close",
    # Bollinger-like
    "(Mean($close,20)+2*Std($close,20)-$close)/$close",
    "(Mean($close,20)-2*Std($close,20)-$close)/$close",
    # High-Low range
    "Mean($high-$low,5)/$close", "Mean($high-$low,20)/$close",
    # Turnover proxy
    "Sum(If($close>Ref($close,1),1,0),5)/5",
    "Sum(If($close>Ref($close,1),1,0),10)/10",
    "Sum(If($close>Ref($close,1),1,0),20)/20",
    # Delta features
    "Delta(Mean($close,5),5)", "Delta(Mean($close,10),5)",
    # === Additional high-alpha factors ===
    # Medium-term momentum (Jegadeesh-Titman -- critical for US equities)
    "Ref($close,60)/$close",
    # Longer-term trend
    "$close/Mean($close,60)",
    # Return volatility (realized vol, different from price std)
    "Std(($close-Ref($close,1))/Ref($close,1)+1e-8,20)",
    "Std(($close-Ref($close,1))/Ref($close,1)+1e-8,60)",
    # Max drawdown (downside risk)
    "Min($close,20)/Max($close,20)",
    "Min($close,60)/Max($close,60)",
    # Amihud illiquidity proxy: |daily return| / relative volume
    "Mean(Abs(($close-Ref($close,1))/$close)/($volume/Mean($volume,20)+1e-8),20)",
    # Return-volume correlation (smart money flow)
    "Corr(($close-Ref($close,1))/$close,$volume,10)",
    "Corr(($close-Ref($close,1))/$close,$volume,20)",
    # RSI-like: ratio of up-move magnitude to total move magnitude
    "Sum(If($close>Ref($close,1),$close-Ref($close,1),0),14)/(Sum(Abs($close-Ref($close,1)),14)+1e-8)",
    # MA crossover signals (trend change detection)
    "EMA($close,5)/EMA($close,20)",
    "EMA($close,12)/EMA($close,26)",
    # Volume kurtosis (distribution tail risk)
    "Kurt($volume,20)",
    # Longer-term volatility context
    "Std($close,20)/Std($close,60)",
]

# Human-readable names for the features above (1:1 positional mapping)
FEATURE_NAMES = [
    "open_close_ratio", "high_close_ratio", "low_close_ratio",
    "close_change", "close_return",
    "ma5", "ma10", "ma20", "ma30", "ma60",
    "close_ma5_ratio", "close_ma10_ratio", "close_ma20_ratio",
    "std5", "std10", "std20", "std60",
    "cv5", "cv10",
    "vol_ma5", "vol_ma10", "vol_ma20", "vol_ma60",
    "vol_ratio5", "vol_ratio10", "vol_ratio20",
    "vol_std5", "vol_std10",
    "corr_close_vol_5", "corr_close_vol_10", "corr_close_vol_20",
    "slope5_norm", "slope10_norm", "slope20_norm",
    "rsquare5", "rsquare10", "rsquare20",
    "ret1", "ret5", "ret10", "ret20",
    "min_low5", "min_low10", "min_low20",
    "max_high5", "max_high10", "max_high20",
    "range_ratio", "close_in_range",
    "vol_ret1", "vol_ret5", "vol_ret10",
    "ema5", "ema10", "ema20",
    "close_ema5_ratio", "close_ema10_ratio", "close_ema20_ratio",
    "skew20", "kurt20", "vol_skew20",
    "macd_raw", "macd_norm",
    "bb_upper_dist", "bb_lower_dist",
    "hl_range5", "hl_range20",
    "up_ratio5", "up_ratio10", "up_ratio20",
    "delta_ma5", "delta_ma10",
    # Additional high-alpha factors
    "ret60",
    "close_ma60_ratio",
    "return_vol20", "return_vol60",
    "drawdown20", "drawdown60",
    "amihud20",
    "corr_ret_vol10", "corr_ret_vol20",
    "rsi14",
    "ema_cross_5_20", "ema_cross_12_26",
    "vol_kurt20",
    "vol_regime",
]


def _empty_result(
    symbol: str, market: str, alpha_type: str
) -> Dict[str, Any]:
    """Return an empty FactorResult-compatible dict."""
    return {
        "symbol": symbol,
        "market": market,
        "alpha_type": alpha_type,
        "mode": "single",
        "factor_count": 0,
        "dates": [],
        "factors": {},
        "top_factors": [],
    }


class FactorService:
    """Alpha158 factor computation using Qlib D.features().

    All methods are synchronous -- designed to run in ThreadPoolExecutor
    via run_qlib_quick().
    """

    @staticmethod
    def compute_factors(
        symbol: str,
        market: str,
        alpha_type: str = "alpha158",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        data_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compute Alpha158 factors for a single symbol.

        Uses Qlib D.features() with single-stock expression list.
        Returns a FactorResult-compatible dict.

        Args:
            symbol: Stock symbol (e.g., AAPL, 600000.SS).
            market: Market code (us, hk, cn, sh, sz, metal).
            alpha_type: Factor set type (currently only alpha158).
            start_date: Start date (YYYY-MM-DD). Defaults to 90 days ago.
            end_date: End date (YYYY-MM-DD). Defaults to today.
            data_dir: Override Qlib data directory.

        Returns:
            Dict with keys: symbol, market, alpha_type, mode, factor_count,
            dates, factors, top_factors.
        """
        from app.config import get_settings
        from app.context import QlibContext
        from app.utils.symbol_mapping import normalize_symbol_for_qlib

        settings = get_settings()
        data_dir = data_dir or settings.QLIB_DATA_DIR

        logger.info(
            "Computing factors: symbol=%s, market=%s, alpha_type=%s",
            symbol, market, alpha_type,
        )

        # Initialize Qlib for the target market
        try:
            QlibContext.ensure_init(market, data_dir)
        except Exception as e:
            logger.error("Qlib init failed for market=%s: %s", market, e)
            return _empty_result(symbol, market, alpha_type)

        # Lazy imports -- qlib must be initialized first
        import pandas as pd
        from qlib.data import D

        qlib_symbol = normalize_symbol_for_qlib(symbol, market)
        expressions = SINGLE_STOCK_FEATURES
        names = FEATURE_NAMES

        # Build date range
        end = end_date or pd.Timestamp.now().strftime("%Y-%m-%d")
        start = start_date or (
            pd.Timestamp(end) - pd.Timedelta(days=90)
        ).strftime("%Y-%m-%d")

        logger.info(
            "D.features() call: symbol=%s, start=%s, end=%s, features=%d",
            qlib_symbol, start, end, len(expressions),
        )

        try:
            df = D.features(
                instruments=[qlib_symbol],
                fields=expressions,
                start_time=start,
                end_time=end,
            )
        except Exception as e:
            logger.error("D.features() failed for %s: %s", qlib_symbol, e)
            return _empty_result(symbol, market, alpha_type)

        if df.empty:
            logger.warning("D.features() returned empty DataFrame for %s", qlib_symbol)
            return _empty_result(symbol, market, alpha_type)

        # Flatten MultiIndex if needed (instrument, datetime) -> (datetime)
        if hasattr(df.index, "levels"):
            df = df.droplevel(0)

        # Rename columns to human-readable names
        df.columns = names[: len(df.columns)]
        dates = [d.strftime("%Y-%m-%d") for d in df.index]

        # Build factors dict: {name: [values]}
        factors: Dict[str, List[Optional[float]]] = {}
        for col in df.columns:
            values = df[col].tolist()
            factors[col] = [
                None if pd.isna(v) else round(float(v), 6) for v in values
            ]

        # Compute top factors by absolute z-score at the latest date
        top_factors = FactorService._compute_top_factors(df, count=20)

        logger.info(
            "Factor computation complete: symbol=%s, dates=%d, factors=%d",
            symbol, len(dates), len(df.columns),
        )

        return {
            "symbol": symbol,
            "market": market,
            "alpha_type": alpha_type,
            "mode": "single",
            "factor_count": len(df.columns),
            "dates": dates,
            "factors": factors,
            "top_factors": top_factors,
        }

    @staticmethod
    def get_factor_summary(
        symbol: str,
        market: str,
        data_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get top 10 factor summary -- optimized for LLM agents.

        Returns a compact FactorSummary-compatible dict with only the most
        significant factors by z-score, suitable for feeding into analysis prompts.

        Args:
            symbol: Stock symbol.
            market: Market code.
            data_dir: Override Qlib data directory.

        Returns:
            Dict with keys: symbol, market, latest_date, top_factors, mode.
        """
        logger.info("Computing factor summary: symbol=%s, market=%s", symbol, market)

        result = FactorService.compute_factors(
            symbol, market, data_dir=data_dir
        )

        return {
            "symbol": result["symbol"],
            "market": result["market"],
            "latest_date": result["dates"][-1] if result["dates"] else None,
            "top_factors": result["top_factors"][:10],
            "mode": result["mode"],
        }

    @staticmethod
    def _compute_top_factors(
        df: "pd.DataFrame", count: int = 20
    ) -> List[Dict[str, Any]]:
        """Rank factors by absolute z-score at the latest date.

        Args:
            df: DataFrame with factor columns and DatetimeIndex.
            count: Number of top factors to return.

        Returns:
            List of dicts with name, value, z_score, sorted by |z_score| descending.
        """
        import pandas as pd

        if df.empty:
            return []

        latest = df.iloc[-1]
        means = df.mean()
        stds = df.std()

        z_scores: Dict[str, float] = {}
        for col in df.columns:
            if stds[col] > 1e-8:
                z_scores[col] = float((latest[col] - means[col]) / stds[col])

        top_factors = sorted(
            [
                {
                    "name": k,
                    "value": round(float(latest[k]), 6),
                    "z_score": round(v, 4),
                }
                for k, v in z_scores.items()
                if not pd.isna(latest[k])
            ],
            key=lambda x: abs(x["z_score"]),
            reverse=True,
        )

        return top_factors[:count]
