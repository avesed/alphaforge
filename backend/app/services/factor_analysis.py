"""Factor IC analysis and cross-sectional ranking.

Provides Information Coefficient (IC) computation for evaluating factor
predictive power, and cross-sectional percentile ranking for comparing
a factor value across a universe of stocks.

All methods are synchronous -- designed to run in ThreadPoolExecutor
via run_qlib_quick().
"""
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class FactorAnalysisService:
    """Factor analysis: IC/ICIR computation and cross-sectional ranking.

    All methods are synchronous -- designed to run in ThreadPoolExecutor
    via run_qlib_quick().
    """

    @staticmethod
    def compute_ic(
        universe: List[str],
        market: str,
        factor_names: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        forward_days: int = 5,
        data_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compute Information Coefficient (IC) and IC Information Ratio (ICIR).

        For each factor, computes the rank IC (Spearman correlation) between
        factor values and forward returns across the universe on each date,
        then averages across dates.

        ICIR = mean(IC) / std(IC), measuring consistency of the signal.

        Args:
            universe: List of stock symbols to evaluate.
            market: Market code (us, hk, cn, etc.).
            factor_names: Specific factor names to evaluate. Empty/None = all Alpha158.
            start_date: Start date (YYYY-MM-DD). Defaults to 180 days ago.
            end_date: End date (YYYY-MM-DD). Defaults to today.
            forward_days: Number of days for forward return calculation (1-20).
            data_dir: Override Qlib data directory.

        Returns:
            ICResult-compatible dict with factor_ic, factor_icir, ic_series.
        """
        from app.config import get_settings
        from app.context import QlibContext
        from app.services.factor_service import FEATURE_NAMES, SINGLE_STOCK_FEATURES
        from app.utils.symbol_mapping import normalize_symbol_for_qlib

        settings = get_settings()
        data_dir = data_dir or settings.QLIB_DATA_DIR

        logger.info(
            "Computing IC: universe=%d symbols, market=%s, forward_days=%d",
            len(universe), market, forward_days,
        )

        # Initialize Qlib
        try:
            QlibContext.ensure_init(market, data_dir)
        except Exception as e:
            logger.error("Qlib init failed for market=%s: %s", market, e)
            return {"factor_ic": {}, "factor_icir": {}, "ic_series": {}}

        # Lazy imports
        import numpy as np
        import pandas as pd
        from qlib.data import D
        from scipy import stats

        # Build date range
        end = end_date or pd.Timestamp.now().strftime("%Y-%m-%d")
        start = start_date or (
            pd.Timestamp(end) - pd.Timedelta(days=180)
        ).strftime("%Y-%m-%d")

        # Normalize symbols
        qlib_symbols = [
            normalize_symbol_for_qlib(s, market) for s in universe
        ]

        # Determine which factors to compute
        if factor_names:
            # Map requested names to expressions
            name_to_expr = dict(zip(FEATURE_NAMES, SINGLE_STOCK_FEATURES))
            expressions = []
            names = []
            for name in factor_names:
                if name in name_to_expr:
                    expressions.append(name_to_expr[name])
                    names.append(name)
                else:
                    logger.warning("Unknown factor name: %s, skipping", name)
        else:
            expressions = SINGLE_STOCK_FEATURES
            names = FEATURE_NAMES

        # Add forward return expression for IC computation
        fwd_return_expr = f"Ref($close,-{forward_days})/$close-1"
        all_expressions = expressions + [fwd_return_expr]
        all_names = names + ["_fwd_return"]

        logger.info(
            "D.features() call: %d symbols, %d factors, start=%s, end=%s",
            len(qlib_symbols), len(expressions), start, end,
        )

        try:
            df = D.features(
                instruments=qlib_symbols,
                fields=all_expressions,
                start_time=start,
                end_time=end,
            )
        except Exception as e:
            logger.error("D.features() failed for IC computation: %s", e)
            return {"factor_ic": {}, "factor_icir": {}, "ic_series": {}}

        if df.empty:
            logger.warning("D.features() returned empty for IC computation")
            return {"factor_ic": {}, "factor_icir": {}, "ic_series": {}}

        # Rename columns
        df.columns = all_names[: len(df.columns)]

        # Compute rank IC per date (cross-sectional Spearman correlation)
        factor_ic: Dict[str, float] = {}
        factor_icir: Dict[str, float] = {}
        ic_series: Dict[str, List[Dict[str, Any]]] = {}

        if "_fwd_return" not in df.columns:
            logger.warning("Forward return column missing from D.features() result")
            return {"factor_ic": {}, "factor_icir": {}, "ic_series": {}}

        # Group by date (second level of MultiIndex)
        if hasattr(df.index, "levels") and len(df.index.levels) == 2:
            dates = df.index.get_level_values(1).unique()
        else:
            # Single symbol -- cannot compute cross-sectional IC
            logger.warning(
                "IC requires multiple symbols. Got single-level index."
            )
            return {"factor_ic": {}, "factor_icir": {}, "ic_series": {}}

        for factor_name in names:
            if factor_name not in df.columns:
                continue

            daily_ics: List[Dict[str, Any]] = []

            for dt in dates:
                try:
                    # Get cross-section for this date
                    cross_section = df.xs(dt, level=1)
                    factor_vals = cross_section[factor_name].dropna()
                    fwd_vals = cross_section["_fwd_return"].dropna()

                    # Align on common instruments
                    common_idx = factor_vals.index.intersection(fwd_vals.index)
                    if len(common_idx) < 3:
                        continue

                    f_vals = factor_vals.loc[common_idx]
                    r_vals = fwd_vals.loc[common_idx]

                    # Spearman rank correlation
                    corr, _ = stats.spearmanr(f_vals.values, r_vals.values)
                    if not np.isnan(corr):
                        daily_ics.append({
                            "date": dt.strftime("%Y-%m-%d"),
                            "value": round(float(corr), 6),
                        })
                except Exception as e:
                    logger.debug(
                        "IC computation skipped for factor=%s date=%s: %s",
                        factor_name, dt, e,
                    )
                    continue

            if daily_ics:
                ic_values = [d["value"] for d in daily_ics]
                mean_ic = float(np.mean(ic_values))
                std_ic = float(np.std(ic_values))

                factor_ic[factor_name] = round(mean_ic, 6)
                factor_icir[factor_name] = (
                    round(mean_ic / std_ic, 6) if std_ic > 1e-8 else 0.0
                )
                ic_series[factor_name] = daily_ics

        logger.info(
            "IC computation complete: %d factors evaluated", len(factor_ic)
        )

        return {
            "factor_ic": factor_ic,
            "factor_icir": factor_icir,
            "ic_series": ic_series,
        }

    @staticmethod
    def compute_cs_rank(
        expression: str,
        symbols: List[str],
        market: str,
        target_date: Optional[str] = None,
        data_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compute cross-sectional percentile rank for an expression.

        Evaluates a Qlib expression across all provided symbols on a single
        date and returns the percentile rank of each symbol (0.0 = lowest,
        1.0 = highest).

        Args:
            expression: Qlib expression (e.g., "Corr($close,$volume,20)").
            symbols: List of symbols to rank.
            market: Market code.
            target_date: Date to evaluate (YYYY-MM-DD). Defaults to latest.
            data_dir: Override Qlib data directory.

        Returns:
            CSRankResult-compatible dict with expression, date, rankings.
        """
        from app.config import get_settings
        from app.context import QlibContext
        from app.utils.symbol_mapping import normalize_symbol_for_qlib

        settings = get_settings()
        data_dir = data_dir or settings.QLIB_DATA_DIR

        logger.info(
            "Computing CS rank: expression=%s, symbols=%d, market=%s",
            expression, len(symbols), market,
        )

        # Initialize Qlib
        try:
            QlibContext.ensure_init(market, data_dir)
        except Exception as e:
            logger.error("Qlib init failed for market=%s: %s", market, e)
            return {"expression": expression, "date": None, "rankings": {}}

        # Lazy imports
        import pandas as pd
        from qlib.data import D

        # Normalize symbols
        qlib_symbols = [
            normalize_symbol_for_qlib(s, market) for s in symbols
        ]

        # Build a mapping to return results with original symbol names
        qlib_to_original = dict(zip(qlib_symbols, symbols))

        # Date range: single date (or last 5 days to ensure we have data)
        end = target_date or pd.Timestamp.now().strftime("%Y-%m-%d")
        start = (pd.Timestamp(end) - pd.Timedelta(days=5)).strftime("%Y-%m-%d")

        try:
            df = D.features(
                instruments=qlib_symbols,
                fields=[expression],
                start_time=start,
                end_time=end,
            )
        except Exception as e:
            logger.error("D.features() failed for CS rank: %s", e)
            return {"expression": expression, "date": None, "rankings": {}}

        if df.empty:
            logger.warning("D.features() returned empty for CS rank")
            return {"expression": expression, "date": None, "rankings": {}}

        # Get the latest date's cross-section
        df.columns = ["value"]

        if hasattr(df.index, "levels") and len(df.index.levels) == 2:
            dates = df.index.get_level_values(1).unique()
            latest_date = sorted(dates)[-1]
            cross_section = df.xs(latest_date, level=1)["value"].dropna()
        else:
            # Single symbol
            latest_date = df.index[-1]
            cross_section = df.iloc[-1:]
            cross_section = pd.Series(
                cross_section["value"].values,
                index=[qlib_symbols[0]] if qlib_symbols else [],
            )

        if cross_section.empty:
            return {"expression": expression, "date": None, "rankings": {}}

        # Compute percentile ranks (0 to 1)
        ranked = cross_section.rank(pct=True)

        # Map back to original symbol names
        rankings: Dict[str, float] = {}
        for qlib_sym, rank_val in ranked.items():
            original = qlib_to_original.get(qlib_sym, qlib_sym)
            if not pd.isna(rank_val):
                rankings[original] = round(float(rank_val), 6)

        result_date = (
            latest_date.strftime("%Y-%m-%d")
            if hasattr(latest_date, "strftime")
            else str(latest_date)
        )

        logger.info(
            "CS rank complete: expression=%s, ranked=%d symbols, date=%s",
            expression, len(rankings), result_date,
        )

        return {
            "expression": expression,
            "date": result_date,
            "rankings": rankings,
        }
