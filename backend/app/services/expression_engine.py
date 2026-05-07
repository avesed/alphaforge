"""Qlib expression engine -- evaluates arbitrary Qlib expressions.

This is the core capability exposed to LLM agents: they can construct
Qlib expressions dynamically (e.g., Corr($close,$volume,20)) and get
results for any stock.

Security: expressions are validated against an operator whitelist and
length limit before execution. No Python code execution -- only Qlib's
built-in expression operators are allowed. Arithmetic operators (+, -, *, /)
and numeric literals pass through intentionally because Qlib's internal
parser is the execution layer, not Python eval().
"""
import logging
import math
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Whitelist of allowed Qlib expression operators
ALLOWED_OPERATORS = {
    # Time-series
    "Ref", "Mean", "Std", "Var", "Skew", "Kurt", "Med", "Mad",
    "EMA", "WMA", "Slope", "Rsquare", "Resi",
    # Cross-sectional
    "Rank", "CSRank", "CSZScore",
    # Math
    "Abs", "Sign", "Log", "Power",
    # Statistics
    "Corr", "Cov", "Delta",
    # Logical
    "Greater", "Less", "If", "And", "Or", "Not",
    # Aggregation
    "Count", "Sum", "Max", "Min",
    "IdxMax", "IdxMin",
    # Other
    "Mask", "Quantile",
}

# Allowed variable names (Qlib feature references)
ALLOWED_VARIABLES = {"$open", "$high", "$low", "$close", "$volume", "$factor"}

# Pattern to extract operator names from expression
OPERATOR_PATTERN = re.compile(r'\b([A-Z][a-zA-Z]+)\s*\(')

# Pattern for variable references
VARIABLE_PATTERN = re.compile(r'\$[a-z]+')

# Dangerous patterns to reject
DANGEROUS_PATTERNS = [
    r'__',           # dunder methods
    r'import\s',     # import statements
    r'exec\s*\(',    # exec calls
    r'eval\s*\(',    # eval calls
    r'open\s*\(',    # file access
    r'os\.',         # os module
    r'sys\.',        # sys module
    r'subprocess',   # subprocess
]


class ExpressionEngine:
    """Qlib expression evaluation with security validation.

    All methods are synchronous -- designed to run in ThreadPoolExecutor
    via run_qlib_quick(). The validate() method is pure string checking
    and can be called directly without Qlib.
    """

    @staticmethod
    def validate(
        expression: str, max_length: int = 500
    ) -> Tuple[bool, str, List[str]]:
        """Validate expression syntax and safety.

        Returns (is_valid, error_message, operators_used).
        Does NOT call Qlib -- pure string validation.
        """
        # Length check
        if len(expression) > max_length:
            return (
                False,
                f"Expression too long ({len(expression)} > {max_length})",
                [],
            )

        if not expression.strip():
            return False, "Empty expression", []

        # Dangerous pattern check (case-insensitive for defense-in-depth)
        for pattern in DANGEROUS_PATTERNS:
            if re.search(pattern, expression, re.IGNORECASE):
                return False, "Expression contains disallowed pattern", []

        # Extract and validate operators
        operators_used = list(set(OPERATOR_PATTERN.findall(expression)))
        unknown_ops = [op for op in operators_used if op not in ALLOWED_OPERATORS]
        if unknown_ops:
            return (
                False,
                f"Unknown operators: {', '.join(unknown_ops)}. "
                f"Allowed: {', '.join(sorted(ALLOWED_OPERATORS))}",
                operators_used,
            )

        # Validate variables
        variables_used = set(VARIABLE_PATTERN.findall(expression))
        unknown_vars = variables_used - ALLOWED_VARIABLES
        if unknown_vars:
            return (
                False,
                f"Unknown variables: {', '.join(unknown_vars)}. "
                f"Allowed: {', '.join(sorted(ALLOWED_VARIABLES))}",
                operators_used,
            )

        # Basic bracket matching
        if expression.count('(') != expression.count(')'):
            return False, "Unbalanced parentheses", operators_used

        return True, "", operators_used

    # Period string to approximate days mapping
    PERIOD_TO_DAYS = {
        "1mo": 30, "3mo": 90, "6mo": 180,
        "1y": 365, "2y": 730, "5y": 1825,
    }

    @staticmethod
    def evaluate(
        symbol: str,
        expression: str,
        market: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = "3mo",
        data_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Evaluate a Qlib expression for a single symbol.

        Synchronous -- runs in ThreadPoolExecutor via run_qlib_quick().
        Returns ExpressionResult-compatible dict.

        Args:
            symbol: Stock symbol (e.g., AAPL, 600000.SS).
            expression: Qlib expression (e.g., Corr($close,$volume,20)).
            market: Market code (us, hk, cn, sh, sz, metal).
            start_date: Start date (YYYY-MM-DD). Defaults based on period.
            end_date: End date (YYYY-MM-DD). Defaults to today.
            period: Lookback period (e.g., 3mo, 1y). Used when start_date is not provided.
            data_dir: Override Qlib data directory.

        Returns:
            Dict with keys: symbol, expression, series, latest_value, count,
            and optionally error.
        """
        from app.config import get_settings
        from app.context import QlibContext
        from app.utils.symbol_mapping import normalize_symbol_for_qlib

        settings = get_settings()
        data_dir = data_dir or settings.QLIB_DATA_DIR

        # Validate first
        is_valid, error, _ = ExpressionEngine.validate(
            expression, settings.MAX_EXPRESSION_LENGTH
        )
        if not is_valid:
            return {
                "symbol": symbol,
                "expression": expression,
                "series": [],
                "latest_value": None,
                "count": 0,
                "error": error,
            }

        logger.info(
            "Evaluating expression for %s: %s", symbol, expression[:100]
        )

        try:
            QlibContext.ensure_init(market, data_dir)
        except Exception as e:
            logger.error("Qlib init failed: %s", e)
            return {
                "symbol": symbol,
                "expression": expression,
                "series": [],
                "latest_value": None,
                "count": 0,
                "error": "Qlib initialization failed for this market",
            }

        # Lazy imports -- qlib must be initialized first
        import pandas as pd
        from qlib.data import D

        end = end_date or pd.Timestamp.now().strftime("%Y-%m-%d")
        if start_date:
            start = start_date
        else:
            # Convert period to days
            lookback_days = ExpressionEngine.PERIOD_TO_DAYS.get(period, 90)
            start = (
                pd.Timestamp(end) - pd.Timedelta(days=lookback_days)
            ).strftime("%Y-%m-%d")

        qlib_symbol = normalize_symbol_for_qlib(symbol, market)

        try:
            df = D.features(
                instruments=[qlib_symbol],
                fields=[expression],
                start_time=start,
                end_time=end,
            )
        except Exception as e:
            logger.error("D.features() failed: %s", e)
            return {
                "symbol": symbol,
                "expression": expression,
                "series": [],
                "latest_value": None,
                "count": 0,
                "error": f"Expression evaluation failed: {e}",
            }

        if df.empty:
            return {
                "symbol": symbol,
                "expression": expression,
                "series": [],
                "latest_value": None,
                "count": 0,
            }

        # Flatten MultiIndex if needed (instrument, datetime) -> (datetime)
        if hasattr(df.index, "levels"):
            df = df.droplevel(0)

        df.columns = ["value"]
        series = []
        for dt, row in df.iterrows():
            v = row["value"]
            if pd.isna(v):
                continue
            try:
                fv = float(v)
                if not math.isfinite(fv):
                    continue
                series.append({
                    "date": dt.strftime("%Y-%m-%d"),
                    "value": round(fv, 6),
                })
            except (OverflowError, ValueError):
                continue

        latest_value = None
        if series:
            latest_value = series[-1]["value"]

        logger.info(
            "Expression result: %d data points, latest=%s",
            len(series),
            f"{latest_value:.6f}" if latest_value is not None else "N/A",
        )

        return {
            "symbol": symbol,
            "expression": expression,
            "series": series,
            "latest_value": latest_value,
            "count": len(series),
        }

    @staticmethod
    def evaluate_batch(
        symbols: List[str],
        expression: str,
        market: str,
        target_date: Optional[str] = None,
        data_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Evaluate expression across multiple symbols (cross-sectional).

        Synchronous -- runs in ThreadPoolExecutor via run_qlib_quick().
        Returns ExpressionBatchResult-compatible dict.

        Args:
            symbols: List of stock symbols.
            expression: Qlib expression.
            market: Market code.
            target_date: Target date (YYYY-MM-DD). Defaults to today.
            data_dir: Override Qlib data directory.

        Returns:
            Dict with keys: expression, results, date, and optionally error.
        """
        from app.config import get_settings
        from app.context import QlibContext
        from app.utils.symbol_mapping import normalize_symbol_for_qlib

        settings = get_settings()
        data_dir = data_dir or settings.QLIB_DATA_DIR

        is_valid, error, _ = ExpressionEngine.validate(
            expression, settings.MAX_EXPRESSION_LENGTH
        )
        if not is_valid:
            return {
                "expression": expression,
                "results": {},
                "date": None,
                "error": error,
            }

        logger.info(
            "Batch evaluating expression for %d symbols: %s",
            len(symbols),
            expression[:100],
        )

        try:
            QlibContext.ensure_init(market, data_dir)
        except Exception as e:
            logger.error("Qlib init failed: %s", e)
            return {
                "expression": expression,
                "results": {},
                "date": None,
                "error": "Qlib initialization failed for this market",
            }

        import pandas as pd
        from qlib.data import D

        qlib_symbols = [
            normalize_symbol_for_qlib(s, market) for s in symbols
        ]
        qlib_to_original = dict(zip(qlib_symbols, symbols))

        end = target_date or pd.Timestamp.now().strftime("%Y-%m-%d")
        # Fetch a few days of data to ensure we get at least one row
        start = (pd.Timestamp(end) - pd.Timedelta(days=5)).strftime("%Y-%m-%d")

        try:
            df = D.features(
                instruments=qlib_symbols,
                fields=[expression],
                start_time=start,
                end_time=end,
            )
        except Exception as e:
            logger.error("D.features() batch failed: %s", e)
            return {
                "expression": expression,
                "results": {},
                "date": None,
                "error": str(e),
            }

        if df.empty:
            return {
                "expression": expression,
                "results": {},
                "date": None,
            }

        df.columns = ["value"]

        # Get latest date cross-section
        if hasattr(df.index, "levels") and len(df.index.levels) == 2:
            dates = df.index.get_level_values(1).unique()
            latest_date = sorted(dates)[-1]
            cross_section = df.xs(latest_date, level=1)["value"]
        else:
            latest_date = df.index[-1]
            cross_section = df.iloc[-1:]
            cross_section = pd.Series(
                cross_section["value"].values,
                index=[qlib_symbols[0]] if qlib_symbols else [],
            )

        results = {}
        for qlib_sym, val in cross_section.items():
            original = qlib_to_original.get(qlib_sym, qlib_sym)
            if pd.isna(val):
                results[original] = None
            else:
                try:
                    fv = float(val)
                    results[original] = round(fv, 6) if math.isfinite(fv) else None
                except (OverflowError, ValueError):
                    results[original] = None

        result_date = (
            latest_date.strftime("%Y-%m-%d")
            if hasattr(latest_date, "strftime")
            else str(latest_date)
        )

        return {
            "expression": expression,
            "results": results,
            "date": result_date,
        }
