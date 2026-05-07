"""Simplified backtest engine using Qlib data infrastructure.

Implements portfolio simulation with configurable strategies (TopK, Signal,
LongShort), rebalancing frequency, slippage, and commission costs. Uses
Qlib D.features() for price data and factor scores, but does NOT depend on
Qlib's backtest module (which requires ML components).

All heavy computation runs in ProcessPoolExecutor via run_qlib_background().
Task state is tracked in-memory via BacktestTask dataclass. Progress updates
are written to the task object during execution so the API can poll status.

Strategy types:
- TopK: Buy top K stocks by factor score, drop N worst at each rebalance.
- Signal: Buy/sell based on expression threshold crossing.
- LongShort: Long top decile, short bottom decile by factor score.
"""
import asyncio
import json
import logging
import math
import threading
import time
import uuid
from concurrent.futures import Future
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _numpy_default(obj: Any) -> Any:
    """JSON serializer fallback for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if math.isnan(v) or math.isinf(v) else v
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# Default strategy configurations
DEFAULT_TOPK_CONFIG = {
    "k": 10,                   # Number of stocks to hold
    "n_drop": 2,               # Drop N worst before reselecting
    "score_expression": "($close-Ref($close,20))/Ref($close,20)",  # Momentum factor
    "rebalance_days": 5,       # Rebalance every N trading days
}

DEFAULT_SIGNAL_CONFIG = {
    "expression": "($close-Ref($close,5))/Ref($close,5)",
    "buy_threshold": 0.02,     # Buy when expression > threshold
    "sell_threshold": -0.02,   # Sell when expression < threshold
    "max_positions": 10,       # Maximum simultaneous positions
    "rebalance_days": 1,       # Check signal every N trading days
}

DEFAULT_LONG_SHORT_CONFIG = {
    "score_expression": "($close-Ref($close,20))/Ref($close,20)",
    "long_pct": 0.1,           # Top 10% long
    "short_pct": 0.1,          # Bottom 10% short
    "rebalance_days": 20,      # Monthly rebalance
}

DEFAULT_EXECUTION_CONFIG = {
    "slippage": 0.0005,        # 5 bps
    "commission": 0.0015,      # 15 bps
    "limit_threshold": None,   # None = no limit, 0.095 = A-share 10%
}


@dataclass
class BacktestTask:
    """In-memory representation of a backtest task."""
    task_id: str
    name: str
    config: dict
    status: str = "pending"         # pending, running, completed, failed, cancelled
    progress: int = 0               # 0-100
    current_date: Optional[str] = None
    current_return: Optional[float] = None
    results: Optional[dict] = None
    error: Optional[str] = None
    future: Optional[Future] = field(default=None, repr=False)
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        """Serialize to API-compatible dict (excludes future).

        Uses a JSON roundtrip to convert numpy types (float32, int64, etc.)
        to native Python types that Pydantic can serialize.
        """
        d = {
            "task_id": self.task_id,
            "name": self.name,
            "status": self.status,
            "progress": self.progress,
            "current_date": self.current_date,
            "current_return": self.current_return,
            "results": self.results,
            "error": self.error,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }
        # JSON roundtrip to convert numpy types to native Python
        return json.loads(json.dumps(d, default=_numpy_default))


class BacktestService:
    """Backtest task management and execution.

    Task lifecycle:
    1. create() -> stores BacktestTask, submits to ProcessPoolExecutor
    2. _run_backtest() -> executes in subprocess, updates task state
    3. get_status() / list_tasks() -> polls current state
    4. cancel() / delete() -> lifecycle management
    """

    _tasks: Dict[str, BacktestTask] = {}
    _lock = threading.Lock()

    @classmethod
    async def create(cls, config: dict) -> str:
        """Create a backtest task and submit to ProcessPoolExecutor.

        Args:
            config: Full backtest configuration dict (from BacktestCreateRequest).

        Returns:
            task_id string.

        Raises:
            RuntimeError: If max concurrent backtests exceeded.
        """
        import asyncio
        from app.config import get_settings
        settings = get_settings()

        with cls._lock:
            running_count = sum(
                1 for t in cls._tasks.values() if t.status in ("pending", "running")
            )
            if running_count >= settings.MAX_CONCURRENT_BACKTESTS:
                raise RuntimeError(
                    f"Maximum concurrent backtests ({settings.MAX_CONCURRENT_BACKTESTS}) reached. "
                    f"Wait for existing tasks to complete or cancel them."
                )

            task_id = uuid.uuid4().hex[:16]
            task = BacktestTask(
                task_id=task_id,
                name=config.get("name", f"backtest-{task_id[:8]}"),
                config=config,
            )
            cls._tasks[task_id] = task

        logger.info(
            "Backtest task created: task_id=%s, name=%s, strategy=%s, symbols=%d",
            task_id, task.name, config.get("strategy_type", "topk"),
            len(config.get("symbols", [])),
        )

        # Submit to ProcessPoolExecutor via run_qlib_background
        asyncio.create_task(cls._submit_task(task_id, config))

        return task_id

    @classmethod
    async def _submit_task(cls, task_id: str, config: dict) -> None:
        """Submit backtest to ProcessPoolExecutor and handle completion."""
        from app.executor import run_qlib_background
        from app.config import get_settings

        settings = get_settings()

        task = cls._tasks.get(task_id)
        if task is None:
            return

        try:
            task.status = "running"
            task.progress = 0

            result = await run_qlib_background(
                _execute_backtest,
                task_id,
                config,
                timeout=settings.BACKTEST_TIMEOUT_SECONDS,
            )

            with cls._lock:
                task = cls._tasks.get(task_id)
                if task is None:
                    return
                if task.status == "cancelled":
                    return

                task.status = result.get("status", "completed")
                task.progress = 100
                task.results = result.get("results")
                task.error = result.get("error")
                task.completed_at = datetime.now()

                if task.error:
                    task.status = "failed"

            logger.info(
                "Backtest task completed: task_id=%s, status=%s",
                task_id, task.status,
            )
        except asyncio.TimeoutError:
            with cls._lock:
                task = cls._tasks.get(task_id)
                if task:
                    task.status = "failed"
                    task.error = "Backtest timed out"
                    task.completed_at = datetime.now()
            logger.error("Backtest task timed out: task_id=%s", task_id)
        except Exception as e:
            with cls._lock:
                task = cls._tasks.get(task_id)
                if task and task.status != "cancelled":
                    task.status = "failed"
                    task.error = str(e)
                    task.completed_at = datetime.now()
            logger.error(
                "Backtest task failed: task_id=%s, error=%s", task_id, e, exc_info=True
            )

    @classmethod
    def get_status(cls, task_id: str) -> Optional[dict]:
        """Return task status dict, or None if not found."""
        with cls._lock:
            task = cls._tasks.get(task_id)
            if task is None:
                return None
            return task.to_dict()

    @classmethod
    def cancel(cls, task_id: str) -> bool:
        """Cancel a running or pending backtest.

        Returns True if the task was found and marked cancelled.
        """
        with cls._lock:
            task = cls._tasks.get(task_id)
            if task is None:
                return False
            if task.status not in ("pending", "running"):
                return False

            task.status = "cancelled"
            task.completed_at = datetime.now()

            if task.future is not None:
                task.future.cancel()

        logger.info("Backtest task cancelled: task_id=%s", task_id)
        return True

    @classmethod
    def delete(cls, task_id: str) -> bool:
        """Delete a completed, failed, or cancelled backtest.

        Returns True if the task was found and deleted.
        """
        with cls._lock:
            task = cls._tasks.get(task_id)
            if task is None:
                return False
            if task.status in ("pending", "running"):
                return False

            del cls._tasks[task_id]

        logger.info("Backtest task deleted: task_id=%s", task_id)
        return True

    @classmethod
    def list_tasks(cls) -> List[dict]:
        """List all backtest tasks, newest first."""
        with cls._lock:
            tasks = sorted(
                cls._tasks.values(),
                key=lambda t: t.created_at,
                reverse=True,
            )
            return [t.to_dict() for t in tasks]


# ------------------------------------------------------------------ #
# Backtest execution (runs in ProcessPoolExecutor subprocess)
# ------------------------------------------------------------------ #


def _execute_backtest(task_id: str, config: dict) -> dict:
    """Execute a backtest in a subprocess.

    This function runs in ProcessPoolExecutor -- it has its own Qlib init
    and cannot share state with the main process.

    Args:
        task_id: Task identifier (for logging).
        config: Full backtest configuration.

    Returns:
        Dict with "status", "results", and optionally "error".
    """
    import pandas as pd

    start_time = time.monotonic()
    logger.info("Backtest execution started: task_id=%s", task_id)

    try:
        # 1. Initialize Qlib for the target market
        from app.context import QlibContext
        from app.config import get_settings
        from app.utils.symbol_mapping import normalize_symbol_for_qlib

        settings = get_settings()
        market = config.get("market", "us")
        QlibContext.ensure_init(market, settings.QLIB_DATA_DIR)

        from qlib.data import D

        # 2. Parse configuration
        symbols = config["symbols"]
        start_date = config["start_date"]
        end_date = config["end_date"]
        strategy_type = config.get("strategy_type", "topk")
        strategy_config = config.get("strategy_config", {})
        execution_config = {**DEFAULT_EXECUTION_CONFIG, **config.get("execution_config", {})}

        qlib_symbols = [normalize_symbol_for_qlib(s, market) for s in symbols]
        qlib_to_original = dict(zip(qlib_symbols, symbols))

        # 3. Fetch price data ($close, $open, $high, $low, $volume)
        logger.info(
            "Fetching price data: %d symbols, %s to %s",
            len(qlib_symbols), start_date, end_date,
        )

        price_df = D.features(
            instruments=qlib_symbols,
            fields=["$close", "$open", "$high", "$low"],
            start_time=str(start_date),
            end_time=str(end_date),
        )

        if price_df.empty:
            return {
                "status": "failed",
                "results": None,
                "error": "No price data available for the specified symbols and date range",
            }

        price_df.columns = ["close", "open", "high", "low"]

        # 4. Fetch factor scores if needed
        score_expression = _get_score_expression(strategy_type, strategy_config)
        score_df = None
        if score_expression:
            logger.info("Computing factor scores: %s", score_expression[:100])
            try:
                score_df = D.features(
                    instruments=qlib_symbols,
                    fields=[score_expression],
                    start_time=str(start_date),
                    end_time=str(end_date),
                )
                if not score_df.empty:
                    score_df.columns = ["score"]
            except Exception as e:
                logger.warning("Factor score computation failed: %s", e)
                score_df = None

        # 5. Build daily close price matrix (instruments x dates)
        close_matrix = _build_price_matrix(price_df, "close", qlib_symbols)
        open_matrix = _build_price_matrix(price_df, "open", qlib_symbols)

        if close_matrix.empty:
            return {
                "status": "failed",
                "results": None,
                "error": "Insufficient price data after processing",
            }

        trading_dates = close_matrix.index.tolist()
        logger.info(
            "Price matrix: %d dates x %d symbols",
            len(trading_dates), len(close_matrix.columns),
        )

        # 6. Build score matrix if available
        score_matrix = None
        if score_df is not None and not score_df.empty:
            score_matrix = _build_price_matrix(score_df, "score", qlib_symbols)

        # 7. Run strategy simulation
        slippage = float(execution_config.get("slippage", 0.0005))
        commission = float(execution_config.get("commission", 0.0015))
        limit_threshold = execution_config.get("limit_threshold")

        logger.info(
            "Running %s strategy simulation: slippage=%.4f, commission=%.4f",
            strategy_type, slippage, commission,
        )

        if strategy_type == "topk":
            sim_result = _simulate_topk(
                close_matrix, open_matrix, score_matrix,
                strategy_config, slippage, commission, limit_threshold,
            )
        elif strategy_type == "signal":
            sim_result = _simulate_signal(
                close_matrix, open_matrix, score_matrix,
                strategy_config, slippage, commission, limit_threshold,
            )
        elif strategy_type == "long_short":
            sim_result = _simulate_long_short(
                close_matrix, open_matrix, score_matrix,
                strategy_config, slippage, commission, limit_threshold,
            )
        else:
            return {
                "status": "failed",
                "results": None,
                "error": f"Unknown strategy type: {strategy_type}",
            }

        # 8. Compute risk metrics
        equity_curve = sim_result["equity_curve"]
        trades = sim_result["trades"]
        turnover_rates = sim_result.get("turnover_rates", [])

        metrics = _compute_risk_metrics(equity_curve, trades, turnover_rates)

        # 9. Map symbols back to original format
        mapped_trades = []
        for trade in trades:
            t = dict(trade)
            t["symbol"] = qlib_to_original.get(t.get("symbol", ""), t.get("symbol", ""))
            mapped_trades.append(t)

        elapsed = time.monotonic() - start_time
        logger.info(
            "Backtest completed: task_id=%s, total_return=%.4f, sharpe=%.4f, duration=%.1fs",
            task_id,
            metrics.get("total_return", 0),
            metrics.get("sharpe_ratio", 0),
            elapsed,
        )

        results = {
            **metrics,
            "strategy_type": strategy_type,
            "symbol_count": len(symbols),
            "date_range": {
                "start": trading_dates[0].strftime("%Y-%m-%d") if trading_dates else None,
                "end": trading_dates[-1].strftime("%Y-%m-%d") if trading_dates else None,
            },
            "total_trades": len(trades),
            "trades": mapped_trades[:500],  # Limit trade list for API response
            "execution_config": execution_config,
            "duration_s": round(elapsed, 2),
        }

        return {"status": "completed", "results": results}

    except Exception as e:
        elapsed = time.monotonic() - start_time
        logger.error(
            "Backtest execution failed: task_id=%s, error=%s, duration=%.1fs",
            task_id, e, elapsed, exc_info=True,
        )
        return {"status": "failed", "results": None, "error": str(e)}


# ------------------------------------------------------------------ #
# Helper functions
# ------------------------------------------------------------------ #


def _get_score_expression(strategy_type: str, strategy_config: dict) -> Optional[str]:
    """Extract the scoring expression from strategy config."""
    if strategy_type == "topk":
        defaults = DEFAULT_TOPK_CONFIG
        return strategy_config.get("score_expression", defaults["score_expression"])
    elif strategy_type == "signal":
        defaults = DEFAULT_SIGNAL_CONFIG
        return strategy_config.get("expression", defaults["expression"])
    elif strategy_type == "long_short":
        defaults = DEFAULT_LONG_SHORT_CONFIG
        return strategy_config.get("score_expression", defaults["score_expression"])
    return None


def _build_price_matrix(
    feature_df: "pd.DataFrame",
    column: str,
    qlib_symbols: List[str],
) -> "pd.DataFrame":
    """Build a (dates x symbols) matrix from a Qlib multi-index DataFrame.

    Args:
        feature_df: D.features() result with MultiIndex (instrument, datetime).
        column: Column name to extract.
        qlib_symbols: Expected symbol list (for consistent column ordering).

    Returns:
        DataFrame indexed by date with symbol columns.
    """
    import pandas as pd

    if feature_df.empty:
        return pd.DataFrame()

    if not hasattr(feature_df.index, "levels") or len(feature_df.index.names) < 2:
        # Single instrument -- convert to matrix format
        series = feature_df[column].dropna()
        if series.empty:
            return pd.DataFrame()
        sym = qlib_symbols[0] if qlib_symbols else "UNKNOWN"
        return pd.DataFrame({sym: series})

    # Multi-instrument: unstack instrument level
    try:
        matrix = feature_df[column].unstack(level=0)
    except Exception:
        # Fallback: build manually
        matrix = pd.DataFrame()
        for sym in qlib_symbols:
            try:
                sym_data = feature_df.xs(sym, level=0)[column]
                matrix[sym] = sym_data
            except (KeyError, TypeError):
                continue

    # Drop dates where all symbols are NaN
    matrix = matrix.dropna(how="all")

    return matrix


def _apply_trade_cost(
    price: float,
    direction: str,
    slippage: float,
    commission: float,
) -> float:
    """Apply slippage and commission to a trade price.

    Args:
        price: Raw execution price.
        direction: "buy" or "sell".
        slippage: Slippage rate (e.g., 0.0005 = 5 bps).
        commission: Commission rate (e.g., 0.0015 = 15 bps).

    Returns:
        Effective execution price after costs.
    """
    if direction == "buy":
        return price * (1.0 + slippage + commission)
    else:
        return price * (1.0 - slippage - commission)


def _check_limit(
    prev_close: float,
    current_price: float,
    limit_threshold: Optional[float],
) -> bool:
    """Check if a stock has hit its daily limit.

    Returns True if the stock is limit-up or limit-down (untradeable).
    """
    if limit_threshold is None or prev_close <= 0:
        return False
    change = abs(current_price - prev_close) / prev_close
    return change >= limit_threshold


# ------------------------------------------------------------------ #
# Strategy simulation functions
# ------------------------------------------------------------------ #


def _simulate_topk(
    close_matrix: "pd.DataFrame",
    open_matrix: "pd.DataFrame",
    score_matrix: Optional["pd.DataFrame"],
    strategy_config: dict,
    slippage: float,
    commission: float,
    limit_threshold: Optional[float],
) -> dict:
    """Simulate a TopK portfolio strategy.

    At each rebalance point:
    1. Rank all stocks by score (descending).
    2. If currently holding, drop the N worst performers.
    3. Fill empty slots from top-ranked stocks not already held.
    4. Equal-weight the portfolio.

    Returns dict with equity_curve, trades, turnover_rates.
    """
    import pandas as pd
    import numpy as np

    defaults = DEFAULT_TOPK_CONFIG
    k = int(strategy_config.get("k", defaults["k"]))
    n_drop = int(strategy_config.get("n_drop", defaults["n_drop"]))
    rebalance_days = int(strategy_config.get("rebalance_days", defaults["rebalance_days"]))

    # Clamp n_drop so the strategy can actually rotate when k is small.
    # If n_drop >= k, every rebalance should re-select from scratch.
    if n_drop >= k:
        n_drop = k

    dates = close_matrix.index.tolist()
    symbols = close_matrix.columns.tolist()

    # Initialize portfolio state
    cash = 1.0
    positions: Dict[str, float] = {}  # symbol -> number of shares (fractional)
    equity_curve: List[dict] = []
    trades: List[dict] = []
    turnover_rates: List[float] = []
    days_since_rebalance = rebalance_days  # Force rebalance on first day

    for i, dt in enumerate(dates):
        close_prices = close_matrix.loc[dt]
        open_prices = open_matrix.loc[dt] if dt in open_matrix.index else close_prices

        # Calculate current portfolio value
        portfolio_value = cash
        for sym, shares in positions.items():
            price = close_prices.get(sym)
            if price is not None and not pd.isna(price) and price > 0:
                portfolio_value += shares * price

        # Record equity
        equity_curve.append({
            "date": dt.strftime("%Y-%m-%d"),
            "value": round(portfolio_value, 6),
        })

        days_since_rebalance += 1

        # Rebalance check
        if days_since_rebalance < rebalance_days:
            continue
        if score_matrix is None or dt not in score_matrix.index:
            continue

        days_since_rebalance = 0
        scores = score_matrix.loc[dt].dropna()

        if scores.empty:
            continue

        # Filter to tradeable symbols (have valid close price)
        tradeable = [
            s for s in scores.index
            if s in close_prices.index
            and not pd.isna(close_prices[s])
            and close_prices[s] > 0
        ]
        scores = scores[tradeable].sort_values(ascending=False)

        # Check limit for current positions
        if limit_threshold is not None and i > 0:
            prev_close = close_matrix.iloc[i - 1]
            tradeable = [
                s for s in tradeable
                if not _check_limit(
                    prev_close.get(s, 0), close_prices.get(s, 0), limit_threshold
                )
            ]
            scores = scores[scores.index.isin(tradeable)]

        # Determine target portfolio
        current_held = set(positions.keys())

        if current_held and n_drop < k:
            # Incremental rotation: drop N worst, fill from top-ranked newcomers
            held_scores = scores[scores.index.isin(current_held)]
            non_held_scores = scores[~scores.index.isin(current_held)]

            if len(held_scores) > n_drop:
                drop_symbols = held_scores.nsmallest(n_drop).index.tolist()
            else:
                drop_symbols = []

            keep_symbols = [s for s in held_scores.index if s not in drop_symbols]
            slots_available = k - len(keep_symbols)

            new_symbols = non_held_scores.head(max(0, slots_available)).index.tolist()
            target_symbols = keep_symbols + new_symbols
        else:
            # Full re-selection: pick the top-k from the entire pool
            target_symbols = scores.head(k).index.tolist()

        if not target_symbols:
            continue

        # Execute rebalance: equal weight
        target_weight = 1.0 / len(target_symbols)
        old_value = portfolio_value
        turnover = 0.0

        # Sell positions not in target
        symbols_to_sell = [s for s in list(positions.keys()) if s not in target_symbols]
        for sym in symbols_to_sell:
            shares = positions.pop(sym)
            exec_price = open_prices.get(sym, close_prices.get(sym))
            if exec_price is None or pd.isna(exec_price) or exec_price <= 0:
                exec_price = close_prices.get(sym, 0)
            if exec_price > 0:
                sell_price = _apply_trade_cost(exec_price, "sell", slippage, commission)
                proceeds = shares * sell_price
                cash += proceeds
                turnover += abs(proceeds)
                trades.append({
                    "date": dt.strftime("%Y-%m-%d"),
                    "symbol": sym,
                    "direction": "sell",
                    "shares": round(shares, 6),
                    "price": round(sell_price, 4),
                    "value": round(proceeds, 4),
                })

        # Recalculate portfolio value after sells
        portfolio_value = cash
        for sym, shares in positions.items():
            price = close_prices.get(sym)
            if price is not None and not pd.isna(price) and price > 0:
                portfolio_value += shares * price

        # Buy / adjust positions to target weight
        for sym in target_symbols:
            target_value = portfolio_value * target_weight
            exec_price = open_prices.get(sym, close_prices.get(sym))
            if exec_price is None or pd.isna(exec_price) or exec_price <= 0:
                continue

            current_shares = positions.get(sym, 0.0)
            current_value = current_shares * exec_price
            diff_value = target_value - current_value

            if abs(diff_value) < 1e-6:
                continue

            if diff_value > 0:
                # Buy
                buy_price = _apply_trade_cost(exec_price, "buy", slippage, commission)
                shares_to_buy = diff_value / buy_price
                cost = shares_to_buy * buy_price

                if cost > cash:
                    shares_to_buy = cash / buy_price
                    cost = shares_to_buy * buy_price

                if shares_to_buy > 1e-8:
                    cash -= cost
                    positions[sym] = current_shares + shares_to_buy
                    turnover += abs(cost)
                    trades.append({
                        "date": dt.strftime("%Y-%m-%d"),
                        "symbol": sym,
                        "direction": "buy",
                        "shares": round(shares_to_buy, 6),
                        "price": round(buy_price, 4),
                        "value": round(cost, 4),
                    })
            else:
                # Sell excess
                sell_price = _apply_trade_cost(exec_price, "sell", slippage, commission)
                shares_to_sell = abs(diff_value) / sell_price
                shares_to_sell = min(shares_to_sell, current_shares)

                if shares_to_sell > 1e-8:
                    proceeds = shares_to_sell * sell_price
                    cash += proceeds
                    positions[sym] = current_shares - shares_to_sell
                    turnover += abs(proceeds)
                    trades.append({
                        "date": dt.strftime("%Y-%m-%d"),
                        "symbol": sym,
                        "direction": "sell",
                        "shares": round(shares_to_sell, 6),
                        "price": round(sell_price, 4),
                        "value": round(proceeds, 4),
                    })

        # Clean up zero positions
        positions = {s: sh for s, sh in positions.items() if sh > 1e-8}

        if old_value > 0:
            turnover_rates.append(turnover / old_value)

    return {
        "equity_curve": equity_curve,
        "trades": trades,
        "turnover_rates": turnover_rates,
    }


def _simulate_signal(
    close_matrix: "pd.DataFrame",
    open_matrix: "pd.DataFrame",
    score_matrix: Optional["pd.DataFrame"],
    strategy_config: dict,
    slippage: float,
    commission: float,
    limit_threshold: Optional[float],
) -> dict:
    """Simulate a signal-based strategy.

    At each rebalance point:
    1. Compute signal (expression value) for each stock.
    2. Buy stocks with signal > buy_threshold (up to max_positions).
    3. Sell stocks with signal < sell_threshold.

    Returns dict with equity_curve, trades, turnover_rates.
    """
    import pandas as pd

    defaults = DEFAULT_SIGNAL_CONFIG
    buy_threshold = float(strategy_config.get("buy_threshold", defaults["buy_threshold"]))
    sell_threshold = float(strategy_config.get("sell_threshold", defaults["sell_threshold"]))
    max_positions = int(strategy_config.get("max_positions", defaults["max_positions"]))
    rebalance_days = int(strategy_config.get("rebalance_days", defaults["rebalance_days"]))

    dates = close_matrix.index.tolist()

    cash = 1.0
    positions: Dict[str, float] = {}  # symbol -> shares
    equity_curve: List[dict] = []
    trades: List[dict] = []
    turnover_rates: List[float] = []
    days_since_rebalance = rebalance_days

    for i, dt in enumerate(dates):
        close_prices = close_matrix.loc[dt]
        open_prices = open_matrix.loc[dt] if dt in open_matrix.index else close_prices

        # Calculate portfolio value
        portfolio_value = cash
        for sym, shares in positions.items():
            price = close_prices.get(sym)
            if price is not None and not pd.isna(price) and price > 0:
                portfolio_value += shares * price

        equity_curve.append({
            "date": dt.strftime("%Y-%m-%d"),
            "value": round(portfolio_value, 6),
        })

        days_since_rebalance += 1
        if days_since_rebalance < rebalance_days:
            continue
        if score_matrix is None or dt not in score_matrix.index:
            continue

        days_since_rebalance = 0
        signals = score_matrix.loc[dt].dropna()
        if signals.empty:
            continue

        old_value = portfolio_value
        turnover = 0.0

        # Sell positions with signal below threshold
        symbols_to_sell = [
            sym for sym in list(positions.keys())
            if sym in signals.index and signals[sym] < sell_threshold
        ]
        for sym in symbols_to_sell:
            shares = positions.pop(sym)
            exec_price = open_prices.get(sym, close_prices.get(sym))
            if exec_price is None or pd.isna(exec_price) or exec_price <= 0:
                continue
            sell_price = _apply_trade_cost(exec_price, "sell", slippage, commission)
            proceeds = shares * sell_price
            cash += proceeds
            turnover += abs(proceeds)
            trades.append({
                "date": dt.strftime("%Y-%m-%d"),
                "symbol": sym,
                "direction": "sell",
                "shares": round(shares, 6),
                "price": round(sell_price, 4),
                "value": round(proceeds, 4),
            })

        # Buy signals above threshold
        buy_candidates = signals[signals > buy_threshold].sort_values(ascending=False)
        slots_available = max_positions - len(positions)

        if slots_available > 0 and len(buy_candidates) > 0:
            candidates = buy_candidates.head(slots_available)
            weight_per_position = 1.0 / max_positions

            # Recalculate portfolio value
            portfolio_value = cash
            for sym, shares in positions.items():
                price = close_prices.get(sym)
                if price is not None and not pd.isna(price) and price > 0:
                    portfolio_value += shares * price

            for sym in candidates.index:
                if sym in positions:
                    continue

                exec_price = open_prices.get(sym, close_prices.get(sym))
                if exec_price is None or pd.isna(exec_price) or exec_price <= 0:
                    continue

                if limit_threshold is not None and i > 0:
                    prev_close = close_matrix.iloc[i - 1]
                    if _check_limit(
                        prev_close.get(sym, 0), close_prices.get(sym, 0), limit_threshold
                    ):
                        continue

                target_value = portfolio_value * weight_per_position
                buy_price = _apply_trade_cost(exec_price, "buy", slippage, commission)
                shares_to_buy = target_value / buy_price

                cost = shares_to_buy * buy_price
                if cost > cash:
                    shares_to_buy = cash / buy_price
                    cost = shares_to_buy * buy_price

                if shares_to_buy > 1e-8:
                    cash -= cost
                    positions[sym] = shares_to_buy
                    turnover += abs(cost)
                    trades.append({
                        "date": dt.strftime("%Y-%m-%d"),
                        "symbol": sym,
                        "direction": "buy",
                        "shares": round(shares_to_buy, 6),
                        "price": round(buy_price, 4),
                        "value": round(cost, 4),
                    })

        positions = {s: sh for s, sh in positions.items() if sh > 1e-8}

        if old_value > 0:
            turnover_rates.append(turnover / old_value)

    return {
        "equity_curve": equity_curve,
        "trades": trades,
        "turnover_rates": turnover_rates,
    }


def _simulate_long_short(
    close_matrix: "pd.DataFrame",
    open_matrix: "pd.DataFrame",
    score_matrix: Optional["pd.DataFrame"],
    strategy_config: dict,
    slippage: float,
    commission: float,
    limit_threshold: Optional[float],
) -> dict:
    """Simulate a long-short strategy.

    At each rebalance point:
    1. Rank all stocks by score.
    2. Long the top decile, short the bottom decile.
    3. Equal weight within each leg.
    4. Portfolio value = long_pnl - short_pnl + initial_capital.

    Returns dict with equity_curve, trades, turnover_rates.
    """
    import pandas as pd

    defaults = DEFAULT_LONG_SHORT_CONFIG
    long_pct = float(strategy_config.get("long_pct", defaults["long_pct"]))
    short_pct = float(strategy_config.get("short_pct", defaults["short_pct"]))
    rebalance_days = int(strategy_config.get("rebalance_days", defaults["rebalance_days"]))

    dates = close_matrix.index.tolist()

    # Track long and short legs separately
    # Cash is split: 50% for long, 50% for short collateral
    long_cash = 0.5
    short_cash = 0.5
    long_positions: Dict[str, float] = {}   # symbol -> shares (positive)
    short_positions: Dict[str, float] = {}  # symbol -> shares (positive = shares sold short)
    short_entry_prices: Dict[str, float] = {}  # track entry price for P&L

    equity_curve: List[dict] = []
    trades: List[dict] = []
    turnover_rates: List[float] = []
    days_since_rebalance = rebalance_days

    for i, dt in enumerate(dates):
        close_prices = close_matrix.loc[dt]
        open_prices = open_matrix.loc[dt] if dt in open_matrix.index else close_prices

        # Calculate portfolio value
        long_value = long_cash
        for sym, shares in long_positions.items():
            price = close_prices.get(sym)
            if price is not None and not pd.isna(price) and price > 0:
                long_value += shares * price

        short_value = short_cash
        for sym, shares in short_positions.items():
            price = close_prices.get(sym)
            entry = short_entry_prices.get(sym, price)
            if price is not None and not pd.isna(price) and price > 0 and entry is not None:
                # Short P&L: profit when price falls
                short_value += shares * (entry - price)

        portfolio_value = long_value + short_value

        equity_curve.append({
            "date": dt.strftime("%Y-%m-%d"),
            "value": round(portfolio_value, 6),
        })

        days_since_rebalance += 1
        if days_since_rebalance < rebalance_days:
            continue
        if score_matrix is None or dt not in score_matrix.index:
            continue

        days_since_rebalance = 0
        scores = score_matrix.loc[dt].dropna()
        if len(scores) < 3:
            continue

        scores = scores.sort_values(ascending=False)
        n_long = max(1, int(len(scores) * long_pct))
        n_short = max(1, int(len(scores) * short_pct))

        long_targets = set(scores.head(n_long).index.tolist())
        short_targets = set(scores.tail(n_short).index.tolist())

        old_value = portfolio_value
        turnover = 0.0

        # Close long positions not in targets
        for sym in list(long_positions.keys()):
            if sym not in long_targets:
                shares = long_positions.pop(sym)
                exec_price = open_prices.get(sym, close_prices.get(sym))
                if exec_price is not None and not pd.isna(exec_price) and exec_price > 0:
                    sell_price = _apply_trade_cost(exec_price, "sell", slippage, commission)
                    proceeds = shares * sell_price
                    long_cash += proceeds
                    turnover += abs(proceeds)
                    trades.append({
                        "date": dt.strftime("%Y-%m-%d"),
                        "symbol": sym,
                        "direction": "sell",
                        "leg": "long",
                        "shares": round(shares, 6),
                        "price": round(sell_price, 4),
                        "value": round(proceeds, 4),
                    })

        # Close short positions not in targets
        for sym in list(short_positions.keys()):
            if sym not in short_targets:
                shares = short_positions.pop(sym)
                entry = short_entry_prices.pop(sym, 0)
                exec_price = open_prices.get(sym, close_prices.get(sym))
                if exec_price is not None and not pd.isna(exec_price) and exec_price > 0:
                    cover_price = _apply_trade_cost(exec_price, "buy", slippage, commission)
                    pnl = shares * (entry - cover_price)
                    short_cash += pnl
                    turnover += abs(shares * cover_price)
                    trades.append({
                        "date": dt.strftime("%Y-%m-%d"),
                        "symbol": sym,
                        "direction": "cover",
                        "leg": "short",
                        "shares": round(shares, 6),
                        "price": round(cover_price, 4),
                        "value": round(abs(shares * cover_price), 4),
                    })

        # Recalculate long value
        long_value = long_cash
        for sym, shares in long_positions.items():
            price = close_prices.get(sym)
            if price is not None and not pd.isna(price) and price > 0:
                long_value += shares * price

        # Open new long positions
        if long_targets:
            long_weight = 1.0 / len(long_targets)
            for sym in long_targets:
                if sym in long_positions:
                    continue
                exec_price = open_prices.get(sym, close_prices.get(sym))
                if exec_price is None or pd.isna(exec_price) or exec_price <= 0:
                    continue

                target_value = long_value * long_weight
                buy_price = _apply_trade_cost(exec_price, "buy", slippage, commission)
                shares_to_buy = target_value / buy_price
                cost = shares_to_buy * buy_price

                if cost > long_cash:
                    shares_to_buy = long_cash / buy_price
                    cost = shares_to_buy * buy_price

                if shares_to_buy > 1e-8:
                    long_cash -= cost
                    long_positions[sym] = shares_to_buy
                    turnover += abs(cost)
                    trades.append({
                        "date": dt.strftime("%Y-%m-%d"),
                        "symbol": sym,
                        "direction": "buy",
                        "leg": "long",
                        "shares": round(shares_to_buy, 6),
                        "price": round(buy_price, 4),
                        "value": round(cost, 4),
                    })

        # Open new short positions
        if short_targets:
            # Use fixed allocation for short leg
            short_allocation = 0.5  # 50% of initial capital
            short_weight = short_allocation / len(short_targets)
            for sym in short_targets:
                if sym in short_positions:
                    continue
                exec_price = open_prices.get(sym, close_prices.get(sym))
                if exec_price is None or pd.isna(exec_price) or exec_price <= 0:
                    continue

                sell_price = _apply_trade_cost(exec_price, "sell", slippage, commission)
                shares_to_short = short_weight / sell_price

                if shares_to_short > 1e-8:
                    short_positions[sym] = shares_to_short
                    short_entry_prices[sym] = sell_price
                    turnover += abs(shares_to_short * sell_price)
                    trades.append({
                        "date": dt.strftime("%Y-%m-%d"),
                        "symbol": sym,
                        "direction": "short",
                        "leg": "short",
                        "shares": round(shares_to_short, 6),
                        "price": round(sell_price, 4),
                        "value": round(shares_to_short * sell_price, 4),
                    })

        # Clean up
        long_positions = {s: sh for s, sh in long_positions.items() if sh > 1e-8}
        short_positions = {s: sh for s, sh in short_positions.items() if sh > 1e-8}

        if old_value > 0:
            turnover_rates.append(turnover / old_value)

    return {
        "equity_curve": equity_curve,
        "trades": trades,
        "turnover_rates": turnover_rates,
    }


# ------------------------------------------------------------------ #
# Risk metrics computation
# ------------------------------------------------------------------ #


def _compute_risk_metrics(
    equity_curve: List[dict],
    trades: List[dict],
    turnover_rates: List[float],
) -> dict:
    """Compute comprehensive risk and performance metrics from equity curve.

    Args:
        equity_curve: List of {date, value} dicts.
        trades: List of trade dicts.
        turnover_rates: List of per-rebalance turnover ratios.

    Returns:
        Dict with all risk metrics.
    """
    import numpy as np

    if not equity_curve or len(equity_curve) < 2:
        return _empty_metrics(equity_curve)

    values = [p["value"] for p in equity_curve]
    values_arr = np.array(values, dtype=np.float64)

    # Guard against zero/negative initial value
    initial_value = values_arr[0]
    if initial_value <= 0:
        return _empty_metrics(equity_curve)

    # Total return
    total_return = (values_arr[-1] - initial_value) / initial_value

    # Daily returns
    daily_returns = np.diff(values_arr) / values_arr[:-1]
    # Filter out infinities from potential zero-division
    daily_returns = daily_returns[np.isfinite(daily_returns)]

    if len(daily_returns) == 0:
        return _empty_metrics(equity_curve)

    # Annualization factor (252 trading days)
    trading_days = len(daily_returns)
    years = trading_days / 252.0

    # Annual return (CAGR)
    if years > 0 and (values_arr[-1] / initial_value) > 0:
        annual_return = (values_arr[-1] / initial_value) ** (1.0 / years) - 1.0
    else:
        annual_return = 0.0

    # Volatility
    daily_vol = float(np.std(daily_returns, ddof=1)) if len(daily_returns) > 1 else 0.0
    annual_volatility = daily_vol * np.sqrt(252)

    # Sharpe ratio (assuming 0% risk-free rate)
    mean_daily_return = float(np.mean(daily_returns))
    if daily_vol > 1e-10:
        sharpe_ratio = (mean_daily_return / daily_vol) * np.sqrt(252)
    else:
        sharpe_ratio = 0.0

    # Maximum drawdown
    running_max = np.maximum.accumulate(values_arr)
    drawdowns = (values_arr - running_max) / running_max
    max_drawdown = float(np.min(drawdowns))

    # Maximum drawdown period
    max_dd_end_idx = int(np.argmin(drawdowns))
    max_dd_start_idx = int(np.argmax(values_arr[:max_dd_end_idx + 1])) if max_dd_end_idx > 0 else 0

    max_drawdown_period = {
        "start": equity_curve[max_dd_start_idx]["date"],
        "end": equity_curve[max_dd_end_idx]["date"],
    }

    # Calmar ratio
    if abs(max_drawdown) > 1e-10:
        calmar_ratio = annual_return / abs(max_drawdown)
    else:
        calmar_ratio = 0.0

    # Win rate and profit/loss ratio
    winning_days = daily_returns[daily_returns > 0]
    losing_days = daily_returns[daily_returns < 0]
    win_rate = len(winning_days) / len(daily_returns) if len(daily_returns) > 0 else 0.0

    avg_win = float(np.mean(winning_days)) if len(winning_days) > 0 else 0.0
    avg_loss = float(np.mean(np.abs(losing_days))) if len(losing_days) > 0 else 0.0
    profit_loss_ratio = avg_win / avg_loss if avg_loss > 1e-10 else 0.0

    # Turnover rate
    avg_turnover = float(np.mean(turnover_rates)) if turnover_rates else 0.0

    # Sortino ratio (downside deviation)
    downside_returns = daily_returns[daily_returns < 0]
    if len(downside_returns) > 1:
        downside_dev = float(np.std(downside_returns, ddof=1)) * np.sqrt(252)
        sortino_ratio = annual_return / downside_dev if downside_dev > 1e-10 else 0.0
    else:
        sortino_ratio = 0.0

    # Round all metrics
    def _round(v: float, decimals: int = 6) -> float:
        if not math.isfinite(v):
            return 0.0
        return round(v, decimals)

    return {
        "equity_curve": equity_curve,
        "total_return": _round(total_return, 6),
        "annual_return": _round(annual_return, 6),
        "sharpe_ratio": _round(sharpe_ratio, 4),
        "sortino_ratio": _round(sortino_ratio, 4),
        "max_drawdown": _round(max_drawdown, 6),
        "max_drawdown_period": max_drawdown_period,
        "annual_volatility": _round(annual_volatility, 6),
        "calmar_ratio": _round(calmar_ratio, 4),
        "turnover_rate": _round(avg_turnover, 4),
        "win_rate": _round(win_rate, 4),
        "profit_loss_ratio": _round(profit_loss_ratio, 4),
        "total_trades": len(trades),
        "trading_days": trading_days,
    }


def _empty_metrics(equity_curve: List[dict]) -> dict:
    """Return zeroed-out metrics when computation is not possible."""
    return {
        "equity_curve": equity_curve,
        "total_return": 0.0,
        "annual_return": 0.0,
        "sharpe_ratio": 0.0,
        "sortino_ratio": 0.0,
        "max_drawdown": 0.0,
        "max_drawdown_period": {"start": None, "end": None},
        "annual_volatility": 0.0,
        "calmar_ratio": 0.0,
        "turnover_rate": 0.0,
        "win_rate": 0.0,
        "profit_loss_ratio": 0.0,
        "total_trades": 0,
        "trading_days": 0,
    }
