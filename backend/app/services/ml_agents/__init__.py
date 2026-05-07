"""ML Agent system -- data profiling and configuration.

The Profiler agent analyzes data statistics (Python + optional LLM call).
Config optimization is now handled by the ML Agent Service in the backend
container (see backend/app/services/ml_agent_service.py).

Usage:
    from app.services.ml_agents import get_training_config
    cfg = await get_training_config(market, feature_df, symbols)
    # cfg is a MarketConfig -- always returns default for the market
"""

import logging

import pandas as pd

from app.services.market_config import MarketConfig, get_market_config

logger = logging.getLogger(__name__)


async def get_training_config(
    market: str,
    feature_df: pd.DataFrame,
    symbols: list[str],
    recent_model_ics: list[float] | None = None,
) -> MarketConfig:
    """Return MarketConfig defaults for production daily training.

    Previously used LLM-guided config generation (profiler -> strategist).
    Config optimization is now handled by the backend ML Agent Service
    during backtests. Production daily training uses static defaults.

    Args:
        market: Market code (us, cn, hk).
        feature_df: Feature matrix from build_feature_matrix().
        symbols: List of stock symbols.
        recent_model_ics: Recent model IC values for trend detection.

    Returns:
        MarketConfig instance (static defaults for the market).
    """
    cfg = get_market_config(market)
    logger.info(
        "Using default MarketConfig for %s: lr=%.3f, leaves=%d, rounds=%d",
        market,
        cfg.lgb_overrides.get("learning_rate", 0.05),
        cfg.lgb_overrides.get("num_leaves", 31),
        cfg.num_boost_round,
    )
    return cfg
