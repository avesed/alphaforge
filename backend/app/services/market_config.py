"""Per-market ML configuration — single source of truth.

All market-specific behavior for prediction training, feature engineering,
and data quality is centralized here. Adding a new market-level feature
requires only a change in this file — no scattered ``if market == "..."``
checks in business logic.

Design principles:
- ``frozen=True`` prevents accidental mutation.
- ``lgb_overrides`` layered on top of ``_BASE_LGB_PARAMS`` in prediction_service.
- ``ffill_limit`` passed to fundamental_service to cap forward-fill age.
"""

import dataclasses
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MarketConfig:
    """All per-market ML behavior configuration.

    Training mode
    -------------
    use_temporal_sort : bool
        True  -> symbol-date sort (CN/HK legacy mode -- temporal momentum signal).
        False -> date-symbol sort (US cross-sectional mode -- relative ranking).

    Label engineering
    -----------------
    use_sector_neutral_labels : bool
        True  -> subtract sector mean return from forward returns (US only).
        False -> use raw forward returns.
        CN/HK: sector groups have 3-10 stocks -> sector mean is noise,
        not a meaningful industry benchmark. Explicitly disabled.
    use_balanced_quintiles : bool
        True  -> rank(method="first") + qcut for uniform 5-bin distribution (US).
        False -> qcut(duplicates="drop") -- sparser labels work with CN/HK legacy mode.

    Feature engineering
    -------------------
    use_sector_rank : bool
        True  -> rank valuation features within sectors (US only).
        False -> cross-sectional rank only.
        Same small-group reasoning as use_sector_neutral_labels.
    use_interactions : bool
        True  -> compute cross-feature interaction features (US only).
        CN/HK: legacy training mode + small universes make interactions harmful.
    nan_threshold : float
        Drop features with NaN rate above this threshold.
        US=0.75: prevents "has-data?" spurious splits from sparse features.
        CN/HK=0.90: sparser fundamental coverage, need to keep more features.
    ffill_limit : int
        Forward-fill limit (calendar days) for fundamental data.
        CN=90: quarterly reporting cycle ~90 days.
        US/HK=45: more frequent data updates.

    Quality gate
    ------------
    min_ic_threshold : float
        Minimum IC to pass quality gate.  Override per-market when baseline
        IC differs (e.g. HK=0.009 is below the global default of 0.01).
    min_icir_threshold : float
        Minimum ICIR to pass quality gate.
    min_t_stat : float
        Minimum pooled-IC t-statistic to pass the significance quality gate
        (Batch C). t_stat = pooled_icir * sqrt(N) where N is the total number
        of validation dates pooled across all walk-forward folds. 2.0 ~ 98%
        one-sided confidence that the IC is genuinely positive. HK uses a
        looser 1.65 (~95%) because its baseline IC is weaker (~0.009) and a
        strict gate would reject every model. Only enforced as the binding
        decision when ``QUALITY_GATE_ENFORCE_SIGNIFICANCE`` is true; otherwise
        computed in shadow mode for calibration.
    min_validation_days : int
        Minimum number of pooled per-date IC observations (N) required for the
        significance gate. Below this, the t-statistic is too noisy to trust,
        so the new gate fails. Independent of ``min_t_stat``.
    require_all_folds_positive : bool
        When true, every walk-forward fold's mean IC must be strictly positive
        (min(fold_ics) > 0) for the significance gate to pass. Guards against a
        model that looks good on pooled IC but is negative in one regime.

    Market-level features
    ---------------------
    index_symbol : str
        Benchmark index symbol used for market-level features (index returns,
        market volatility).  Must be a symbol recognised by StockPulse's
        ``/api/v1/internal/history/batch`` endpoint.
        US: ``^GSPC`` (S&P 500), CN: ``000300.SS`` (CSI 300), HK: ``^HSI`` (Hang Seng).

    LightGBM hyperparameters
    ------------------------
    lgb_overrides : dict
        Merged on top of _BASE_LGB_PARAMS in prediction_service.
        Includes min_child_samples: US=30, CN/HK=50 (larger for smaller universes).
    num_boost_round : int
    early_stopping_rounds : int

    Direction model
    ---------------
    direction_lgb_overrides : dict
        Merged on top of ``_DIRECTION_LGB_PARAMS`` in direction_service.
        Allows per-market hyperparameter tuning for binary classification
        independently of the ranking model.
    direction_min_auc : float
        Minimum AUC-ROC to pass the direction model quality gate.
        Default 0.52 (barely above random). HK has lower baseline (0.51).
    direction_max_brier : float
        Maximum Brier score to pass the direction model quality gate.
        Lower is better. Default 0.25 (uncalibrated baseline ~0.25).

    SLA monitoring
    --------------
    sla_max_model_age_days : int
        Maximum acceptable age (calendar days) of the freshest model for this
        market before the daily SLA check raises a structured WARNING and the
        admin scheduler surface reports a breach. HK is given a wider window
        because its training cadence + data coverage are sparser.
    sla_max_prediction_lag_days : int
        Maximum acceptable lag (calendar days) between today and the freshest
        ``stock_predictions.prediction_date`` for this market before the SLA
        check flags a breach. HK is wider for the same reason.
    """

    # Training mode
    use_temporal_sort: bool
    # Label engineering
    use_sector_neutral_labels: bool
    use_balanced_quintiles: bool
    # Feature engineering
    use_sector_rank: bool
    use_interactions: bool
    nan_threshold: float
    ffill_limit: int
    # Market-level features
    index_symbol: str = "^GSPC"
    # Quality gate (per-market overrides for global PREDICTION_MIN_IC/ICIR)
    min_ic_threshold: float = 0.01
    min_icir_threshold: float = 0.10
    # Significance quality gate (Batch C). Only enforced as the binding
    # approved/rejected decision when QUALITY_GATE_ENFORCE_SIGNIFICANCE=true;
    # otherwise computed in shadow mode for calibration.
    min_t_stat: float = 2.0
    min_validation_days: int = 60
    require_all_folds_positive: bool = True
    # LightGBM hyperparameters
    lgb_overrides: dict[str, Any] = field(default_factory=dict)
    num_boost_round: int = 1000
    early_stopping_rounds: int = 100
    # Direction model
    direction_lgb_overrides: dict[str, Any] = field(default_factory=dict)
    direction_min_auc: float = 0.52
    direction_max_brier: float = 0.255
    # SLA monitoring (Batch D). HK gets a wider window (sparser cadence/data).
    sla_max_model_age_days: int = 10
    sla_max_prediction_lag_days: int = 5


MARKET_CONFIGS: dict[str, MarketConfig] = {
    "us": MarketConfig(
        use_temporal_sort=False,
        use_sector_neutral_labels=True,
        use_balanced_quintiles=True,
        use_sector_rank=True,
        use_interactions=True,
        nan_threshold=0.75,
        ffill_limit=45,
        lgb_overrides={
            "learning_rate": 0.01,
            "num_leaves": 31,
            "min_child_samples": 30,
            "lambda_l2": 1.0,
        },
        num_boost_round=1000,
        early_stopping_rounds=100,
        # Direction: low LR + few leaves for weak-signal classification.
        # US has more data (195k rows) but absolute direction is still noisy.
        direction_lgb_overrides={
            "learning_rate": 0.005,
            "num_leaves": 15,
            "min_child_samples": 100,
            "lambda_l2": 5.0,
        },
    ),
    "cn": MarketConfig(
        # CN uses temporal sort (symbol-date) -- strong momentum signal in A-shares.
        # Sector groups have 3-10 stocks -> sector neutralization is noise, not signal.
        # Quarterly reporting cycle -> ffill up to 90 calendar days.
        use_temporal_sort=True,
        use_sector_neutral_labels=False,
        use_balanced_quintiles=False,
        use_sector_rank=False,
        use_interactions=False,
        nan_threshold=0.90,
        ffill_limit=90,
        index_symbol="000300.SS",
        lgb_overrides={
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_child_samples": 50,
            "lambda_l2": 0.1,
        },
        num_boost_round=500,
        early_stopping_rounds=50,
        # CN direction AUC baseline ~0.52, lower threshold to match HK
        direction_min_auc=0.51,
        # Direction: low LR + few leaves -- prevent overfit in 1-2 iters
        direction_lgb_overrides={
            "learning_rate": 0.01,
            "num_leaves": 15,
            "min_child_samples": 100,
            "lambda_l2": 5.0,
        },
    ),
    "hk": MarketConfig(
        # HK uses temporal sort (symbol-date) -- small universe (~79 stocks).
        # Sector groups have 3-10 stocks -> same reasoning as CN.
        # More frequent data updates than CN -> ffill 45 days.
        # Baseline IC=0.009 -- below global 0.01, needs lower quality gate.
        use_temporal_sort=True,
        use_sector_neutral_labels=False,
        use_balanced_quintiles=False,
        use_sector_rank=False,
        use_interactions=False,
        nan_threshold=0.90,
        ffill_limit=45,
        index_symbol="^HSI",
        min_ic_threshold=0.005,
        min_icir_threshold=0.05,
        # HK baseline IC=0.009 -- a strict 2.0 t-stat gate would reject every
        # model, so use a looser 1.65 (~95% one-sided). This only takes effect
        # once QUALITY_GATE_ENFORCE_SIGNIFICANCE=true; in shadow mode it merely
        # changes what the logged "would decide" verdict is for HK.
        min_t_stat=1.65,
        lgb_overrides={
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_child_samples": 50,
            "lambda_l2": 0.1,
        },
        num_boost_round=500,
        early_stopping_rounds=50,
        direction_min_auc=0.51,
        direction_lgb_overrides={
            "learning_rate": 0.01,
            "num_leaves": 15,
            "min_child_samples": 100,
            "lambda_l2": 5.0,
        },
        # HK trains less frequently and has sparser data coverage -> wider SLA.
        sla_max_model_age_days=21,
        sla_max_prediction_lag_days=10,
    ),
}


def get_market_config(market: str) -> MarketConfig:
    """Return MarketConfig for the given market code.

    Falls back to US config for unknown markets (safe default).
    All callers should use this function rather than accessing
    MARKET_CONFIGS directly, to benefit from the fallback.
    """
    cfg = MARKET_CONFIGS.get(market.lower())
    if cfg is None:
        logger.warning(
            "Unknown market %r -- falling back to US MarketConfig. "
            "Add a dedicated entry to MARKET_CONFIGS if needed.",
            market,
        )
        cfg = MARKET_CONFIGS["us"]
    return cfg


def apply_override(market: str, overrides: dict[str, Any] | None = None) -> MarketConfig:
    """Return MarketConfig for market with optional field overrides.

    Uses dataclasses.replace() on the base config. Only fields present
    in MarketConfig are applied; unknown keys are logged and ignored.

    This is the primary entry point when callers want to tweak specific
    parameters (e.g. from a backtest config_override or LLM suggestion)
    without replacing the entire config.
    """
    cfg = get_market_config(market)
    if not overrides:
        return cfg

    valid_fields = {f.name for f in dataclasses.fields(MarketConfig)}
    filtered = {}
    unknown = []
    for k, v in overrides.items():
        if k in valid_fields:
            filtered[k] = v
        else:
            unknown.append(k)

    if unknown:
        logger.warning("Ignoring unknown MarketConfig fields: %s", unknown)

    if not filtered:
        return cfg

    return dataclasses.replace(cfg, **filtered)
