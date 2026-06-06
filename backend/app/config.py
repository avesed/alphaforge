"""AlphaForge application settings.

Loaded once from environment / .env file and cached via lru_cache.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "AlphaForge"
    debug: bool = False

    # Local DB (auth only)
    DATABASE_URL: str = "postgresql+asyncpg://alphaforge:alphaforge@postgres:5432/alphaforge"
    DATABASE_POOL_MIN_SIZE: int = 2
    DATABASE_POOL_MAX_SIZE: int = 10
    DATABASE_COMMAND_TIMEOUT: int = 120

    # Redis
    REDIS_URL: str = "redis://redis:6379/0"

    # JWT
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # CORS
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:5173"

    # Admin bootstrap
    FIRST_ADMIN_EMAIL: str = ""

    # StockPulse API (ALL ML data access)
    STOCKPULSE_URL: str = "http://stockpulse:8010"
    STOCKPULSE_API_KEY: str = ""

    # NewsForge API (sentiment data)
    NEWSFORGE_URL: str = "http://newsforge:8080"
    NEWSFORGE_API_KEY: str = ""

    # AI Gateway (LLM proxy for RD-Agent)
    AI_GATEWAY_URL: str = "http://ai-gateway:8004"

    # Qlib
    QLIB_DATA_DIR: str = "/app/data/qlib"
    MAX_EXPRESSION_LENGTH: int = 500
    MAX_CONCURRENT_BACKTESTS: int = 1
    BACKTEST_TIMEOUT_SECONDS: int = 1800  # 30 minutes

    # ML model artifacts
    PREDICTION_DATA_DIR: str = "/app/data/predictions"
    PREDICTION_MIN_IC: float = 0.01
    PREDICTION_MIN_ICIR: float = 0.1
    ENSEMBLE_SIZE: int = 5
    WALKFORWARD_FOLDS: int = 3
    # Feature-selection stage (noise reduction). Global kill-switch. When false
    # (default) the training pipeline behaves exactly as before -- no features
    # are dropped. When true, AND a market's MarketConfig.use_feature_selection
    # is also true, a leakage-safe selection (computed only on the first
    # walk-forward fold's training window) reduces feature_cols before the fold
    # loop. Both flags must be true for selection to run.
    FEATURE_SELECTION_ENABLED: bool = False
    # Comma-separated forward-return horizons (trading days) to train & serve.
    # "5,20,30": 5d (timely) + 20d/30d (stronger signal — clean-universe + tuned
    # reg: IC ~0.036 / 0.117 / 0.135). Longer horizons have higher signal-to-noise
    # (60d overfits — negative fold — so it is excluded). A deployment .env may override.
    PREDICTION_HORIZONS: str = "5,20,30"
    # Default horizon to SERVE/TRADE when the caller does not specify one. All
    # horizons in PREDICTION_HORIZONS are still trained & served; this only
    # changes the implicit pick. 30d is preferred: highest gross quintile spread
    # and lowest rebalance turnover (5d is the worst on both). If this value is
    # not among the trained horizons, the serving path falls back to the largest
    # trained horizon (never crashes).
    DEFAULT_TRADE_HORIZON: int = 30
    PREDICTION_UNIVERSE_SIZE: int = 500

    # Short-interest feature mode (US-only block). "naive" preserves the exact
    # current behavior byte-for-byte: an exact (symbol, date) merge of EVERY
    # short column (short_pct_float, short_ratio, shares_short,
    # shares_short_prior, short_pct_shares_out) followed by a per-symbol
    # forward-fill, no derived features, and no publication lag. "improved" is a
    # point-in-time as-of merge: each reading is attached only from its (lagged)
    # public date = stored date + PUBLICATION_LAG_BDAYS business days, restricted
    # to an explicit whitelist plus a derived short_change squeeze feature (raw
    # share-count size proxies are NOT carried in). "off" drops the short block
    # entirely (no fetch, no columns).
    #
    # Default is "off": a 2026-06-06 US rolling-forward A/B (10 bi-monthly
    # cutoffs, LGB OOS) found the short-interest signal NOISE-LEVEL. "improved"
    # was the best of the three but beat "off" by only +0.004..0.013 mean IC
    # (below the +0.02 bar, t-insignificant) and did NOT repair the 2025-02
    # crash window; "naive" was even mildly HARMFUL at 20d/30d (leaked PIT + raw
    # size-proxy cols). So the block is off by default. "naive"/"improved" are
    # kept as configurable options (improved is strictly best if short data is
    # ever worth re-enabling -- would require FINRA incremental feed first).
    SHORT_INTEREST_MODE: str = "off"   # off | naive | improved
    # Business-day lag applied to the stored short-interest date before the
    # "improved" as-of merge. Upstream date semantics are MIXED: FINRA-backfilled
    # rows store the SETTLEMENT date (public ~8 bdays later -> the lag is the
    # true dissemination model), while ongoing yfinance rows store the COLLECTION
    # date (already public -> the lag is a conservative safety buffer). The shift
    # only ever moves data LATER, so it is leakage-safe in both cases.
    SHORT_INTEREST_PUBLICATION_LAG_BDAYS: int = 8
    PREDICTION_MAX_STALE_DAYS: int = 5
    INFERENCE_MIN_COVERAGE: float = 0.5
    MODEL_RETENTION_DAYS: int = 90
    MODEL_MIN_QUALITY_KEEP: int = 3

    # Quality-gate binding (Batch A). When true, a rejected model is never
    # served: the serving path prefers the latest prior approved model, and
    # only serves today's rejected model (tagged low_confidence) when no
    # approved model has ever existed for that market. Set false to revert to
    # the legacy behavior of serving the latest model on disk regardless of
    # quality.
    QUALITY_GATE_BINDING: bool = True

    # Significance quality gate (Batch C). When false (default = SHADOW mode),
    # the actual approved/rejected decision still uses the legacy gate
    # (ic_mean > min_ic AND icir > min_icir); the new significance gate
    # (pooled IC + per-fold lower bound + N validation days + t-statistic) is
    # only computed and logged for calibration ("what the new gate WOULD
    # decide"). Flip to true to ENFORCE the significance gate as the binding
    # approved/rejected decision -- only after the true per-fold N has been
    # calibrated from real retraining runs.
    QUALITY_GATE_ENFORCE_SIGNIFICANCE: bool = False

    # Net-of-cost / turnover quality gate. When false (default = SHADOW mode),
    # the binding approved/rejected decision is byte-identical to today: the
    # gross quintile spread, rebalance turnover, and net-of-cost spread are
    # computed, persisted, and logged for calibration ("what the net-cost gate
    # WOULD decide") but do NOT affect the decision. When true, an ADDITIONAL
    # binding requirement (mean_net_spread > 0) is AND-ed onto the existing gate:
    # a model is approved only if it passes the legacy-or-significance gate AND
    # its turnover-adjusted spread is still positive. Mirrors the significance
    # gate's shadow pattern.
    NET_COST_GATE_ENABLED: bool = False
    # One-way trading cost in basis points (commission + slippage). 10 bps is a
    # conservative US large-cap default; tune per universe via .env. The gate's
    # net spread drag uses 4x this: two legs (long Q5 + short Q1) x round-trip
    # (exit + enter). Net spread per rebalance =
    #   gross_spread - turnover * 4 * (TRADING_COST_BPS_ONEWAY / 10000).
    TRADING_COST_BPS_ONEWAY: float = 10.0
    # Trading days per year, used to annualize the net spread + turnover so
    # horizons are calendar-comparable (frequency-aware).
    TRADING_DAYS_PER_YEAR: int = 252
    # Annual-turnover ceiling for the net-cost gate (annual_turnover =
    # turnover_per_rebalance * TRADING_DAYS_PER_YEAR / forward_days). When
    # NET_COST_GATE_ENABLED, a model is rejected if its annual turnover exceeds
    # this -- the real penalty for high-frequency (e.g. 5d) churn. Default is
    # high (effectively no ceiling); lower it after inspecting shadow
    # annual_turnover logs (e.g. ~5 rejects 5d ~7.7 but passes 30d ~2.1 here).
    MAX_ANNUAL_TURNOVER: float = 999.0

    # RD-Agent
    RDAGENT_CHAT_MODEL: str = "gpt-4o-mini"
    RDAGENT_EMBED_MODEL: str = "text-embedding-3-small"

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8015
    LOG_LEVEL: str = "info"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]


@lru_cache()
def get_settings() -> Settings:
    return Settings()
