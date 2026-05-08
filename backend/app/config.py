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
    PREDICTION_HORIZONS: str = "5"
    PREDICTION_UNIVERSE_SIZE: int = 500
    PREDICTION_MAX_STALE_DAYS: int = 5
    INFERENCE_MIN_COVERAGE: float = 0.5
    MODEL_RETENTION_DAYS: int = 90
    MODEL_MIN_QUALITY_KEEP: int = 3

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
