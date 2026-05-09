"""Tests for application settings."""
import pytest

from app.config import Settings, get_settings


class TestSettingsDefaults:

    def test_app_name(self, test_settings):
        assert test_settings.app_name == "AlphaForge"

    def test_debug_default_false(self, test_settings):
        assert test_settings.debug is False

    def test_jwt_defaults(self, test_settings):
        assert test_settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES == 30
        assert test_settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS == 7

    def test_ml_defaults(self, test_settings):
        assert test_settings.ENSEMBLE_SIZE == 5
        assert test_settings.WALKFORWARD_FOLDS == 3
        assert test_settings.PREDICTION_MIN_IC == 0.01
        assert test_settings.PREDICTION_MIN_ICIR == 0.1

    def test_qlib_defaults(self, test_settings):
        assert test_settings.MAX_EXPRESSION_LENGTH == 500
        assert test_settings.BACKTEST_TIMEOUT_SECONDS == 1800

    def test_injected_env_values(self, test_settings):
        assert test_settings.STOCKPULSE_URL == "http://test-stockpulse:8010"
        assert test_settings.NEWSFORGE_URL == "http://test-newsforge:8080"


class TestCorsOriginList:

    def test_comma_separated(self, monkeypatch):
        monkeypatch.setenv("CORS_ORIGINS", "http://a.com,http://b.com")
        monkeypatch.setenv("DATABASE_URL", "sqlite://")
        monkeypatch.setenv("REDIS_URL", "redis://fake")
        s = Settings()
        assert s.cors_origin_list == ["http://a.com", "http://b.com"]

    def test_with_whitespace(self, monkeypatch):
        monkeypatch.setenv("CORS_ORIGINS", " http://a.com , http://b.com ")
        monkeypatch.setenv("DATABASE_URL", "sqlite://")
        monkeypatch.setenv("REDIS_URL", "redis://fake")
        s = Settings()
        assert s.cors_origin_list == ["http://a.com", "http://b.com"]

    def test_empty_string(self, monkeypatch):
        monkeypatch.setenv("CORS_ORIGINS", "")
        monkeypatch.setenv("DATABASE_URL", "sqlite://")
        monkeypatch.setenv("REDIS_URL", "redis://fake")
        s = Settings()
        assert s.cors_origin_list == []

    def test_single_origin(self, monkeypatch):
        monkeypatch.setenv("CORS_ORIGINS", "http://localhost:3000")
        monkeypatch.setenv("DATABASE_URL", "sqlite://")
        monkeypatch.setenv("REDIS_URL", "redis://fake")
        s = Settings()
        assert s.cors_origin_list == ["http://localhost:3000"]


class TestGetSettingsCache:

    def test_lru_cache_returns_same_instance(self, test_settings):
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_cache_clear_produces_new_instance(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "sqlite://")
        monkeypatch.setenv("REDIS_URL", "redis://fake")
        s1 = get_settings()
        get_settings.cache_clear()
        s2 = get_settings()
        assert s1 is not s2
