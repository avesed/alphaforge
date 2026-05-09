"""Tests for app.core.secrets -- JWT secret bootstrapping.

~5 tests covering:
- get_jwt_secret returns stored value when set
- get_jwt_secret raises RuntimeError when _jwt_secret is None
- _env_secret_usable helper logic
"""

import pytest

import app.core.secrets as secrets_mod
from app.core.secrets import (
    JWT_SECRET_PLACEHOLDER,
    MIN_SECRET_LEN,
    _env_secret_usable,
    get_jwt_secret,
)


class TestGetJwtSecret:
    def test_returns_value_when_set(self, monkeypatch):
        secret = "a-real-secret-that-is-long-enough-for-tests"
        monkeypatch.setattr(secrets_mod, "_jwt_secret", secret)
        assert get_jwt_secret() == secret

    def test_raises_runtime_error_when_none(self, monkeypatch):
        monkeypatch.setattr(secrets_mod, "_jwt_secret", None)
        with pytest.raises(RuntimeError, match="JWT secret not bootstrapped"):
            get_jwt_secret()


class TestEnvSecretUsable:
    def test_returns_none_when_env_not_set(self, monkeypatch):
        monkeypatch.delenv("JWT_SECRET_KEY", raising=False)
        assert _env_secret_usable() is None

    def test_returns_none_for_placeholder_value(self, monkeypatch):
        monkeypatch.setenv("JWT_SECRET_KEY", JWT_SECRET_PLACEHOLDER)
        assert _env_secret_usable() is None

    def test_returns_none_for_short_secret(self, monkeypatch):
        monkeypatch.setenv("JWT_SECRET_KEY", "short")
        assert _env_secret_usable() is None

    def test_returns_value_for_valid_secret(self, monkeypatch):
        valid = "x" * MIN_SECRET_LEN
        monkeypatch.setenv("JWT_SECRET_KEY", valid)
        assert _env_secret_usable() == valid

    def test_returns_value_for_long_secret(self, monkeypatch):
        long_secret = "a" * 128
        monkeypatch.setenv("JWT_SECRET_KEY", long_secret)
        assert _env_secret_usable() == long_secret
