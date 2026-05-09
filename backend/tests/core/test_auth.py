"""Tests for app.core.auth -- password hashing, JWT creation/decoding, API key hashing.

~20 tests covering:
- hash_password / verify_password (bcrypt)
- create_access_token / create_refresh_token (JWT creation)
- decode_refresh_token (validation, error cases)
- claim_refresh_jti (Redis-backed replay detection)
- hash_api_key (SHA-256 determinism)
"""

import hashlib
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from jose import jwt

from app.core.auth import (
    ALGORITHM,
    REFRESH_DENYLIST_PREFIX,
    claim_refresh_jti,
    create_access_token,
    create_refresh_token,
    decode_refresh_token,
    hash_api_key,
    hash_password,
    verify_password,
)


# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------


class TestHashPassword:
    def test_returns_bcrypt_hash(self):
        hashed = hash_password("secret123")
        assert hashed.startswith("$2b$")

    def test_different_passwords_produce_different_hashes(self):
        h1 = hash_password("password_a")
        h2 = hash_password("password_b")
        assert h1 != h2

    def test_same_password_produces_unique_hashes(self):
        """bcrypt salts should differ on each call."""
        h1 = hash_password("same")
        h2 = hash_password("same")
        assert h1 != h2


class TestVerifyPassword:
    def test_correct_password_returns_true(self):
        hashed = hash_password("correct-horse")
        assert verify_password("correct-horse", hashed) is True

    def test_wrong_password_returns_false(self):
        hashed = hash_password("correct-horse")
        assert verify_password("wrong-horse", hashed) is False

    def test_empty_password_does_not_match(self):
        hashed = hash_password("notempty")
        assert verify_password("", hashed) is False

    def test_password_truncated_at_72_bytes(self):
        """bcrypt only considers the first 72 bytes; auth.py slices at [:72]."""
        long_pw = "a" * 100
        hashed = hash_password(long_pw)
        # Verifying with the first 72 chars should still match
        assert verify_password(long_pw[:72], hashed) is True


# ---------------------------------------------------------------------------
# API key hashing
# ---------------------------------------------------------------------------


class TestHashApiKey:
    def test_sha256_determinism(self):
        key = "af_test_abc123xyz"
        expected = hashlib.sha256(key.encode("utf-8")).hexdigest()
        assert hash_api_key(key) == expected

    def test_different_keys_produce_different_hashes(self):
        assert hash_api_key("key_a") != hash_api_key("key_b")

    def test_hash_length_is_64_hex_chars(self):
        assert len(hash_api_key("any_key")) == 64


# ---------------------------------------------------------------------------
# JWT access token
# ---------------------------------------------------------------------------


class TestCreateAccessToken:
    def test_returns_valid_jwt_string(self, test_settings, jwt_secret):
        token = create_access_token(user_id=42, role="admin")
        assert isinstance(token, str)
        # Must be decodable
        payload = jwt.decode(token, jwt_secret, algorithms=[ALGORITHM])
        assert payload is not None

    def test_contains_sub_claim(self, test_settings, jwt_secret):
        token = create_access_token(user_id=7, role="user")
        payload = jwt.decode(token, jwt_secret, algorithms=[ALGORITHM])
        assert payload["sub"] == "7"

    def test_contains_role_claim(self, test_settings, jwt_secret):
        token = create_access_token(user_id=1, role="admin")
        payload = jwt.decode(token, jwt_secret, algorithms=[ALGORITHM])
        assert payload["role"] == "admin"

    def test_type_is_access(self, test_settings, jwt_secret):
        token = create_access_token(user_id=1, role="user")
        payload = jwt.decode(token, jwt_secret, algorithms=[ALGORITHM])
        assert payload["type"] == "access"

    def test_exp_claim_is_in_future(self, test_settings, jwt_secret):
        token = create_access_token(user_id=1, role="user")
        payload = jwt.decode(token, jwt_secret, algorithms=[ALGORITHM])
        assert payload["exp"] > int(datetime.now(timezone.utc).timestamp())


# ---------------------------------------------------------------------------
# JWT refresh token
# ---------------------------------------------------------------------------


class TestCreateRefreshToken:
    def test_type_is_refresh(self, test_settings, jwt_secret):
        token = create_refresh_token(user_id=5)
        payload = jwt.decode(token, jwt_secret, algorithms=[ALGORITHM])
        assert payload["type"] == "refresh"

    def test_contains_jti_claim(self, test_settings, jwt_secret):
        token = create_refresh_token(user_id=5)
        payload = jwt.decode(token, jwt_secret, algorithms=[ALGORITHM])
        assert "jti" in payload
        assert len(payload["jti"]) == 32  # uuid4().hex is 32 hex chars

    def test_contains_sub_claim(self, test_settings, jwt_secret):
        token = create_refresh_token(user_id=99)
        payload = jwt.decode(token, jwt_secret, algorithms=[ALGORITHM])
        assert payload["sub"] == "99"


# ---------------------------------------------------------------------------
# decode_refresh_token
# ---------------------------------------------------------------------------


class TestDecodeRefreshToken:
    def test_valid_token_returns_tuple(self, test_settings, jwt_secret):
        token = create_refresh_token(user_id=10)
        user_id, jti, exp_ts = decode_refresh_token(token)
        assert user_id == 10
        assert isinstance(jti, str) and len(jti) == 32
        assert exp_ts > int(datetime.now(timezone.utc).timestamp())

    def test_invalid_signature_raises_401(self, test_settings, jwt_secret):
        payload = {
            "sub": "1",
            "type": "refresh",
            "jti": "abc123",
            "exp": int((datetime.now(timezone.utc) + timedelta(hours=1)).timestamp()),
        }
        bad_token = jwt.encode(payload, "wrong-secret-key-wrong-secret-key", algorithm=ALGORITHM)
        with pytest.raises(HTTPException) as exc_info:
            decode_refresh_token(bad_token)
        assert exc_info.value.status_code == 401
        assert "Invalid refresh token" in exc_info.value.detail

    def test_wrong_token_type_raises_401(self, test_settings, jwt_secret):
        """An access token passed as refresh should be rejected."""
        token = create_access_token(user_id=1, role="user")
        with pytest.raises(HTTPException) as exc_info:
            decode_refresh_token(token)
        assert exc_info.value.status_code == 401
        assert "Not a refresh token" in exc_info.value.detail

    def test_expired_token_raises_401(self, test_settings, jwt_secret):
        payload = {
            "sub": "1",
            "type": "refresh",
            "jti": "deadbeef" * 4,
            "exp": int((datetime.now(timezone.utc) - timedelta(hours=1)).timestamp()),
        }
        expired_token = jwt.encode(payload, jwt_secret, algorithm=ALGORITHM)
        with pytest.raises(HTTPException) as exc_info:
            decode_refresh_token(expired_token)
        assert exc_info.value.status_code == 401


# ---------------------------------------------------------------------------
# claim_refresh_jti (Redis-backed replay detection)
# ---------------------------------------------------------------------------


class TestClaimRefreshJti:
    async def test_first_claim_returns_true(self, test_settings, jwt_secret, fake_redis):
        """First claim of a jti should succeed."""
        exp_ts = int((datetime.now(timezone.utc) + timedelta(days=7)).timestamp())

        with patch("app.core.auth.get_redis", return_value=fake_redis):
            result = await claim_refresh_jti("unique-jti-001", exp_ts)

        assert result is True

    async def test_second_claim_returns_false(self, test_settings, jwt_secret, fake_redis):
        """Replaying the same jti should fail (replay detection)."""
        exp_ts = int((datetime.now(timezone.utc) + timedelta(days=7)).timestamp())
        jti = "replay-jti-002"

        with patch("app.core.auth.get_redis", return_value=fake_redis):
            first = await claim_refresh_jti(jti, exp_ts)
            second = await claim_refresh_jti(jti, exp_ts)

        assert first is True
        assert second is False

    async def test_different_jtis_both_succeed(self, test_settings, jwt_secret, fake_redis):
        """Different jtis should both succeed independently."""
        exp_ts = int((datetime.now(timezone.utc) + timedelta(days=7)).timestamp())

        with patch("app.core.auth.get_redis", return_value=fake_redis):
            r1 = await claim_refresh_jti("jti-aaa", exp_ts)
            r2 = await claim_refresh_jti("jti-bbb", exp_ts)

        assert r1 is True
        assert r2 is True

    async def test_redis_key_uses_denylist_prefix(self, test_settings, jwt_secret, fake_redis):
        """Verify the key stored in Redis uses the correct prefix."""
        exp_ts = int((datetime.now(timezone.utc) + timedelta(days=7)).timestamp())
        jti = "prefix-check-jti"

        with patch("app.core.auth.get_redis", return_value=fake_redis):
            await claim_refresh_jti(jti, exp_ts)

        stored = await fake_redis.get(f"{REFRESH_DENYLIST_PREFIX}{jti}")
        assert stored == "1"
