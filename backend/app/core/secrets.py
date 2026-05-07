"""Startup-time bootstrap of process-level secrets.

Env takes priority when set to a non-placeholder, sufficiently long value;
otherwise we generate a random secret on first startup and persist it
in ``system_settings`` so every container agrees on the signing key.

Concurrency: two instances starting on a fresh DB can both race to insert.
``INSERT ... ON CONFLICT DO NOTHING`` keeps the winner; the loser's generated
value is discarded when we re-select the row.
"""

from __future__ import annotations

import logging
import os
import secrets as _secrets

from sqlalchemy import text

from app.core.orm import get_session_factory

logger = logging.getLogger(__name__)

JWT_SECRET_PLACEHOLDER = "change-me-in-production"
MIN_SECRET_LEN = 32

_ENV_KEY = "JWT_SECRET_KEY"
_DB_KEY = "jwt_secret_key"

_jwt_secret: str | None = None


def get_jwt_secret() -> str:
    """Return the bootstrapped JWT secret. Must be called post-startup."""
    if _jwt_secret is None:
        raise RuntimeError(
            "JWT secret not bootstrapped -- did bootstrap_jwt_secret() run in lifespan?"
        )
    return _jwt_secret


def _env_secret_usable() -> str | None:
    """Decide whether ``JWT_SECRET_KEY`` env is a real override."""
    val = os.environ.get(_ENV_KEY)
    if not val:
        return None
    if val == JWT_SECRET_PLACEHOLDER:
        logger.warning(
            "JWT_SECRET_KEY is set to the example placeholder; falling back to DB-backed secret"
        )
        return None
    if len(val) < MIN_SECRET_LEN:
        logger.warning(
            "JWT_SECRET_KEY is shorter than %d chars; rejecting as likely typo "
            "and falling back to DB-backed secret",
            MIN_SECRET_LEN,
        )
        return None
    return val


async def bootstrap_jwt_secret() -> None:
    """Resolve the JWT secret from env or DB, generating+persisting if absent."""
    global _jwt_secret

    env_val = _env_secret_usable()
    if env_val is not None:
        _jwt_secret = env_val
        logger.info("JWT secret loaded from env (JWT_SECRET_KEY)")
        return

    generated = _secrets.token_urlsafe(64)
    factory = get_session_factory()
    async with factory() as session:
        await session.execute(
            text(
                "INSERT INTO system_settings (key, value) "
                "VALUES (:key, :value) "
                "ON CONFLICT (key) DO NOTHING"
            ),
            {"key": _DB_KEY, "value": generated},
        )
        await session.commit()
        row = (
            await session.execute(
                text("SELECT value FROM system_settings WHERE key = :key"),
                {"key": _DB_KEY},
            )
        ).first()

    if row is None:
        raise RuntimeError("Failed to bootstrap jwt_secret_key in system_settings")

    stored = row[0]
    _jwt_secret = stored
    if stored == generated:
        logger.info("JWT secret generated and persisted to system_settings")
    else:
        logger.info("JWT secret loaded from system_settings (existing row)")
