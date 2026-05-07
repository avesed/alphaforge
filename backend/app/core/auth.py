"""Authentication -- JWT for admin users, X-API-Key for machine consumers.

Dual auth system:
- JWT Bearer: admin UI login (access 30min + refresh 7d with rotation)
- X-API-Key: machine consumers (SHA-256 hash lookup, rate limiting)

Refresh token rotation uses Redis-backed atomic jti claiming.
Redis outage => fail-closed (503, never silently skip revocation).
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import redis.exceptions
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.secrets import get_jwt_secret
from app.core.orm import get_db
from app.core.redis import get_redis
from app.models.user import User
from app.models.api_consumer import ApiConsumer

logger = logging.getLogger(__name__)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto", bcrypt__rounds=12)
security = HTTPBearer(auto_error=False)

ALGORITHM = "HS256"
REFRESH_DENYLIST_PREFIX = "af:jwt:refresh:revoked:"


# --- Password helpers ---

def hash_password(password: str) -> str:
    return pwd_context.hash(password[:72])


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain[:72], hashed)


# --- API key helpers ---

def hash_api_key(raw_key: str) -> str:
    """Hash an API key with SHA-256 for storage."""
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


# --- JWT token creation ---

def create_access_token(user_id: int, role: str) -> str:
    settings = get_settings()
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {"sub": str(user_id), "role": role, "exp": expire, "type": "access"}
    return jwt.encode(payload, get_jwt_secret(), algorithm=ALGORITHM)


def create_refresh_token(user_id: int) -> str:
    settings = get_settings()
    expire = datetime.now(timezone.utc) + timedelta(days=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS)
    payload = {
        "sub": str(user_id),
        "exp": expire,
        "type": "refresh",
        "jti": uuid4().hex,
    }
    return jwt.encode(payload, get_jwt_secret(), algorithm=ALGORITHM)


def decode_refresh_token(token: str) -> tuple[int, str, int]:
    """Validate refresh token signature and structure.

    Returns ``(user_id, jti, exp_unix)``. Does NOT consult Redis.
    """
    try:
        payload = jwt.decode(token, get_jwt_secret(), algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token")

    if payload.get("type") != "refresh":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not a refresh token")

    user_id = int(payload.get("sub", 0))
    jti = payload.get("jti")
    exp_ts = int(payload.get("exp", 0))
    if not user_id or not jti or not exp_ts:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Malformed refresh token")

    return user_id, jti, exp_ts


async def claim_refresh_jti(jti: str, exp_ts: int) -> bool:
    """Atomically revoke a refresh jti; returns True iff first to claim."""
    settings = get_settings()
    now = int(datetime.now(timezone.utc).timestamp())
    ttl = max(60, min(exp_ts - now, settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS * 86400))
    try:
        redis_client = await get_redis()
        result = await redis_client.set(
            f"{REFRESH_DENYLIST_PREFIX}{jti}", "1", nx=True, ex=ttl
        )
    except redis.exceptions.RedisError:
        logger.warning("Redis unavailable during refresh jti claim", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Token revocation store unavailable",
        )
    return bool(result)


# --- JWT dependencies ---

async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
    db: AsyncSession = Depends(get_db),
) -> User:
    """Extract and validate JWT, return User."""
    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")

    try:
        payload = jwt.decode(credentials.credentials, get_jwt_secret(), algorithms=[ALGORITHM])
        user_id = int(payload.get("sub", 0))
        token_type = payload.get("type", "")
        if not user_id or token_type != "access":
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if user is None or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found or inactive")

    return user


async def require_admin(user: User = Depends(get_current_user)) -> User:
    """Require admin role."""
    if user.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin required")
    return user


# --- X-API-Key dependency ---

async def get_api_consumer(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> ApiConsumer:
    """Validate X-API-Key header, hash with SHA-256, lookup in DB.

    Updates last_used_at on success (throttled to once per 5 minutes).
    """
    raw_key = request.headers.get("X-API-Key")
    if not raw_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="X-API-Key header required")

    key_hash = hash_api_key(raw_key)

    result = await db.execute(
        select(ApiConsumer).where(
            ApiConsumer.api_key == key_hash,
            ApiConsumer.is_active.is_(True),
        )
    )
    consumer = result.scalar_one_or_none()
    if consumer is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or inactive API key")

    # Throttle last_used_at writes: skip if updated within the last 5 minutes
    throttle_key = f"af:consumer:last_used:{consumer.id}"
    redis_client = None
    try:
        redis_client = await get_redis()
        already_fresh = await redis_client.get(throttle_key)
    except Exception:
        already_fresh = None

    if not already_fresh:
        await db.execute(
            update(ApiConsumer)
            .where(ApiConsumer.id == consumer.id)
            .values(last_used_at=datetime.now(timezone.utc))
        )
        if redis_client is not None:
            try:
                await redis_client.set(throttle_key, "1", ex=300)
            except Exception:
                pass

    return consumer
