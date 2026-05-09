"""API test fixtures -- FastAPI test app, httpx clients, users, tokens.

Overrides all external dependencies (DB, Redis, JWT secret) so tests
run against in-memory SQLite + fakeredis without lifespan/startup side
effects.
"""

from __future__ import annotations

import pytest
import httpx
from httpx import ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.auth import create_access_token, hash_api_key, hash_password
from app.core.orm import get_db
from app.models.api_consumer import ApiConsumer
from app.models.user import User


# ---------------------------------------------------------------------------
# app fixture -- FastAPI test application with overridden dependencies
# ---------------------------------------------------------------------------

@pytest.fixture
async def app(db_engine, fake_redis, test_settings, jwt_secret, monkeypatch):
    """Create a FastAPI test app with all external deps replaced.

    * ORM ``get_db`` -> aiosqlite session backed by ``db_engine``
    * Redis -> fakeredis instance
    * JWT secret -> deterministic test value
    * Lifespan is NOT triggered (no startup/shutdown side effects).
    """
    import app.core.redis as redis_mod
    import app.core.orm as orm_mod

    # Patch Redis singleton so get_redis() returns fakeredis
    monkeypatch.setattr(redis_mod, "_redis_client", fake_redis)

    # Reset ORM module-level singletons so they don't leak between tests
    monkeypatch.setattr(orm_mod, "_engine", None)
    monkeypatch.setattr(orm_mod, "_session_factory", None)

    # Build the app (without lifespan)
    from app.main import create_app
    test_app = create_app()

    # Override the get_db dependency to use the test engine
    _test_session_factory = async_sessionmaker(
        db_engine, class_=AsyncSession, expire_on_commit=False
    )

    async def _override_get_db():
        async with _test_session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    test_app.dependency_overrides[get_db] = _override_get_db

    yield test_app

    test_app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# HTTP clients
# ---------------------------------------------------------------------------

@pytest.fixture
async def client(app):
    """Unauthenticated httpx AsyncClient against the test app."""
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture
async def auth_client(app, admin_token):
    """Authenticated httpx AsyncClient with admin Bearer token."""
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
        headers={"Authorization": f"Bearer {admin_token}"},
    ) as c:
        yield c


@pytest.fixture
async def regular_client(app, regular_token):
    """Authenticated httpx AsyncClient with regular-user Bearer token."""
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
        headers={"Authorization": f"Bearer {regular_token}"},
    ) as c:
        yield c


# ---------------------------------------------------------------------------
# Users + tokens
# ---------------------------------------------------------------------------

@pytest.fixture
async def admin_user(db_engine):
    """Create an admin user in the test DB and return it."""
    factory = async_sessionmaker(
        db_engine, class_=AsyncSession, expire_on_commit=False
    )
    async with factory() as session:
        user = User(
            email="admin@test.com",
            password_hash=hash_password("Admin123"),
            role="admin",
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)
        return user


@pytest.fixture
def admin_token(admin_user, jwt_secret):
    """Create a real JWT access token for the admin user."""
    return create_access_token(admin_user.id, admin_user.role)


@pytest.fixture
async def regular_user(db_engine):
    """Create a regular (non-admin) user in the test DB."""
    factory = async_sessionmaker(
        db_engine, class_=AsyncSession, expire_on_commit=False
    )
    async with factory() as session:
        user = User(
            email="user@test.com",
            password_hash=hash_password("User1234"),
            role="user",
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)
        return user


@pytest.fixture
def regular_token(regular_user, jwt_secret):
    """Create a real JWT access token for the regular user."""
    return create_access_token(regular_user.id, regular_user.role)


# ---------------------------------------------------------------------------
# API consumer + X-API-Key client
# ---------------------------------------------------------------------------

RAW_API_KEY = "af_test_key_0123456789abcdef"


@pytest.fixture
async def api_consumer(db_engine):
    """Create an ApiConsumer in the test DB with a known raw key."""
    factory = async_sessionmaker(
        db_engine, class_=AsyncSession, expire_on_commit=False
    )
    async with factory() as session:
        consumer = ApiConsumer(
            name="test-consumer",
            api_key=hash_api_key(RAW_API_KEY),
            api_key_prefix=RAW_API_KEY[:8],
            is_active=True,
            rate_limit=100,
        )
        session.add(consumer)
        await session.commit()
        await session.refresh(consumer)
        return consumer


@pytest.fixture
async def apikey_client(app, api_consumer):
    """Authenticated httpx AsyncClient with X-API-Key header."""
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
        headers={"X-API-Key": RAW_API_KEY},
    ) as c:
        yield c
