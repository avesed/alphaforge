"""Root test configuration and fixtures.

Phase 1: Minimal setup for pure-function tests.
Phase 2: Auth layer + core infrastructure fixtures.
Phase 3: API endpoint test fixtures (db_engine, jwt_secret promoted here).
"""
import os
import pytest
import fakeredis.aioredis
from sqlalchemy import event, String, Text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.config import get_settings
from app.core.orm import Base


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """Ensure Settings @lru_cache is fresh for each test."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def test_settings(monkeypatch):
    """Inject test-safe environment variables before Settings loads."""
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite://")
    monkeypatch.setenv("REDIS_URL", "redis://fake:6379/0")
    monkeypatch.setenv("STOCKPULSE_URL", "http://test-stockpulse:8010")
    monkeypatch.setenv("STOCKPULSE_API_KEY", "test-key")
    monkeypatch.setenv("NEWSFORGE_URL", "http://test-newsforge:8080")
    monkeypatch.setenv("NEWSFORGE_API_KEY", "test-key")
    monkeypatch.setenv("LOG_LEVEL", "warning")
    return get_settings()


# ---------------------------------------------------------------------------
# Phase 2 shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
async def fake_redis():
    """Provide a fakeredis async client with decode_responses=True."""
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    yield client
    await client.aclose()


# ---------------------------------------------------------------------------
# Shared DB + JWT fixtures (promoted from tests/core/conftest.py in Phase 3)
# ---------------------------------------------------------------------------

# A deterministic test secret (>= 32 chars as required by MIN_SECRET_LEN)
TEST_JWT_SECRET = "test-secret-key-for-unit-tests-at-least-32-chars-long"


@pytest.fixture
def jwt_secret(monkeypatch):
    """Patch the module-level _jwt_secret so get_jwt_secret() returns our test value."""
    import app.core.secrets as secrets_mod
    monkeypatch.setattr(secrets_mod, "_jwt_secret", TEST_JWT_SECRET)
    return TEST_JWT_SECRET


def _create_tables_sqlite(connection):
    """Create all tables, substituting PG-specific types for SQLite.

    Saves and restores original column types to avoid permanently mutating
    Base.metadata (which is a process-level singleton).
    """
    from sqlalchemy.dialects.postgresql import ARRAY, UUID, JSON
    from sqlalchemy import Text as SAText, String as SAString

    originals: list[tuple] = []

    for table in Base.metadata.tables.values():
        for column in table.columns:
            col_type = column.type
            if isinstance(col_type, ARRAY):
                originals.append((column, col_type))
                column.type = SAText()
            elif isinstance(col_type, UUID):
                originals.append((column, col_type))
                column.type = SAString(36)
            elif isinstance(col_type, JSON):
                originals.append((column, col_type))
                column.type = SAText()

    try:
        Base.metadata.create_all(connection)
    finally:
        for column, original_type in originals:
            column.type = original_type


@pytest.fixture
async def db_engine():
    """Create an in-memory SQLite async engine with all ORM tables.

    PostgreSQL-specific column types (ARRAY, UUID, JSON) are adapted by
    hooking into DDL compilation events.
    """
    import sqlite3
    import uuid as uuid_mod

    # Register adapter so sqlite3 can bind uuid.UUID as TEXT
    sqlite3.register_adapter(uuid_mod.UUID, str)

    engine = create_async_engine(
        "sqlite+aiosqlite://",
        echo=False,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    # Import models to ensure they register with Base.metadata
    import app.models  # noqa: F401

    # Replace PostgreSQL-specific types with SQLite-compatible ones at DDL time.
    @event.listens_for(engine.sync_engine, "connect")
    def _set_sqlite_pragma(dbapi_conn, connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.close()

    # Convert UUID bind parameters to strings for SQLite compatibility.
    # SQLAlchemy's Mapped[uuid.UUID] causes bind values to be processed
    # through its UUID type handler, which strips hyphens (e.g.,
    # "abc12345..." instead of "abc1-2345-..."). We need to re-format
    # these to match the hyphenated form stored by uuid.uuid4().__str__().
    import re
    _HEX32_RE = re.compile(r"^[0-9a-f]{32}$")

    @event.listens_for(engine.sync_engine, "before_cursor_execute", retval=True)
    def _convert_uuid_params(conn, cursor, statement, parameters, context, executemany):
        if parameters is None:
            return statement, parameters

        def _fix(v):
            if isinstance(v, uuid_mod.UUID):
                return str(v)
            if isinstance(v, str) and _HEX32_RE.match(v):
                # Re-insert hyphens: 8-4-4-4-12
                return f"{v[:8]}-{v[8:12]}-{v[12:16]}-{v[16:20]}-{v[20:]}"
            return v

        if isinstance(parameters, dict):
            parameters = {k: _fix(v) for k, v in parameters.items()}
        elif isinstance(parameters, (list, tuple)):
            parameters = tuple(_fix(v) for v in parameters)
        return statement, parameters

    # Create tables, adapting PG types to SQLite-compatible ones
    async with engine.begin() as conn:
        await conn.run_sync(_create_tables_sqlite)

    yield engine

    await engine.dispose()


@pytest.fixture
async def db_session(db_engine):
    """Provide an async DB session backed by the in-memory SQLite engine."""
    factory = async_sessionmaker(
        db_engine, class_=AsyncSession, expire_on_commit=False
    )
    async with factory() as session:
        yield session
        await session.rollback()
