"""Tests for app.core.orm -- async session generator and Base.

~5 tests covering:
- get_db yields an AsyncSession
- Session is usable for basic queries
- Base metadata contains expected tables after model import
"""

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.orm import Base


class TestGetDb:
    async def test_yields_async_session(self, db_engine):
        """get_db pattern: factory -> context manager -> session."""
        factory = async_sessionmaker(
            db_engine, class_=AsyncSession, expire_on_commit=False
        )
        async with factory() as session:
            assert isinstance(session, AsyncSession)

    async def test_session_can_execute_query(self, db_session):
        """A basic SQL query should succeed against the in-memory SQLite DB."""
        result = await db_session.execute(text("SELECT 1"))
        row = result.scalar()
        assert row == 1

    async def test_session_can_read_tables(self, db_session):
        """Verify that ORM tables were created (users, system_settings, etc.)."""
        result = await db_session.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        )
        table_names = {row[0] for row in result.fetchall()}
        assert "users" in table_names
        assert "system_settings" in table_names
        assert "api_consumers" in table_names


class TestBaseMetadata:
    def test_base_has_registered_tables(self):
        """After importing models, Base.metadata should know about our tables."""
        import app.models  # noqa: F401
        table_names = set(Base.metadata.tables.keys())
        assert "users" in table_names
        assert "api_consumers" in table_names
        assert "system_settings" in table_names

    def test_user_table_has_expected_columns(self):
        import app.models  # noqa: F401
        user_table = Base.metadata.tables["users"]
        col_names = {c.name for c in user_table.columns}
        assert "id" in col_names
        assert "email" in col_names
        assert "password_hash" in col_names
        assert "role" in col_names
        assert "is_active" in col_names
