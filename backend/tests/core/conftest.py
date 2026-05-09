"""Core test fixtures -- re-exports from root conftest.

All shared fixtures (jwt_secret, db_engine, db_session) were promoted
to tests/conftest.py in Phase 3. This file retains only the constant
for backward compatibility with any tests that import it directly.
"""

from tests.conftest import TEST_JWT_SECRET  # noqa: F401
