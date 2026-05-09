"""Tests for /api/v1/admin/settings endpoints.

Settings use a key-value store. Protected keys (e.g. jwt_secret_key)
cannot be read or modified via the API.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# GET /api/v1/admin/settings
# ---------------------------------------------------------------------------

class TestGetAllSettings:

    async def test_get_all_settings_empty(self, auth_client, admin_user):
        """Initially, settings dict is empty (or lacks protected keys)."""
        resp = await auth_client.get("/api/v1/admin/settings")
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body, dict)
        # jwt_secret_key must never appear in output
        assert "jwt_secret_key" not in body

    async def test_get_all_settings_requires_admin(self, regular_client, regular_user):
        """Non-admin gets 403."""
        resp = await regular_client.get("/api/v1/admin/settings")
        assert resp.status_code == 403


# ---------------------------------------------------------------------------
# PUT /api/v1/admin/settings (bulk update)
# ---------------------------------------------------------------------------

class TestBulkUpdateSettings:

    async def test_bulk_create_settings(self, auth_client, admin_user):
        """Bulk update creates new settings and returns them."""
        resp = await auth_client.put(
            "/api/v1/admin/settings",
            json={"my_key": "my_value", "another": "val2"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["my_key"] == "my_value"
        assert body["another"] == "val2"

    async def test_bulk_update_existing(self, auth_client, admin_user):
        """Bulk update modifies existing settings."""
        await auth_client.put(
            "/api/v1/admin/settings",
            json={"color": "blue"},
        )
        resp = await auth_client.put(
            "/api/v1/admin/settings",
            json={"color": "red"},
        )
        assert resp.status_code == 200
        assert resp.json()["color"] == "red"

    async def test_protected_key_rejected(self, auth_client, admin_user):
        """Attempting to set a protected key returns 403."""
        resp = await auth_client.put(
            "/api/v1/admin/settings",
            json={"jwt_secret_key": "hacked"},
        )
        assert resp.status_code == 403


# ---------------------------------------------------------------------------
# GET /api/v1/admin/settings/{key}
# ---------------------------------------------------------------------------

class TestGetSetting:

    async def test_get_single_setting(self, auth_client, admin_user):
        """Retrieve a single setting by key."""
        await auth_client.put(
            "/api/v1/admin/settings",
            json={"theme": "dark"},
        )
        resp = await auth_client.get("/api/v1/admin/settings/theme")
        assert resp.status_code == 200
        body = resp.json()
        assert body["key"] == "theme"
        assert body["value"] == "dark"

    async def test_get_nonexistent_setting(self, auth_client, admin_user):
        """Requesting a non-existent key returns 404."""
        resp = await auth_client.get("/api/v1/admin/settings/does_not_exist")
        assert resp.status_code == 404

    async def test_get_protected_key_forbidden(self, auth_client, admin_user):
        """Reading a protected key returns 403."""
        resp = await auth_client.get("/api/v1/admin/settings/jwt_secret_key")
        assert resp.status_code == 403


# ---------------------------------------------------------------------------
# PUT /api/v1/admin/settings/{key}
# ---------------------------------------------------------------------------

class TestUpsertSetting:

    async def test_create_single_setting(self, auth_client, admin_user):
        """PUT a new key creates it."""
        resp = await auth_client.put(
            "/api/v1/admin/settings/new_setting",
            json={"value": "hello"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["key"] == "new_setting"
        assert body["value"] == "hello"

    async def test_update_single_setting(self, auth_client, admin_user):
        """PUT an existing key updates its value."""
        await auth_client.put(
            "/api/v1/admin/settings/counter",
            json={"value": "1"},
        )
        resp = await auth_client.put(
            "/api/v1/admin/settings/counter",
            json={"value": "2"},
        )
        assert resp.status_code == 200
        assert resp.json()["value"] == "2"

    async def test_upsert_protected_key_forbidden(self, auth_client, admin_user):
        """Cannot upsert a protected key."""
        resp = await auth_client.put(
            "/api/v1/admin/settings/jwt_secret_key",
            json={"value": "nope"},
        )
        assert resp.status_code == 403
