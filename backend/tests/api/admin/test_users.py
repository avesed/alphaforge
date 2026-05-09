"""Tests for /api/v1/admin/users endpoints.

All user-management endpoints require admin role (``require_admin``).
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# GET /api/v1/admin/users
# ---------------------------------------------------------------------------

class TestListUsers:

    async def test_list_users_as_admin(self, auth_client, admin_user):
        """Admin can list all users."""
        resp = await auth_client.get("/api/v1/admin/users")
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body, list)
        assert len(body) >= 1
        emails = [u["email"] for u in body]
        assert "admin@test.com" in emails

    async def test_list_users_forbidden_for_regular_user(self, regular_client, regular_user):
        """Non-admin gets 403."""
        resp = await regular_client.get("/api/v1/admin/users")
        assert resp.status_code == 403

    async def test_list_users_unauthorized_without_token(self, client):
        """No token returns 401."""
        resp = await client.get("/api/v1/admin/users")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# GET /api/v1/admin/users/{user_id}
# ---------------------------------------------------------------------------

class TestGetUser:

    async def test_get_existing_user(self, auth_client, admin_user):
        """Admin can retrieve a single user by ID."""
        resp = await auth_client.get(f"/api/v1/admin/users/{admin_user.id}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["email"] == "admin@test.com"

    async def test_get_nonexistent_user(self, auth_client, admin_user):
        """Requesting a non-existent user_id returns 404."""
        resp = await auth_client.get("/api/v1/admin/users/99999")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# PUT /api/v1/admin/users/{user_id}
# ---------------------------------------------------------------------------

class TestUpdateUser:

    async def test_update_user_role(self, auth_client, admin_user, regular_user):
        """Admin can change another user's role."""
        resp = await auth_client.put(
            f"/api/v1/admin/users/{regular_user.id}",
            json={"role": "admin"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["role"] == "admin"

    async def test_update_user_locale(self, auth_client, admin_user, regular_user):
        """Admin can update a user's locale."""
        resp = await auth_client.put(
            f"/api/v1/admin/users/{regular_user.id}",
            json={"locale": "en"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["locale"] == "en"

    async def test_cannot_modify_own_role(self, auth_client, admin_user):
        """Admin cannot change their own role."""
        resp = await auth_client.put(
            f"/api/v1/admin/users/{admin_user.id}",
            json={"role": "user"},
        )
        assert resp.status_code == 400

    async def test_cannot_deactivate_self(self, auth_client, admin_user):
        """Admin cannot set their own is_active to False."""
        resp = await auth_client.put(
            f"/api/v1/admin/users/{admin_user.id}",
            json={"isActive": False},
        )
        assert resp.status_code == 400

    async def test_invalid_role_rejected(self, auth_client, admin_user, regular_user):
        """An invalid role string returns 400."""
        resp = await auth_client.put(
            f"/api/v1/admin/users/{regular_user.id}",
            json={"role": "superadmin"},
        )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# DELETE /api/v1/admin/users/{user_id}
# ---------------------------------------------------------------------------

class TestDeleteUser:

    async def test_deactivate_user(self, auth_client, admin_user, regular_user):
        """DELETE soft-deletes the user (sets is_active=False), returns 204."""
        resp = await auth_client.delete(f"/api/v1/admin/users/{regular_user.id}")
        assert resp.status_code == 204

        # Verify the user is deactivated
        resp2 = await auth_client.get(f"/api/v1/admin/users/{regular_user.id}")
        assert resp2.status_code == 200
        assert resp2.json()["isActive"] is False

    async def test_cannot_deactivate_self(self, auth_client, admin_user):
        """Admin cannot delete (deactivate) themselves."""
        resp = await auth_client.delete(f"/api/v1/admin/users/{admin_user.id}")
        assert resp.status_code == 400

    async def test_deactivate_nonexistent_user(self, auth_client, admin_user):
        """Deleting a non-existent user returns 404."""
        resp = await auth_client.delete("/api/v1/admin/users/99999")
        assert resp.status_code == 404
