"""Tests for /api/v1/admin/auth endpoints.

Covers login, register, refresh, me, and change-password flows.
"""

from __future__ import annotations

import pytest

from app.core.auth import create_refresh_token, hash_password
from app.models.user import User


# ---------------------------------------------------------------------------
# POST /api/v1/admin/auth/register
# ---------------------------------------------------------------------------

class TestRegister:

    async def test_first_user_becomes_admin(self, client):
        """The very first registered user should be promoted to admin."""
        resp = await client.post(
            "/api/v1/admin/auth/register",
            json={"email": "first@test.com", "password": "Secret123"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["role"] == "admin"
        assert body["email"] == "first@test.com"
        assert body["isActive"] is True

    async def test_second_user_is_regular(self, client, admin_user):
        """Subsequent registrations default to role=user."""
        resp = await client.post(
            "/api/v1/admin/auth/register",
            json={"email": "second@test.com", "password": "Secret123"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["role"] == "user"

    async def test_duplicate_email_returns_409(self, client, admin_user):
        """Registering with an existing email returns 409."""
        resp = await client.post(
            "/api/v1/admin/auth/register",
            json={"email": "admin@test.com", "password": "Whatever1"},
        )
        assert resp.status_code == 409

    async def test_short_password_rejected(self, client):
        """Password shorter than 6 characters should fail validation."""
        resp = await client.post(
            "/api/v1/admin/auth/register",
            json={"email": "short@test.com", "password": "12345"},
        )
        assert resp.status_code == 422

    async def test_invalid_email_rejected(self, client):
        """Malformed email should fail validation."""
        resp = await client.post(
            "/api/v1/admin/auth/register",
            json={"email": "not-an-email", "password": "Secret123"},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /api/v1/admin/auth/login
# ---------------------------------------------------------------------------

class TestLogin:

    async def test_valid_credentials(self, client, admin_user):
        """Correct email + password returns access and refresh tokens."""
        resp = await client.post(
            "/api/v1/admin/auth/login",
            json={"email": "admin@test.com", "password": "Admin123"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "accessToken" in body
        assert "refreshToken" in body
        assert body["tokenType"] == "bearer"

    async def test_wrong_password(self, client, admin_user):
        """Wrong password returns 401."""
        resp = await client.post(
            "/api/v1/admin/auth/login",
            json={"email": "admin@test.com", "password": "WrongPass"},
        )
        assert resp.status_code == 401

    async def test_nonexistent_user(self, client):
        """Login with unregistered email returns 401."""
        resp = await client.post(
            "/api/v1/admin/auth/login",
            json={"email": "nobody@test.com", "password": "Whatever1"},
        )
        assert resp.status_code == 401

    async def test_inactive_user_returns_403(self, client, admin_user, db_engine):
        """Deactivated user cannot log in."""
        from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
        factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)
        async with factory() as session:
            from sqlalchemy import update
            await session.execute(
                update(User).where(User.id == admin_user.id).values(is_active=False)
            )
            await session.commit()

        resp = await client.post(
            "/api/v1/admin/auth/login",
            json={"email": "admin@test.com", "password": "Admin123"},
        )
        assert resp.status_code == 403

    async def test_login_response_contains_both_tokens(self, client, admin_user):
        """Verify the token response schema contains all required fields."""
        resp = await client.post(
            "/api/v1/admin/auth/login",
            json={"email": "admin@test.com", "password": "Admin123"},
        )
        body = resp.json()
        assert len(body["accessToken"]) > 20
        assert len(body["refreshToken"]) > 20


# ---------------------------------------------------------------------------
# POST /api/v1/admin/auth/refresh
# ---------------------------------------------------------------------------

class TestRefresh:

    async def test_valid_refresh(self, client, admin_user, jwt_secret):
        """A valid refresh token yields new access + refresh tokens."""
        refresh = create_refresh_token(admin_user.id)
        resp = await client.post(
            "/api/v1/admin/auth/refresh",
            json={"refreshToken": refresh},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "accessToken" in body
        assert "refreshToken" in body

    async def test_replayed_refresh_token_rejected(self, client, admin_user, jwt_secret):
        """Using the same refresh token twice should fail (jti rotation)."""
        refresh = create_refresh_token(admin_user.id)

        # First use -- should succeed
        resp1 = await client.post(
            "/api/v1/admin/auth/refresh",
            json={"refreshToken": refresh},
        )
        assert resp1.status_code == 200

        # Second use -- jti already claimed
        resp2 = await client.post(
            "/api/v1/admin/auth/refresh",
            json={"refreshToken": refresh},
        )
        assert resp2.status_code == 401

    async def test_invalid_refresh_token(self, client):
        """Garbage token returns 401."""
        resp = await client.post(
            "/api/v1/admin/auth/refresh",
            json={"refreshToken": "not.a.valid.jwt"},
        )
        assert resp.status_code == 401

    async def test_access_token_rejected_as_refresh(self, client, admin_user, admin_token):
        """An access token must not be accepted as a refresh token."""
        resp = await client.post(
            "/api/v1/admin/auth/refresh",
            json={"refreshToken": admin_token},
        )
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# GET /api/v1/admin/auth/me
# ---------------------------------------------------------------------------

class TestMe:

    async def test_me_with_valid_token(self, auth_client, admin_user):
        """GET /me with a valid admin token returns user info."""
        resp = await auth_client.get("/api/v1/admin/auth/me")
        assert resp.status_code == 200
        body = resp.json()
        assert body["email"] == "admin@test.com"
        assert body["role"] == "admin"
        assert body["isActive"] is True

    async def test_me_without_token(self, client):
        """GET /me without Bearer token returns 401."""
        resp = await client.get("/api/v1/admin/auth/me")
        assert resp.status_code == 401

    async def test_me_with_invalid_token(self, client):
        """GET /me with a garbage Bearer token returns 401."""
        resp = await client.get(
            "/api/v1/admin/auth/me",
            headers={"Authorization": "Bearer garbage.token.here"},
        )
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# POST /api/v1/admin/auth/change-password
# ---------------------------------------------------------------------------

class TestChangePassword:

    async def test_successful_password_change(self, auth_client, admin_user):
        """Valid current password + new password succeeds."""
        resp = await auth_client.post(
            "/api/v1/admin/auth/change-password",
            json={"currentPassword": "Admin123", "newPassword": "NewPass456"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["detail"] == "Password changed successfully"

    async def test_wrong_current_password(self, auth_client, admin_user):
        """Incorrect current password returns 400."""
        resp = await auth_client.post(
            "/api/v1/admin/auth/change-password",
            json={"currentPassword": "WrongOld", "newPassword": "NewPass456"},
        )
        assert resp.status_code == 400

    async def test_change_password_requires_auth(self, client):
        """Changing password without a token returns 401."""
        resp = await client.post(
            "/api/v1/admin/auth/change-password",
            json={"currentPassword": "Admin123", "newPassword": "NewPass456"},
        )
        assert resp.status_code == 401
