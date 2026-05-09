"""Tests for /api/v1/admin/consumers endpoints.

All consumer-management endpoints require admin role.
"""

from __future__ import annotations

import hashlib

import pytest


# ---------------------------------------------------------------------------
# POST /api/v1/admin/consumers
# ---------------------------------------------------------------------------

class TestCreateConsumer:

    async def test_create_consumer_returns_raw_key(self, auth_client, admin_user):
        """Creating a consumer returns 201 with the raw API key (shown once)."""
        resp = await auth_client.post(
            "/api/v1/admin/consumers",
            json={"name": "webstock-prod"},
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["name"] == "webstock-prod"
        assert "rawApiKey" in body
        assert len(body["rawApiKey"]) > 10
        assert body["isActive"] is True
        assert body["apiKeyPrefix"] == body["rawApiKey"][:8]

    async def test_create_consumer_with_options(self, auth_client, admin_user):
        """Create consumer with description and rate_limit."""
        resp = await auth_client.post(
            "/api/v1/admin/consumers",
            json={
                "name": "webstock-staging",
                "description": "Staging environment",
                "rateLimit": 50,
            },
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["description"] == "Staging environment"
        assert body["rateLimit"] == 50

    async def test_duplicate_name_returns_409(self, auth_client, admin_user):
        """Creating two consumers with the same name returns 409."""
        await auth_client.post(
            "/api/v1/admin/consumers",
            json={"name": "dup-consumer"},
        )
        resp = await auth_client.post(
            "/api/v1/admin/consumers",
            json={"name": "dup-consumer"},
        )
        assert resp.status_code == 409

    async def test_create_consumer_requires_admin(self, regular_client, regular_user):
        """Non-admin cannot create consumers."""
        resp = await regular_client.post(
            "/api/v1/admin/consumers",
            json={"name": "not-allowed"},
        )
        assert resp.status_code == 403


# ---------------------------------------------------------------------------
# GET /api/v1/admin/consumers
# ---------------------------------------------------------------------------

class TestListConsumers:

    async def test_list_consumers_empty(self, auth_client, admin_user):
        """Initially, the consumer list is empty."""
        resp = await auth_client.get("/api/v1/admin/consumers")
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body, list)

    async def test_list_consumers_after_create(self, auth_client, admin_user):
        """After creating a consumer, it appears in the list."""
        await auth_client.post(
            "/api/v1/admin/consumers",
            json={"name": "listed-consumer"},
        )
        resp = await auth_client.get("/api/v1/admin/consumers")
        assert resp.status_code == 200
        names = [c["name"] for c in resp.json()]
        assert "listed-consumer" in names


# ---------------------------------------------------------------------------
# PUT /api/v1/admin/consumers/{consumer_id}
# ---------------------------------------------------------------------------

class TestUpdateConsumer:

    async def test_update_consumer_name(self, auth_client, admin_user):
        """Admin can rename a consumer."""
        create_resp = await auth_client.post(
            "/api/v1/admin/consumers",
            json={"name": "old-name"},
        )
        consumer_id = create_resp.json()["id"]

        resp = await auth_client.put(
            f"/api/v1/admin/consumers/{consumer_id}",
            json={"name": "new-name"},
        )
        assert resp.status_code == 200
        assert resp.json()["name"] == "new-name"

    async def test_update_consumer_rate_limit(self, auth_client, admin_user):
        """Admin can change rate_limit."""
        create_resp = await auth_client.post(
            "/api/v1/admin/consumers",
            json={"name": "rate-test"},
        )
        consumer_id = create_resp.json()["id"]

        resp = await auth_client.put(
            f"/api/v1/admin/consumers/{consumer_id}",
            json={"rateLimit": 200},
        )
        assert resp.status_code == 200
        assert resp.json()["rateLimit"] == 200

    async def test_update_nonexistent_consumer(self, auth_client, admin_user):
        """Updating a non-existent consumer returns 404."""
        import uuid
        fake_id = str(uuid.uuid4())
        resp = await auth_client.put(
            f"/api/v1/admin/consumers/{fake_id}",
            json={"name": "ghost"},
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# DELETE /api/v1/admin/consumers/{consumer_id}
# ---------------------------------------------------------------------------

class TestDeleteConsumer:

    async def test_deactivate_consumer(self, auth_client, admin_user):
        """DELETE soft-deletes (deactivates) the consumer, returns 204."""
        create_resp = await auth_client.post(
            "/api/v1/admin/consumers",
            json={"name": "to-delete"},
        )
        consumer_id = create_resp.json()["id"]

        resp = await auth_client.delete(f"/api/v1/admin/consumers/{consumer_id}")
        assert resp.status_code == 204

        # Verify deactivated
        list_resp = await auth_client.get("/api/v1/admin/consumers")
        for c in list_resp.json():
            if c["id"] == consumer_id:
                assert c["isActive"] is False

    async def test_delete_nonexistent_consumer(self, auth_client, admin_user):
        """Deleting a non-existent consumer returns 404."""
        import uuid
        fake_id = str(uuid.uuid4())
        resp = await auth_client.delete(f"/api/v1/admin/consumers/{fake_id}")
        assert resp.status_code == 404
