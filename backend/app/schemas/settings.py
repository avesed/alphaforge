"""Schemas for system settings management."""

from __future__ import annotations

from datetime import datetime

from app.schemas.base import CamelModel


class SettingResponse(CamelModel):
    key: str
    value: str


class SettingUpdate(CamelModel):
    value: str
