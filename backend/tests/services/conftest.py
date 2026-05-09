"""Service-level test fixtures.

Provides mock HTTP clients for external API dependencies
(StockPulse, NewsForge) and common service test helpers.

Also stubs out ``lightgbm`` at the sys.modules level so that service
modules which ``import lightgbm as lgb`` at the top can be loaded in
environments where the heavy C++ library is not installed.  The stub
is inserted **only if** the real package is missing.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# lightgbm stub — allows importing prediction_service / direction_service
# without the C++ library.  Must happen before any ``from app.services...``
# import that transitively touches lightgbm.
# ---------------------------------------------------------------------------
if "lightgbm" not in sys.modules:
    try:
        import lightgbm  # noqa: F401
    except ImportError:
        sys.modules["lightgbm"] = MagicMock()
