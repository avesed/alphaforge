"""Qlib initialization context manager.

Qlib's qlib.init() sets global state -- only one market/region can be active at a time
in a single process. QlibContext serializes all init calls via a threading.Lock and
tracks the current region to avoid unnecessary re-initialization.

The MARKET_TO_REGION mapping consolidates Shanghai (sh) and Shenzhen (sz) markets
into "cn" since they share the same trading calendar and data directory.
"""
import logging
import os
import threading
from typing import Optional

logger = logging.getLogger(__name__)


class QlibContext:
    """Process-level Qlib initialization manager.

    Thread-safe: uses a threading.Lock to serialize qlib.init() calls.
    Re-init is only performed when the target region differs from the current one.
    """

    _lock = threading.Lock()
    _current_region: Optional[str] = None
    _current_provider_uri: Optional[str] = None
    _initialized: bool = False

    # Market codes -> Qlib region mapping
    # Qlib only supports two built-in regions: "cn" and "us".
    # HK and metals use "us" region for init (actual calendar comes from provider_uri).
    MARKET_TO_REGION = {
        "us": "us",
        "hk": "us",
        "sh": "cn",
        "sz": "cn",
        "cn": "cn",
        "metal": "us",
    }

    # Data directory per market (not per region, since multiple markets share a region)
    MARKET_TO_DATA_DIR = {
        "us": "us_data",
        "hk": "hk_data",
        "sh": "cn_data",
        "sz": "cn_data",
        "cn": "cn_data",
        "metal": "metal_data",
    }

    # Canonical markets with distinct data directories (for status reporting)
    STATUS_MARKETS = {
        "us": "us_data",
        "hk": "hk_data",
        "cn": "cn_data",
        "metal": "metal_data",
    }

    @classmethod
    def ensure_init(cls, market: str, data_dir: Optional[str] = None) -> None:
        """Ensure Qlib is initialized for the given market.

        Args:
            market: Market code (us, hk, sh, sz, cn, metal)
            data_dir: Override base data directory (default from QLIB_DATA_DIR env)
        """
        region = cls.MARKET_TO_REGION.get(market)
        if region is None:
            raise ValueError(
                f"Unknown market: {market}. Valid: {list(cls.MARKET_TO_REGION.keys())}"
            )

        base_dir = data_dir or os.environ.get("QLIB_DATA_DIR", "/app/data/qlib")
        provider_uri = os.path.join(base_dir, cls.MARKET_TO_DATA_DIR[market])

        with cls._lock:
            if cls._current_provider_uri == provider_uri and cls._initialized:
                logger.debug(
                    "Qlib already initialized for provider_uri=%s, skipping",
                    provider_uri,
                )
                return

            logger.info(
                "Initializing Qlib for region=%s (market=%s), provider_uri=%s",
                region,
                market,
                provider_uri,
            )
            try:
                import qlib

                qlib.init(provider_uri=provider_uri, region=region)
                cls._current_region = region
                cls._current_provider_uri = provider_uri
                cls._initialized = True
                logger.info(
                    "Qlib initialized successfully for region=%s", region
                )
            except Exception as e:
                logger.error(
                    "Failed to initialize Qlib for region=%s: %s", region, e
                )
                cls._initialized = False
                cls._current_region = None
                cls._current_provider_uri = None
                raise

    @classmethod
    def get_current_region(cls) -> Optional[str]:
        """Return the currently active Qlib region, or None if not initialized."""
        return cls._current_region

    @classmethod
    def is_initialized(cls) -> bool:
        return cls._initialized

    @classmethod
    def reset(cls) -> None:
        """Reset state (for testing)."""
        with cls._lock:
            cls._current_region = None
            cls._current_provider_uri = None
            cls._initialized = False
