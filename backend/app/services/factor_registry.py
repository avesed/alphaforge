"""Factor registry for RD-Agent discovered factors.

Manages the lifecycle of factors discovered by RD-Agent:
- Registration of new factors (expression, IC, ICIR, metadata)
- Activation/deactivation for production use
- Retrieval for feature_service integration

Factors are stored in the discovered_factors table and cached in Redis.
Active factors are automatically included in the feature matrix during
the next LightGBM training run.

AlphaForge adaptation:
- Uses get_db_pool() (asyncpg pool) instead of settings_cache.pool.
"""

import json
import logging
from typing import Any

from app.core.database import get_db_pool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SQL queries (asyncpg parameterized: $1, $2, ...)
# ---------------------------------------------------------------------------

_SQL_UPSERT_FACTOR = """
INSERT INTO discovered_factors (
    name, expression, market, ic, icir,
    source, is_active, metadata
) VALUES ($1, $2, $3, $4, $5, $6, TRUE, $7::jsonb)
ON CONFLICT DO NOTHING
RETURNING id
"""

_SQL_GET_ACTIVE_FACTORS = """
SELECT id, name, expression, market,
       ic, icir, source, is_active, metadata, created_at
FROM discovered_factors
WHERE market = $1 AND is_active = true
ORDER BY ABS(ic) DESC
"""

_SQL_GET_ALL_FACTORS = """
SELECT id, name, expression, market,
       ic, icir, source, is_active, metadata, created_at
FROM discovered_factors
ORDER BY created_at DESC
"""

_SQL_GET_ALL_FACTORS_BY_MARKET = """
SELECT id, name, expression, market,
       ic, icir, source, is_active, metadata, created_at
FROM discovered_factors
WHERE market = $1
ORDER BY created_at DESC
"""

_SQL_TOGGLE_FACTOR = """
UPDATE discovered_factors SET is_active = $1
WHERE id = $2
RETURNING id
"""


def _row_to_dict(row) -> dict[str, Any]:
    """Convert an asyncpg Record to a JSON-safe dict."""
    return {
        "id": str(row["id"]),
        "name": row["name"],
        "expression": row["expression"],
        "market": row["market"],
        "ic": float(row["ic"]) if row["ic"] is not None else None,
        "icir": float(row["icir"]) if row["icir"] is not None else None,
        "source": row.get("source", "rdagent"),
        "is_active": row["is_active"],
        "metadata": (
            json.loads(row["metadata"]) if isinstance(row["metadata"], str)
            else row["metadata"]
        ),
        "created_at": row["created_at"].isoformat() if row["created_at"] else None,
    }


class FactorRegistry:
    """Registry for RD-Agent discovered factors.

    Uses the asyncpg pool from get_db_pool() for all DB operations.
    All methods are async and safe for concurrent use.
    """

    async def register_factor(
        self,
        name: str,
        expression: str,
        market: str,
        ic: float,
        icir: float,
        source: str = "rdagent",
        metadata: dict | None = None,
        **kwargs,
    ) -> str:
        """Register a single discovered factor.

        Args:
            name: Human-readable factor name.
            expression: Qlib expression string.
            market: Market code (us, hk, cn).
            ic: Information Coefficient.
            icir: IC Information Ratio.
            source: Discovery source (rdagent/manual).
            metadata: Optional extra metadata dict.

        Returns:
            Factor ID string.
        """
        pool = get_db_pool()
        meta_json = json.dumps(metadata) if metadata else None

        async with pool.acquire(timeout=10) as conn:
            row = await conn.fetchrow(
                _SQL_UPSERT_FACTOR,
                name,
                expression,
                market,
                ic,
                icir,
                source,
                meta_json,
            )

        if row is None:
            logger.info("Factor already exists: name=%s, market=%s", name, market)
            return ""

        factor_id = str(row["id"])
        logger.info(
            "Registered factor: name=%s, market=%s, ic=%.4f, icir=%.4f, id=%s",
            name, market, ic, icir, factor_id,
        )
        return factor_id

    async def get_active_factors(self, market: str) -> list[dict]:
        """Get all active factors for a market, ordered by |IC| descending.

        Args:
            market: Market code (us, hk, cn).

        Returns:
            List of factor dicts with id, name, expression, ic, icir, etc.
        """
        try:
            pool = get_db_pool()
        except RuntimeError:
            return []

        async with pool.acquire(timeout=10) as conn:
            rows = await conn.fetch(_SQL_GET_ACTIVE_FACTORS, market)

        return [_row_to_dict(r) for r in rows]

    async def get_all_factors(self, market: str | None = None) -> list[dict]:
        """Get all factors, optionally filtered by market.

        Args:
            market: If provided, filter to this market only.

        Returns:
            List of factor dicts ordered by created_at descending.
        """
        try:
            pool = get_db_pool()
        except RuntimeError:
            return []

        async with pool.acquire(timeout=10) as conn:
            if market:
                rows = await conn.fetch(_SQL_GET_ALL_FACTORS_BY_MARKET, market)
            else:
                rows = await conn.fetch(_SQL_GET_ALL_FACTORS)

        return [_row_to_dict(r) for r in rows]

    async def toggle_factor(self, factor_id: str, is_active: bool) -> bool:
        """Activate or deactivate a factor for production use.

        Args:
            factor_id: Factor UUID string.
            is_active: New activation state.

        Returns:
            True if factor was found and updated, False if not found.
        """
        try:
            pool = get_db_pool()
        except RuntimeError:
            return False

        async with pool.acquire(timeout=10) as conn:
            row = await conn.fetchrow(_SQL_TOGGLE_FACTOR, is_active, factor_id)

        if row:
            logger.info(
                "Factor %s set is_active=%s", factor_id, is_active,
            )
            return True

        logger.warning("Factor not found: %s", factor_id)
        return False

    async def register_batch(
        self,
        factors: list[dict],
        market: str,
        **kwargs,
    ) -> int:
        """Register multiple factors from RD-Agent output."""
        registered = 0
        for f in factors:
            try:
                await self.register_factor(
                    name=f["name"],
                    expression=f["expression"],
                    market=market,
                    ic=float(f.get("ic", 0.0)),
                    icir=float(f.get("icir", 0.0)),
                    source=f.get("source", "rdagent"),
                    metadata=f.get("metadata"),
                )
                registered += 1
            except Exception as e:
                logger.warning(
                    "Failed to register factor %s: %s",
                    f.get("name", "?"), e,
                )

        logger.info(
            "Batch registration complete: %d/%d factors for market=%s",
            registered, len(factors), market,
        )
        return registered


# Module singleton
factor_registry = FactorRegistry()
