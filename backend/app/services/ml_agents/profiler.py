"""Data profiling agent -- analyzes feature matrix statistics.

The Profiler does two things:
1. Python computation: NaN rates, return stats, sector distribution (no LLM needed)
2. LLM analysis: regime characterization + data quality warnings (1 LLM call)
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.services.ml_agents.llm_client import MLAgentClient, MLAgentError
from app.services.ml_agents.schemas import DataProfile

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent / "prompts" / "profiler_system.md"


class Profiler:
    """Analyze feature matrix and produce a DataProfile."""

    def __init__(self, client: MLAgentClient | None = None):
        self._client = client or MLAgentClient()
        self._system_prompt: str | None = None

    def _get_system_prompt(self) -> str:
        if self._system_prompt is None:
            self._system_prompt = _PROMPT_PATH.read_text(encoding="utf-8")
        return self._system_prompt

    async def analyze(
        self,
        feature_df: pd.DataFrame,
        market: str,
        symbols: list[str],
        recent_model_ics: list[float] | None = None,
    ) -> DataProfile:
        """Profile the feature matrix.

        Args:
            feature_df: Feature matrix from build_feature_matrix() (already rank-transformed).
                        Must have columns: symbol, date, + feature columns.
            market: Market code.
            symbols: Symbol list.
            recent_model_ics: Recent model IC values (most recent last) for trend detection.

        Returns:
            DataProfile with computed statistics + LLM analysis.
        """
        # 1. Compute statistics from the feature DataFrame
        stats = self._compute_stats(feature_df, market, symbols, recent_model_ics)

        # 2. Call LLM for regime analysis + quality warnings
        try:
            llm_result = await self._llm_analyze(stats)
            stats["regime_analysis"] = llm_result.get("regime_analysis", "")
            stats["data_quality_warnings"] = llm_result.get(
                "data_quality_warnings", []
            )
        except (MLAgentError, Exception) as e:
            logger.warning("Profiler LLM analysis failed, using empty: %s", e)
            stats["regime_analysis"] = ""
            stats["data_quality_warnings"] = []

        return DataProfile(**stats)

    def _compute_stats(
        self,
        feature_df: pd.DataFrame,
        market: str,
        symbols: list[str],
        recent_model_ics: list[float] | None,
    ) -> dict[str, Any]:
        """Compute all statistics from the feature DataFrame (no LLM)."""
        feature_cols = [
            c for c in feature_df.columns if c not in ("symbol", "date")
        ]

        # NaN rates per feature
        nan_rates: dict[str, float] = {}
        if feature_cols:
            nan_series = feature_df[feature_cols].isna().mean()
            nan_rates = {
                col: round(float(v), 4) for col, v in nan_series.items()
            }

        median_nan = (
            float(np.median(list(nan_rates.values()))) if nan_rates else 0.0
        )
        sparse_features = [
            col for col, rate in nan_rates.items() if rate > 0.70
        ]

        # Date range
        dates = pd.to_datetime(feature_df["date"])
        date_range = (
            dates.min().strftime("%Y-%m-%d"),
            dates.max().strftime("%Y-%m-%d"),
        )
        n_trading_days = feature_df["date"].nunique()

        # Sector distribution (try to get from fundamental service)
        # For now, just count unique symbols -- sector detail comes from feature data
        sector_distribution: dict[str, int] = {}
        min_sector_size = 0
        # If there are sector-related columns or we can infer, compute
        # Otherwise leave empty -- the LLM can still analyze other signals

        # Return stats -- compute from forward_return if available, else from a proxy
        return_stats: dict[str, float] = {}
        # Use a technical return feature as proxy for return distribution
        for ret_col in ["ret5", "ret20", "forward_return"]:
            if ret_col in feature_df.columns:
                vals = feature_df[ret_col].dropna()
                if len(vals) > 100:
                    return_stats = {
                        "mean": round(float(vals.mean()), 6),
                        "std": round(float(vals.std()), 6),
                        "skew": round(float(vals.skew()), 4),
                        "kurtosis": round(float(vals.kurtosis()), 4),
                        "pct_positive": round(float((vals > 0).mean()), 4),
                    }
                    break

        return {
            "market": market,
            "universe_size": len(symbols),
            "n_trading_days": n_trading_days,
            "date_range": date_range,
            "feature_nan_rates": nan_rates,
            "median_nan_rate": round(median_nan, 4),
            "sparse_features": sparse_features,
            "sector_distribution": sector_distribution,
            "min_sector_size": min_sector_size,
            "return_stats": return_stats,
            "recent_model_ics": recent_model_ics,
        }

    async def _llm_analyze(self, stats: dict[str, Any]) -> dict[str, Any]:
        """Call LLM for regime analysis and data quality warnings."""
        # Build compact input for LLM (exclude full nan_rates, just summary)
        llm_input: dict[str, Any] = {
            "market": stats["market"],
            "universe_size": stats["universe_size"],
            "date_range": {
                "start": stats["date_range"][0],
                "end": stats["date_range"][1],
                "trading_days": stats["n_trading_days"],
            },
            "return_stats": stats["return_stats"],
        }
        # Include top sparse features (not all nan rates -- too long)
        if stats["sparse_features"]:
            llm_input["feature_nan_rates"] = {
                f: stats["feature_nan_rates"][f]
                for f in stats["sparse_features"][:15]
            }

        if stats["sector_distribution"]:
            llm_input["sector_distribution"] = stats["sector_distribution"]

        if stats["recent_model_ics"]:
            llm_input["recent_model_ics"] = stats["recent_model_ics"]

        # Wrap data with explicit output instruction to prevent LLM
        # from echoing the input JSON in json_object mode
        user_message = (
            "Below is the input data. Based on this data and the system prompt, "
            "generate a JSON response with these exact fields: "
            "regime_analysis, data_quality_warnings.\n\n"
            f"Input data:\n{json.dumps(llm_input)}"
        )

        result = await self._client.chat_json(
            system_prompt=self._get_system_prompt(),
            user_content=user_message,
            temperature=0.1,
            max_tokens=800,
        )
        return result


# Module singleton
profiler = Profiler()
