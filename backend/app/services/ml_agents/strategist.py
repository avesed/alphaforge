"""Training strategist agent -- generates LightGBM training config from data profile.

Given a DataProfile (from the Profiler) and optional previous evaluation feedback,
produces a TrainingConfig with optimized hyperparameters via a single LLM call.
"""

import json
import logging
from pathlib import Path
from typing import Any

from app.services.ml_agents.llm_client import MLAgentClient, MLAgentError
from app.services.ml_agents.schemas import DataProfile, TrainingConfig
from app.services.market_config import get_market_config

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent / "prompts" / "strategist_system.md"


class Strategist:
    """Generate training configuration from a data profile."""

    def __init__(self, client: MLAgentClient | None = None):
        self._client = client or MLAgentClient()
        self._system_prompt: str | None = None

    def _get_system_prompt(self) -> str:
        if self._system_prompt is None:
            self._system_prompt = _PROMPT_PATH.read_text(encoding="utf-8")
        return self._system_prompt

    async def generate(
        self,
        profile: DataProfile | None,
        market: str,
        previous_evaluation: dict[str, Any] | None = None,
    ) -> TrainingConfig:
        """Generate a TrainingConfig for the given market and data profile.

        Args:
            profile: DataProfile from the Profiler (None if profiling failed).
            market: Market code (us, cn, hk).
            previous_evaluation: Feedback from a prior evaluator run (for retry iterations).

        Returns:
            TrainingConfig with LLM-optimized parameters.

        Raises:
            MLAgentError: If LLM call fails (caller should catch and use defaults).
        """
        base = get_market_config(market)

        if profile is None:
            raise MLAgentError("No data profile available for strategist")

        llm_input: dict[str, Any] = {
            "market": market,
            "data_profile": {
                "universe_size": profile.universe_size,
                "n_trading_days": profile.n_trading_days,
                "median_nan_rate": profile.median_nan_rate,
                "sparse_features": profile.sparse_features[:10],
                "return_stats": profile.return_stats,
                "regime_analysis": profile.regime_analysis[:200],
                "data_quality_warnings": profile.data_quality_warnings[:5],
            },
        }

        if previous_evaluation:
            llm_input["previous_evaluation"] = previous_evaluation

        user_message = (
            "Below is the input data. Based on this data and the system prompt, "
            "generate a JSON response with the training configuration fields.\n\n"
            f"Input data:\n{json.dumps(llm_input)}"
        )

        result = await self._client.chat_json(
            system_prompt=self._get_system_prompt(),
            user_content=user_message,
            temperature=0.2,
            max_tokens=1000,
        )

        tc = TrainingConfig(**result)
        logger.info(
            "Strategist generated config for %s: boost=%d, leaves=%s, reasoning=%s",
            market,
            tc.num_boost_round,
            tc.lgb_overrides.get("num_leaves", "default"),
            tc.reasoning[:80],
        )
        return tc


strategist = Strategist()
