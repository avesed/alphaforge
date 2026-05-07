"""Model evaluator agent -- decides deploy/retry/reject for trained models.

Given training metrics, config, and data context, produces an EvaluationResult
that determines whether the model should be deployed, retried with adjustments,
or rejected outright.
"""

import json
import logging
from pathlib import Path
from typing import Any

from app.services.ml_agents.llm_client import MLAgentClient, MLAgentError
from app.services.ml_agents.schemas import EvaluationResult

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent / "prompts" / "evaluator_system.md"


class Evaluator:
    """Evaluate trained model metrics and decide on deployment."""

    def __init__(self, client: MLAgentClient | None = None):
        self._client = client or MLAgentClient()
        self._system_prompt: str | None = None

    def _get_system_prompt(self) -> str:
        if self._system_prompt is None:
            self._system_prompt = _PROMPT_PATH.read_text(encoding="utf-8")
        return self._system_prompt

    async def evaluate(
        self,
        market: str,
        training_results: dict[str, Any],
        training_config: dict[str, Any],
        data_profile: dict[str, Any],
        quality_thresholds: dict[str, float],
        is_retry: bool = False,
    ) -> EvaluationResult:
        """Evaluate a trained model and decide deploy/retry/reject.

        Args:
            market: Market code (us, cn, hk).
            training_results: Training metrics (ic, icir, fold_ics, val_ic, etc.).
            training_config: Config used for training (as dict).
            data_profile: Summary data profile context.
            quality_thresholds: min_ic and min_icir thresholds.
            is_retry: Whether this is a retry iteration.

        Returns:
            EvaluationResult with decision, reasoning, and optional adjustments.

        Raises:
            MLAgentError: If LLM call fails.
        """
        llm_input: dict[str, Any] = {
            "market": market,
            "training_results": training_results,
            "training_config": training_config,
            "data_profile": data_profile,
            "quality_thresholds": quality_thresholds,
            "is_retry": is_retry,
        }

        user_message = (
            "Below is the input data. Based on this data and the system prompt, "
            "generate a JSON response with the evaluation fields: "
            "decision, reasoning, suggested_adjustments, confidence.\n\n"
            f"Input data:\n{json.dumps(llm_input)}"
        )

        result = await self._client.chat_json(
            system_prompt=self._get_system_prompt(),
            user_content=user_message,
            temperature=0.1,
            max_tokens=800,
        )

        ev = EvaluationResult(**result)
        logger.info(
            "Evaluator decision for %s: %s (confidence=%.2f, reasoning=%s)",
            market,
            ev.decision,
            ev.confidence,
            ev.reasoning[:80],
        )
        return ev


evaluator = Evaluator()
