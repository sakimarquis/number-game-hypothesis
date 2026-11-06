"""Abstract backend interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from configs.settings import GREEDY, GenerationSamplingConfig

# Surface forms of the binary answer labels; the first of each is the canonical choice.
BINARY_VARIANTS = {"yes": ("Yes", "yes", "YES"), "no": ("No", "no", "NO")}


@dataclass
class GenerateResult:
    """Return value from Backend.generate().

    `thinking_tokens` is None when the backend cannot count them. `forced_close`
    means thinking reached its budget and was closed by our code; `truncated`
    means the answer stopped at its output cap.
    """
    text: str
    thinking_text: str = ""
    thinking_tokens: int | None = None
    forced_close: bool = False
    truncated: bool = False

    def thinking_record(self) -> dict:
        return {"thinking_tokens": self.thinking_tokens, "forced_close": self.forced_close, "truncated": self.truncated}


def assemble_binary_scores(results) -> tuple[np.ndarray, int]:
    """Combine (p_yes, fell_back) results into (score vector, fallback count)."""
    return (
        np.asarray([result[0] for result in results], dtype=np.float32),
        sum(result[1] for result in results),
    )


class Backend(ABC):
    """Base class for LLM backends.

    All public generation and scoring methods are async and take a list of
    prompts. Subclasses implement generate(); logprob backends also implement
    score_binary() and the shared-prefix hooks.
    """
    supports_per_target_thinking = True
    supports_json_schema = False
    enable_thinking = False
    reasoning = False
    # Gemini Batch API execution: requests are submitted as condition-wide jobs.
    batch_api_enabled = False
    # Online text-trial requests are checkpointed and failed keys retried when > 0.
    text_trial_retry_sweeps = 0
    # "vector" asks for a full binary array; "yes_index" asks for {"yes": [targets]}.
    text_trial_encoding = "vector"
    minimum_thinking_headroom_tokens = 0
    structured_recovery_thinking_headroom_tokens = 0

    @property
    def checkpoints_text_trials(self) -> bool:
        return self.text_trial_retry_sweeps > 0

    def completion_token_limit(self, answer_tokens: int, thinking_budget: int) -> int:
        """Provider output cap for one request with the given answer and thinking budgets."""
        return answer_tokens + thinking_budget

    def is_retryable_generation_error(self, exc: Exception) -> bool:
        return False

    async def release_batch_sidecars(self, condition) -> None:
        """Remove completed Batch sidecars after the condition's canonical cache is published."""

    def _effective_thinking_budget(self, thinking_budget: int | None) -> int:
        if thinking_budget is not None:
            return thinking_budget
        if self.enable_thinking or self.reasoning:
            raise ValueError("thinking_budget is required for thinking/reasoning calls.")
        return 0

    @abstractmethod
    async def generate(self, messages_list: list[list[dict]], max_tokens: int = 1,
                       *, thinking_budget: int | None = None,
                       generation_config: GenerationSamplingConfig = GREEDY,
                       response_json_schema: dict | None = None,
                       batch_key: str | None = None) -> list[GenerateResult]:
        """Generate one result per prompt; `batch_key` names a Gemini Batch job and is ignored elsewhere."""
        ...

    async def score_binary(self, messages_list: list[list[dict]],
                           *, thinking_budget: int | None = None) -> tuple[np.ndarray, int]:
        """Return (p_yes, n_fallback) for a batch of prompts."""
        raise NotImplementedError(f"{type(self).__name__} does not support score_binary")

    async def score_binary_with_shared_thinking(
        self,
        thinking_messages: list[dict],
        messages_list: list[list[dict]],
        *,
        thinking_budget: int | None = None,
    ) -> tuple[np.ndarray, int, dict]:
        """Generate one shared thought, splice it before each target, then score answer-only.

        Returns (p_yes, n_fallback, thinking record of the shared thought).
        """
        thinking_budget = self._effective_thinking_budget(thinking_budget)
        thinking_message, thinking_record = await self._shared_thinking_message(thinking_messages, thinking_budget)
        continued = [[*thinking_messages, thinking_message, messages[-1]] for messages in messages_list]
        scores, n_fallback = await self._score_binary_after_shared_thinking(continued)
        return scores, n_fallback, thinking_record

    async def _shared_thinking_message(self, messages: list[dict], thinking_budget: int) -> tuple[dict, dict]:
        """Return (assistant message holding only the thinking block, its thinking record)."""
        raise NotImplementedError(f"{type(self).__name__} does not support shared-prefix thinking")

    async def _score_binary_after_shared_thinking(self, continued_messages: list[list[dict]]) -> tuple[np.ndarray, int]:
        raise NotImplementedError(f"{type(self).__name__} does not support shared-prefix thinking")
