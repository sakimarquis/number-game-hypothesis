"""Abstract backend interface."""

import asyncio
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from configs.settings import GenerationSamplingConfig


@dataclass
class GenerateResult:
    """Return value from Backend.generate()."""
    text: str
    thinking_text: str = ""


def binary_token_label(token: str) -> str | None:
    text = token.strip()
    while text.startswith(("Ġ", "▁")):
        text = text[1:].lstrip()
    label = text.strip().casefold()
    return label if label in {"yes", "no"} else None


def score_topk_pairs(pairs) -> tuple[float, bool]:
    """Exp-sum p(yes) over (token_str, logprob) pairs; True when no pair is binary."""
    yes = 0.0
    no = 0.0
    for token, logprob in pairs:
        label = binary_token_label(token)
        if label == "yes":
            yes += math.exp(float(logprob))
        elif label == "no":
            no += math.exp(float(logprob))
    total = yes + no
    if total == 0.0:
        return 0.5, True
    return yes / total, False


def assemble_binary_scores(results) -> tuple[np.ndarray, int]:
    """Combine (p_yes, fell_back) results into (score vector, fallback count)."""
    return (
        np.asarray([result[0] for result in results], dtype=np.float32),
        sum(result[1] for result in results),
    )


async def warm_cache_then_gather(score_one, messages_list) -> tuple[np.ndarray, int]:
    """Await the first scoring call alone so its shared preamble warms the
    provider prompt cache, then burst the rest concurrently — subsequent calls
    hit cache instead of every first-wave request missing simultaneously."""
    if not messages_list:
        return np.empty(0, dtype=np.float32), 0
    first = await score_one(messages_list[0])
    rest = await asyncio.gather(*[score_one(m) for m in messages_list[1:]])
    return assemble_binary_scores([first, *rest])


class Backend(ABC):
    """Base class for LLM backends.

    All public generation and scoring methods are async. Subclasses implement
    generate(); logprob backends also implement score_binary() and the
    shared-prefix hooks.
    """
    # True = caller chunks requests by `batch_size` (HF GPU forward passes).
    # False = backend owns request concurrency internally.
    caller_batched = False
    batch_size: int = 1
    supports_per_target_thinking = True
    supports_logprobs = True
    supports_json_schema = False
    enable_thinking = False
    is_reasoning = False
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
        if self.enable_thinking or self.is_reasoning:
            raise ValueError("thinking_budget is required for thinking/reasoning calls.")
        return 0

    @staticmethod
    def _continued_messages(
        thinking_messages: list[dict],
        thinking_message: dict,
        messages_list: list[list[dict]],
    ) -> list[list[dict]]:
        return [
            [*thinking_messages, thinking_message, messages[-1]]
            for messages in messages_list
        ]

    @abstractmethod
    async def generate(self, messages: list[dict], max_tokens: int = 1,
                       temperature: float = 0.0, *, thinking_budget: int | None = None,
                       generation_config: GenerationSamplingConfig | None = None,
                       response_json_schema: dict | None = None) -> GenerateResult:
        ...

    async def generate_batch(self, messages_list: list[list[dict]], max_tokens: int = 1,
                             temperature: float = 0.0, *, thinking_budget: int | None = None,
                             generation_config: GenerationSamplingConfig | None = None,
                             response_json_schema: dict | None = None,
                             batch_key: str | None = None) -> list[GenerateResult]:
        """Generate all prompts; the default runs concurrent generate() calls."""
        return await asyncio.gather(*[
            self.generate(
                messages, max_tokens=max_tokens, temperature=temperature,
                thinking_budget=thinking_budget, generation_config=generation_config,
                response_json_schema=response_json_schema,
            )
            for messages in messages_list
        ])

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
    ) -> tuple[np.ndarray, int]:
        """Generate one shared thought, splice it before each target, then score answer-only."""
        thinking_budget = self._effective_thinking_budget(thinking_budget)
        thinking_message = await self._shared_thinking_message(thinking_messages, thinking_budget)
        continued = self._continued_messages(thinking_messages, thinking_message, messages_list)
        return await self._score_binary_after_shared_thinking(continued)

    async def _shared_thinking_message(self, messages: list[dict], thinking_budget: int) -> dict:
        raise NotImplementedError(f"{type(self).__name__} does not support shared-prefix thinking")

    async def _score_binary_after_shared_thinking(self, continued_messages: list[list[dict]]) -> tuple[np.ndarray, int]:
        raise NotImplementedError(f"{type(self).__name__} does not support shared-prefix thinking")
