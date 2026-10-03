"""OpenAI-compatible Chat Completions backend."""

import asyncio
import math
import os

import numpy as np
from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI

from backends.base import (
    Backend,
    GenerateResult,
    assemble_binary_scores,
    score_topk_pairs,
    warm_cache_then_gather,
)
from backends.thinking_codec import ThinkingProtocol, resolve_api_thinking_protocol
from configs.backend_registry import API_PROVIDER_CONFIGS
from configs.settings import GenerationSamplingConfig

OPENAI_NON_REASONING_MIN_COMPLETION_TOKENS = 32
OPENROUTER_REASONING_MIN_TOKENS = 1024
OPENROUTER_REASONING_EFFORT_PCT = {
    "minimal": 10,
    "low": 20,
    "medium": 50,
    "high": 80,
    "xhigh": 95,
    "max": 95,
}
BINARY_CHOICES = ["Yes", "No"]


def _debug_top_logprobs(top) -> None:
    if not os.environ.get("DEBUG_BINARY_LOGPROBS"):
        return
    preview = ", ".join(f"{t.token!r}:{t.logprob:.3f}" for t in top[:10])
    print(f"    [posterior] missing Yes/No in top_logprobs: {preview}")


def _score_topk_logprobs(top) -> tuple[float, bool]:
    score, fallback = score_topk_pairs((str(item.token), item.logprob) for item in top)
    if fallback:
        _debug_top_logprobs(top)
    return score, fallback


def _sampling_kwargs(
    temperature: float,
    generation_config: GenerationSamplingConfig | None,
) -> tuple[dict, dict]:
    """Translate shared sampling config into standard and local-only API fields."""
    if generation_config is None:
        return {"temperature": temperature}, {}
    standard = {
        "temperature": generation_config.temperature,
        "top_p": generation_config.top_p,
    }
    if generation_config.presence_penalty is not None:
        standard["presence_penalty"] = generation_config.presence_penalty
    extended = {
        name: getattr(generation_config, name)
        for name in ("top_k", "min_p", "repetition_penalty")
        if getattr(generation_config, name) is not None
    }
    return standard, extended


class ApiBackend(Backend):
    """Chat Completions client; its semaphore bounds in-flight requests."""

    def __init__(self, model_id: str, provider: str, *, concurrency: int = 8,
                 enable_thinking: bool = False, reasoning_effort: str | None = None,
                 top_logprobs: int = 5, thinking_protocol: str | None = None,
                 supports_logprobs: bool = True, supports_json_schema: bool = False):
        cfg = API_PROVIDER_CONFIGS[provider]
        self.model_id = model_id
        self.provider = provider
        self.semaphore = asyncio.Semaphore(concurrency)
        self.reasoning_effort = reasoning_effort
        self.enable_thinking = enable_thinking
        self.top_logprobs = top_logprobs
        self.text_trial_retry_sweeps = cfg.get("text_trial_retry_sweeps", 0)
        self.text_trial_encoding = cfg.get("text_trial_encoding", "vector")
        self.supports_logprobs = supports_logprobs
        self.supports_json_schema = supports_json_schema
        self.thinking_protocol = thinking_protocol
        self._thinking_protocol = resolve_api_thinking_protocol(thinking_protocol)
        if provider == "local" and enable_thinking and self._thinking_protocol is None:
            raise ValueError(
                f"Local thinking for {model_id} requires an explicit thinking_protocol binding."
            )

        api_key = os.environ.get(cfg["api_key_env"], cfg.get("default_api_key"))
        if api_key is None:
            raise KeyError(cfg["api_key_env"])
        base_url = os.environ.get(cfg["base_url_env"], cfg["base_url"]) if "base_url_env" in cfg else cfg["base_url"]
        if cfg.get("require_base_url") and not base_url:
            raise KeyError(cfg["base_url_env"])
        client_kwargs = {"api_key": api_key, "base_url": base_url}
        if "request_timeout_seconds" in cfg:
            client_kwargs["timeout"] = cfg["request_timeout_seconds"]
        if "sdk_max_retries" in cfg:
            client_kwargs["max_retries"] = cfg["sdk_max_retries"]
        self.client = AsyncOpenAI(**client_kwargs)

    @property
    def is_reasoning(self) -> bool:
        """True when the call consumes reasoning tokens (anything except 'none')."""
        return self.reasoning_effort not in (None, "none")

    @property
    def minimum_thinking_headroom_tokens(self) -> int:
        if self.provider == "openrouter" and self.is_reasoning:
            return OPENROUTER_REASONING_MIN_TOKENS
        return 0

    def completion_token_limit(self, answer_tokens: int, thinking_budget: int) -> int:
        """Return the provider output cap while preserving answer headroom."""
        if self.provider != "openrouter" or not self.is_reasoning:
            return answer_tokens + thinking_budget
        reasoning_pct = OPENROUTER_REASONING_EFFORT_PCT.get(self.reasoning_effort)
        if reasoning_pct is None:
            raise ValueError(f"Unsupported OpenRouter reasoning effort: {self.reasoning_effort!r}")
        return max(
            answer_tokens + max(thinking_budget, OPENROUTER_REASONING_MIN_TOKENS),
            math.ceil(answer_tokens * 100 / (100 - reasoning_pct)),
        )

    def is_retryable_generation_error(self, exc: Exception) -> bool:
        if isinstance(exc, (APIConnectionError, APITimeoutError)):
            return True
        return isinstance(exc, APIStatusError) and (exc.status_code in {408, 409, 429} or exc.status_code >= 500)

    @property
    def uses_channel_thinking(self) -> bool:
        """Local channel-protocol thinking streams thought and answer in one call."""
        return self.provider == "local" and self.enable_thinking and self.thinking_protocol == "channel"

    @property
    def needs_two_stage_thinking(self) -> bool:
        """Local text-marker thinking needs separate thinking and answer budgets."""
        return self.provider == "local" and self.enable_thinking and self.thinking_protocol == "text"

    def _require_thinking_protocol(self) -> ThinkingProtocol:
        if self._thinking_protocol is None:
            raise RuntimeError("This backend call requires a configured thinking protocol.")
        return self._thinking_protocol

    def _local_extra_body(self, *, enable_thinking: bool | None,
                          guided_choice: list[str] | None = None,
                          sampling_fields: dict | None = None) -> dict | None:
        """Build vLLM/SGLang request extensions (no-op off the local provider).

        `enable_thinking=None` omits chat_template_kwargs entirely (the non-thinking
        scoring path); True/False pins the flag for thinking and answer-only paths.
        """
        if self.provider != "local":
            return None
        body: dict = {}
        if enable_thinking is not None:
            body["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
        if guided_choice is not None:
            body["guided_choice"] = list(guided_choice)
        if sampling_fields:
            body.update(sampling_fields)
        return body or None

    def _params(self, max_tokens: int, temperature: float, thinking_budget: int, *,
                logprobs: bool = False, guided_choice: list[str] | None = None,
                generation_config: GenerationSamplingConfig | None = None,
                response_json_schema: dict | None = None) -> dict:
        """Build chat.completions kwargs with answer-only and thinking budgets."""
        sampling, local_sampling = _sampling_kwargs(temperature, generation_config)
        p: dict = {"model": self.model_id}
        if self.provider == "openai" and self.reasoning_effort is not None:
            p["reasoning_effort"] = self.reasoning_effort
            p["max_completion_tokens"] = (
                max_tokens + thinking_budget
                if self.is_reasoning
                else max(max_tokens, OPENAI_NON_REASONING_MIN_COMPLETION_TOKENS)
            )
            if not self.is_reasoning:
                p.update(sampling)
        elif self.provider == "openrouter" and self.is_reasoning:
            p["max_tokens"] = self.completion_token_limit(max_tokens, thinking_budget)
            p.update(sampling)
        elif self.uses_channel_thinking:
            p["max_tokens"] = max_tokens + thinking_budget
            p.update(sampling)
        else:
            p["max_tokens"] = max_tokens
            p.update(sampling)
        if logprobs:
            p.update(logprobs=True, top_logprobs=self.top_logprobs)
        if response_json_schema is not None:
            if not self.supports_json_schema:
                raise ValueError(f"{self.model_id} does not support JSON-schema responses")
            p["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_response",
                    "strict": True,
                    "schema": response_json_schema,
                },
            }
        extra = self._local_extra_body(
            enable_thinking=True if self.enable_thinking else None,
            guided_choice=guided_choice,
            sampling_fields=local_sampling,
        ) or {}
        if self.provider == "openrouter" and self.is_reasoning:
            extra["reasoning"] = {"effort": self.reasoning_effort}
        if extra:
            p["extra_body"] = extra
        return p

    @staticmethod
    def _stage1_thinking_text(message) -> str:
        reasoning = getattr(message, "reasoning_content", None)
        if isinstance(reasoning, str) and reasoning:
            return reasoning
        return str(message.content or "")

    async def _run_thinking_stage(self, messages, thinking_budget: int,
                                  temperature: float, *, stop: bool,
                                  generation_config: GenerationSamplingConfig | None = None) -> str:
        """Run one thinking pass and return the normalized text-marker block.

        Caller owns any semaphore scope. `stop` force-closes at the text close
        marker (the two-stage text-protocol path).
        """
        sampling, local_sampling = _sampling_kwargs(temperature, generation_config)
        kwargs: dict = {
            "model": self.model_id,
            "max_tokens": thinking_budget,
            "extra_body": self._local_extra_body(
                enable_thinking=True, sampling_fields=local_sampling,
            ),
        }
        kwargs.update(sampling)
        if stop:
            kwargs["stop"] = [self._require_thinking_protocol().end_text]
        response = await self.client.chat.completions.create(messages=messages, **kwargs)
        return self._require_thinking_protocol().format_block(
            self._stage1_thinking_text(response.choices[0].message),
        )

    async def _two_stage_call(self, messages, max_tokens, temperature, thinking_budget, *,
                              logprobs=False, guided_choice=None, generation_config=None):
        """Run local text-marker thinking, then resume with an answer-only call."""
        thinking_block = await self._run_thinking_stage(
            messages, thinking_budget, temperature, stop=True,
            generation_config=generation_config,
        )

        stage2_extra: dict = {
            "continue_final_message": True,
            "add_generation_prompt": False,
        }
        sampling, local_sampling = _sampling_kwargs(temperature, generation_config)
        if guided_choice is not None:
            stage2_extra["guided_choice"] = list(guided_choice)
        stage2_extra.update(local_sampling)
        stage2_kwargs: dict = {
            "model": self.model_id,
            "max_tokens": max_tokens,
            "extra_body": stage2_extra,
        }
        stage2_kwargs.update(sampling)
        if logprobs:
            stage2_kwargs["logprobs"] = True
            stage2_kwargs["top_logprobs"] = self.top_logprobs
        cont_messages = [*messages, {"role": "assistant", "content": thinking_block}]
        return await self.client.chat.completions.create(messages=cont_messages, **stage2_kwargs)

    async def _shared_thinking_message(self, messages, thinking_budget: int) -> dict:
        async with self.semaphore:
            thinking_block = await self._run_thinking_stage(
                messages, thinking_budget, 0.0, stop=self.needs_two_stage_thinking,
            )
        return {"role": "assistant", "content": thinking_block}

    def _score_channel_logprobs(self, content) -> tuple[float, bool]:
        """Score the first answer slot after the channel thought-close marker.

        Reasoning-parser output emits thinking then the configured close boundary; the
        visible Yes/No follows. Scan past the marker for the first binary slot.
        """
        close_marker = self._require_thinking_protocol().end_text
        start = next((i + 1 for i, slot in enumerate(content) if slot.token == close_marker), None)
        if start is None:
            _debug_top_logprobs(content[0].top_logprobs)
            return 0.5, True
        for slot in content[start:]:
            score, fallback = score_topk_pairs((str(item.token), item.logprob) for item in slot.top_logprobs)
            if not fallback:
                return score, False
        debug_slot = content[start] if start < len(content) else content[0]
        _debug_top_logprobs(debug_slot.top_logprobs)
        return 0.5, True

    async def _score_one_answer_only(self, messages: list[dict]) -> tuple[float, bool]:
        kwargs = {
            "model": self.model_id,
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": True,
            "top_logprobs": self.top_logprobs,
        }
        extra = self._local_extra_body(enable_thinking=False, guided_choice=BINARY_CHOICES)
        if extra is not None:
            kwargs["extra_body"] = extra
        async with self.semaphore:
            response = await self.client.chat.completions.create(messages=messages, **kwargs)
        content = response.choices[0].logprobs.content
        if not content:
            return 0.5, True
        return _score_topk_logprobs(content[0].top_logprobs)

    async def generate(self, messages, max_tokens=1, temperature=0.0, *,
                       thinking_budget: int | None = None, generation_config=None,
                       response_json_schema: dict | None = None):
        thinking_budget = self._effective_thinking_budget(thinking_budget)
        async with self.semaphore:
            if self.needs_two_stage_thinking:
                response = await self._two_stage_call(
                    messages, max_tokens, temperature, thinking_budget,
                    generation_config=generation_config,
                )
            else:
                response = await self.client.chat.completions.create(
                    messages=messages,
                    **self._params(
                        max_tokens, temperature, thinking_budget,
                        generation_config=generation_config,
                        response_json_schema=response_json_schema,
                    ),
                )
        text = response.choices[0].message.content or ""
        return GenerateResult(text=text.strip())

    async def _score_one(self, messages: list[dict], thinking_budget: int | None = None) -> tuple[float, bool]:
        """Single API call with logprobs; returns (p_yes, fell_back)."""
        thinking_budget = self._effective_thinking_budget(thinking_budget)
        if self.needs_two_stage_thinking:
            async with self.semaphore:
                response = await self._two_stage_call(
                    messages, max_tokens=1, temperature=0.0, thinking_budget=thinking_budget,
                    logprobs=True, guided_choice=BINARY_CHOICES,
                )
            content = response.choices[0].logprobs.content if response.choices[0].logprobs else None
            if not content:
                return 0.5, True
            return _score_topk_logprobs(content[0].top_logprobs)

        async with self.semaphore:
            response = await self.client.chat.completions.create(
                messages=messages,
                **self._params(
                    max_tokens=1,
                    temperature=0.0,
                    thinking_budget=thinking_budget,
                    logprobs=True,
                    guided_choice=BINARY_CHOICES if self.provider == "local" else None,
                ),
            )
        content = response.choices[0].logprobs.content
        if self.uses_channel_thinking:
            return self._score_channel_logprobs(content)
        return _score_topk_logprobs(content[0].top_logprobs)

    async def score_binary(
        self,
        messages_list: list[list[dict]],
        *,
        thinking_budget: int | None = None,
    ) -> tuple[np.ndarray, int]:
        """Concurrent logprob-based scoring returning (p_yes, n_fallback)."""
        return await warm_cache_then_gather(
            lambda messages: self._score_one(messages, thinking_budget),
            messages_list,
        )

    async def _score_binary_after_shared_thinking(
        self,
        continued_messages: list[list[dict]],
    ) -> tuple[np.ndarray, int]:
        results = await asyncio.gather(*[self._score_one_answer_only(m) for m in continued_messages])
        return assemble_binary_scores(results)
