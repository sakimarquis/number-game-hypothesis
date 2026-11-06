"""Tinker backend using cookbook renderers and teacher-forced logprobs."""

import asyncio
import math

import numpy as np

from backends.base import BINARY_VARIANTS, Backend, GenerateResult, assemble_binary_scores
from backends.thinking_codec import text_thinking_protocol
from configs.settings import GREEDY, GenerationSamplingConfig


# Teacher-forced candidates also cover space-prefixed tokenizations of the cased forms.
CANDIDATE_VARIANTS = {label: (*variants, *(f" {text}" for text in variants[:2])) for label, variants in BINARY_VARIANTS.items()}


def _is_invalid_jwt_error(exc: Exception) -> bool:
    return exc.__class__.__name__ == "AuthenticationError" and "Invalid JWT" in str(exc)


def _parsed_message(renderer, tokens: list[int]) -> dict:
    """Cookbook renderers parse sampled tokens into (message, parse_ok)."""
    message, _parse_ok = renderer.parse_response(tokens)
    return message


def _message_parts(message: dict, part_type: str) -> list[str]:
    """Text or thinking parts of a parsed message; plain-string content may inline `<think>`."""
    content = message["content"]
    if not isinstance(content, str):
        return [str(part.get(part_type, "")) for part in content if part.get("type") == part_type]
    if part_type == "text":
        return [content]
    protocol = text_thinking_protocol()
    _before, found, rest = content.partition(protocol.start_text)
    thought, closed, _answer = rest.partition(protocol.end_text)
    return [thought] if found and closed else []


def _logsumexp(values: list[float]) -> float | None:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return None
    peak = max(finite)
    return peak + math.log(sum(math.exp(v - peak) for v in finite))


def _score_candidate_logprobs(logprobs: dict[str, list[float]]) -> tuple[float, bool]:
    """Normalize Yes against No; falls back to 0.5 unless both labels were scored."""
    yes = _logsumexp(logprobs["yes"])
    no = _logsumexp(logprobs["no"])
    if yes is None or no is None:
        return 0.5, True
    return math.exp(yes) / (math.exp(yes) + math.exp(no)), False


class TinkerBackend(Backend):
    """Adapter from number_game's backend contract to Tinker's sampling API.

    Concurrency is bounded by one backend-owned semaphore shared across all
    requests, letting Tinker batch on the GPU without chunk barriers.

    Binary scoring teacher-forces Yes/No candidate tokens with
    `compute_logprobs_async`. This keeps posterior semantics independent of
    whether those tokens appear in an unconstrained top-k next-token list.
    """

    supports_per_target_thinking = False

    def __init__(
        self,
        model_id: str,
        renderer_name: str | None,
        thinking_renderer_name: str | None = None,
        enable_thinking: bool = False,
        concurrency: int = 8,
        tokenizer_trust_remote_code: bool = False,
    ):
        import tinker
        from tinker_cookbook.renderers import get_renderer

        self.enable_thinking = enable_thinking
        selected_renderer = thinking_renderer_name if enable_thinking else renderer_name
        if selected_renderer is None:
            raise ValueError("Tinker backend requires renderer, and --thinking requires thinking_renderer.")

        self.model_id = model_id
        self.answer_renderer_name = renderer_name
        self.renderer_name = selected_renderer
        self.semaphore = asyncio.Semaphore(concurrency)
        self.types = tinker.types
        self._service_client_factory = tinker.ServiceClient
        self._renderer_factory = get_renderer
        self._tokenizer_trust_remote_code = tokenizer_trust_remote_code
        self._connect()

    def _connect(self):
        """Create a fresh Tinker service/sampling client, tokenizer, and renderers."""
        self.service_client = self._service_client_factory()
        self.sampling_client = self.service_client.create_sampling_client(base_model=self.model_id)
        if self._tokenizer_trust_remote_code:
            from transformers.models.auto.tokenization_auto import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, fast=True, trust_remote_code=True)
        else:
            self.tokenizer = self.sampling_client.get_tokenizer()
        self.renderer = self._renderer_factory(
            self.renderer_name, self.tokenizer, image_processor=None, model_name=self.model_id,
        )
        self.answer_renderer = self.renderer
        if self.enable_thinking and self.answer_renderer_name is not None:
            self.answer_renderer = self._renderer_factory(
                self.answer_renderer_name, self.tokenizer, image_processor=None, model_name=self.model_id,
            )
        self._candidates = {label: self._candidate_token_variants(label) for label in CANDIDATE_VARIANTS}

    async def _with_invalid_jwt_retry(self, coro_factory):
        """Run an async unit; on Invalid JWT, reconnect once and retry the whole unit."""
        try:
            return await coro_factory()
        except Exception as exc:
            if not _is_invalid_jwt_error(exc):
                raise
            self._connect()
            return await coro_factory()

    def _prompt(self, messages, renderer=None):
        return (renderer or self.renderer).build_generation_prompt(messages)

    def _sampling_params(self, max_tokens: int, thinking_budget: int, generation_config: GenerationSamplingConfig):
        total_tokens = max_tokens + thinking_budget if self.enable_thinking else max_tokens
        kwargs = {"max_tokens": total_tokens, "temperature": generation_config.temperature}
        for name in ("top_p", "top_k"):
            if getattr(generation_config, name) is not None:
                kwargs[name] = getattr(generation_config, name)
        stop = self.renderer.get_stop_sequences()
        if stop:
            kwargs["stop"] = stop
        return self.types.SamplingParams(**kwargs)

    async def _sample_one(self, messages, max_tokens, thinking_budget, generation_config) -> GenerateResult:
        async with self.semaphore:
            response = await self.sampling_client.sample_async(
                prompt=self._prompt(messages),
                num_samples=1,
                sampling_params=self._sampling_params(max_tokens, thinking_budget, generation_config),
            )
        sequence = response.sequences[0]
        tokens = list(sequence.tokens)
        message = _parsed_message(self.renderer, tokens)
        truncated = sequence.stop_reason == "length"
        return GenerateResult(
            text="".join(_message_parts(message, "text")).strip(),
            thinking_tokens=self._thinking_tokens(message, tokens, truncated) if self.enable_thinking else None,
            truncated=truncated,
        )

    def _thinking_tokens(self, message: dict, tokens: list[int], truncated: bool) -> int:
        """Re-encoded length of the parsed thought; a response cut before its thought closed is all thinking."""
        thoughts = _message_parts(message, "thinking")
        if thoughts:
            return len(self.tokenizer.encode("".join(thoughts), add_special_tokens=False))
        return len(tokens) if truncated else 0

    async def generate(self, messages_list, max_tokens=1,
                       *, thinking_budget: int | None = None,
                       generation_config=GREEDY, response_json_schema=None, batch_key=None):
        thinking_budget = self._effective_thinking_budget(thinking_budget)
        return await self._with_invalid_jwt_retry(
            lambda: asyncio.gather(*[
                self._sample_one(messages, max_tokens, thinking_budget, generation_config)
                for messages in messages_list
            ])
        )

    def _candidate_token_variants(self, label: str) -> list[list[int]]:
        """Distinct token sequences for one label's surface variants."""
        seen: dict[tuple[int, ...], None] = {}
        for text in CANDIDATE_VARIANTS[label]:
            tokens = tuple(self.tokenizer.encode(text, add_special_tokens=False))
            if tokens:
                seen.setdefault(tokens)
        return [list(tokens) for tokens in seen]

    async def _candidate_logprob(self, prompt_tokens: list[int], candidate: list[int]) -> float | None:
        async with self.semaphore:
            logprobs = await self.sampling_client.compute_logprobs_async(
                self.types.ModelInput.from_ints(prompt_tokens + candidate)
            )
        suffix = logprobs[-len(candidate):]
        if any(v is None for v in suffix):
            return None
        return float(sum(suffix))

    async def _score_binary_once(
        self,
        messages_list: list[list[dict]],
        renderer=None,
    ) -> tuple[np.ndarray, int]:
        rows = [{"yes": [], "no": []} for _ in messages_list]
        targets: list[tuple[int, str]] = []
        coros = []
        for row_idx, messages in enumerate(messages_list):
            prompt_tokens = list(self._prompt(messages, renderer=renderer).to_ints())
            for label, candidates in self._candidates.items():
                for candidate in candidates:
                    targets.append((row_idx, label))
                    coros.append(self._candidate_logprob(prompt_tokens, candidate))

        candidate_logprobs = await asyncio.gather(*coros)
        for (row_idx, label), logprob in zip(targets, candidate_logprobs):
            if logprob is not None:
                rows[row_idx][label].append(logprob)
        return assemble_binary_scores([_score_candidate_logprobs(row) for row in rows])

    async def _shared_thinking_message(self, messages: list[dict], thinking_budget: int) -> tuple[dict, dict]:
        async with self.semaphore:
            response = await self.sampling_client.sample_async(
                prompt=self._prompt(messages),
                num_samples=1,
                sampling_params=self._sampling_params(0, thinking_budget, GREEDY),
            )
        sequence = response.sequences[0]
        tokens = list(sequence.tokens)
        thoughts = _message_parts(_parsed_message(self.renderer, tokens), "thinking")
        thinking_text = "\n".join(thoughts).strip() if thoughts else str(self.tokenizer.decode(tokens)).strip()
        # A thought cut at the budget is closed by format_block.
        record = GenerateResult("", thinking_tokens=len(tokens), forced_close=sequence.stop_reason == "length").thinking_record()
        return {"role": "assistant", "content": text_thinking_protocol().format_block(thinking_text)}, record

    async def _score_binary_after_shared_thinking(
        self,
        continued_messages: list[list[dict]],
    ) -> tuple[np.ndarray, int]:
        return await self._score_binary_once(continued_messages, renderer=self.answer_renderer)

    async def score_binary(self, messages_list: list[list[dict]],
                           *, thinking_budget: int | None = None) -> tuple[np.ndarray, int]:
        # Candidate logprobs do not generate, so no thinking budget is consumed.
        return await self._with_invalid_jwt_retry(lambda: self._score_binary_once(messages_list))

    async def score_binary_with_shared_thinking(
        self,
        thinking_messages: list[dict],
        messages_list: list[list[dict]],
        *,
        thinking_budget: int | None = None,
    ) -> tuple[np.ndarray, int, dict]:
        return await self._with_invalid_jwt_retry(
            lambda: super(TinkerBackend, self).score_binary_with_shared_thinking(
                thinking_messages, messages_list, thinking_budget=thinking_budget,
            )
        )
