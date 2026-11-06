"""Native Gemini Developer API backend using the Google Gen AI SDK."""

import asyncio
import os
import uuid

from google import genai
from google.genai import types

from backends.base import Backend, GenerateResult
from backends.gemini_batch import GeminiBatchExecutor
from configs.layout import condition_batch_key
from configs.settings import GREEDY, GenerationSamplingConfig


MEDIUM_THINKING_HEADROOM_TOKENS = 2048
STRUCTURED_RECOVERY_THINKING_HEADROOM_TOKENS = 8192
BATCH_JOB_CONCURRENCY = 4


class GoogleCloudBackend(Backend):
    """Gemini generation through the native Gemini Developer API (no logprob path)."""

    minimum_thinking_headroom_tokens = MEDIUM_THINKING_HEADROOM_TOKENS
    structured_recovery_thinking_headroom_tokens = STRUCTURED_RECOVERY_THINKING_HEADROOM_TOKENS

    def __init__(
        self,
        model_id: str,
        *,
        api_key_env: str,
        concurrency: int = 8,
        enable_thinking: bool = False,
        execution: str = "online",
        batch_priority: int = 0,
        batch_root: str | None = None,
        supports_json_schema: bool = False,
    ):
        if execution not in {"batch", "online"}:
            raise ValueError(f"Unknown Gemini execution mode: {execution!r}")
        if execution == "batch" and batch_root is None:
            raise ValueError("batch_root is required for Gemini Batch execution")
        api_key = os.environ.get(api_key_env)
        if api_key is None:
            raise KeyError(api_key_env)

        self.client = genai.Client(api_key=api_key)
        self.model_id = model_id
        self.enable_thinking = enable_thinking
        self.supports_json_schema = supports_json_schema
        self.batch_api_enabled = execution == "batch"
        self.semaphore = asyncio.Semaphore(concurrency)
        self.batch_job_concurrency = min(concurrency, BATCH_JOB_CONCURRENCY)
        self.batch_semaphore = asyncio.Semaphore(self.batch_job_concurrency)
        self.batch = (
            GeminiBatchExecutor(self.client, model_id, batch_root, priority=batch_priority)
            if self.batch_api_enabled
            else None
        )

    @staticmethod
    def _split_messages(messages: list[dict]) -> tuple[str | None, list]:
        system_parts: list[str] = []
        grouped: list[tuple[str, list[str]]] = []
        for message in messages:
            role = message["role"]
            content = message["content"]
            if role == "system":
                system_parts.append(content)
                continue
            google_role = {"user": "user", "assistant": "model"}[role]
            if grouped and grouped[-1][0] == google_role:
                grouped[-1][1].append(content)
            else:
                grouped.append((google_role, [content]))

        contents = [
            types.Content(role=role, parts=[types.Part.from_text(text=text) for text in texts])
            for role, texts in grouped
        ]
        return "\n\n".join(system_parts) or None, contents

    def _generation_config(
        self,
        *,
        max_output_tokens: int,
        generation_config: GenerationSamplingConfig,
        response_json_schema: dict | None,
    ):
        # Gemini 3.x rejects explicit temperature/top-p/top-k controls, so the
        # shared sampling fields are deliberately not forwarded.
        unsupported = [
            name for name in ("min_p", "presence_penalty", "repetition_penalty")
            if getattr(generation_config, name) is not None
        ]
        if unsupported:
            raise ValueError(f"google_cloud does not support generation parameters: {', '.join(unsupported)}")
        kwargs = {
            "thinking_config": types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.MEDIUM if self.enable_thinking else types.ThinkingLevel.MINIMAL,
            ),
            "max_output_tokens": max_output_tokens,
        }
        if response_json_schema is not None:
            kwargs.update(response_mime_type="application/json", response_json_schema=response_json_schema)
        return types.GenerateContentConfig(**kwargs)

    def _inlined_request(self, messages, config):
        system_instruction, contents = self._split_messages(messages)
        if system_instruction is not None:
            config.system_instruction = system_instruction
        return types.InlinedRequest(contents=contents, config=config)

    async def _online_request(self, request):
        async with self.semaphore:
            return await self.client.aio.models.generate_content(
                model=self.model_id,
                contents=request.contents,
                config=request.config,
            )

    @staticmethod
    def _response_fields(response) -> dict:
        """JSON-safe GenerateResult fields: visible answer text (thought summaries excluded) plus thinking usage."""
        if not response.candidates:
            raise RuntimeError("Gemini returned no candidates.")
        candidate = response.candidates[0]
        parts = getattr(candidate.content, "parts", None) or []
        text = "".join(
            str(part.text) for part in parts if getattr(part, "text", None) and not getattr(part, "thought", False)
        ).strip()
        usage = response.usage_metadata
        return {
            "text": text,
            "thinking_tokens": usage.thoughts_token_count if usage else None,
            "truncated": candidate.finish_reason == types.FinishReason.MAX_TOKENS,
        }

    async def generate(
        self,
        messages_list: list[list[dict]],
        max_tokens: int = 1,
        *,
        thinking_budget: int | None = None,
        generation_config: GenerationSamplingConfig = GREEDY,
        response_json_schema: dict | None = None,
        batch_key: str | None = None,
    ) -> list[GenerateResult]:
        if not messages_list:
            return []
        thinking_budget = self._effective_thinking_budget(thinking_budget)
        # Gemini 3.x thinking levels are dynamic, so MEDIUM reserves at least the
        # documented headroom on top of the answer-only budget.
        max_output_tokens = (
            max_tokens + max(thinking_budget, self.minimum_thinking_headroom_tokens)
            if self.enable_thinking
            else max_tokens
        )
        requests = [
            self._inlined_request(
                messages,
                self._generation_config(
                    max_output_tokens=max_output_tokens,
                    generation_config=generation_config,
                    response_json_schema=response_json_schema,
                ),
            )
            for messages in messages_list
        ]

        if self.batch_api_enabled:
            for index, request in enumerate(requests):
                request.metadata = {"key": str(index)}
            async with self.batch_semaphore:
                fields = await self.batch.run(requests, batch_key=batch_key or uuid.uuid4().hex, decode=self._response_fields)
        else:
            responses = await asyncio.gather(*[self._online_request(request) for request in requests])
            fields = [self._response_fields(response) for response in responses]
        return [GenerateResult(**row) for row in fields]

    async def release_batch_sidecars(self, condition) -> None:
        if self.batch is not None:
            # Every batch key of a condition extends its eval-mode key (see scoring/eval_modes.py).
            await self.batch.cleanup(condition_batch_key(condition, condition.eval_mode))

